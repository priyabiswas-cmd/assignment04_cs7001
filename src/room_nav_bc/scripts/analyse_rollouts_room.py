#!/usr/bin/env python3
"""
Diagnose BC rollout failures: averaging vs out-of-distribution (OOD).

    ros2 run room_nav_bc bc_policy_node_room.py --policy policy_room.pt \
        --starts test --trials 5 --steps-csv logs/test_steps.csv
    ros2 run room_nav_bc analyse_rollouts_room.py \
        --steps-csv logs/test_steps.csv --dataset dataset_room.npz

For every logged control step it asks three questions:

1. WHICH MODE did the policy's turn rate match?  At the same state, the
   expert's left-route and right-route controllers each give a turn rate
   (wL, wR). The policy's turn rate (pred_w) is then classed as:
     committed-left / committed-right  - it is close to one route's rate
     averaging                         - it sits near (wL + wR) / 2 and not
                                         near either route (the mean-seeking
                                         failure this exercise is about)
     other                             - neither
     indeterminate                     - the two routes give nearly the same
                                         turn rate here, so the step can't
                                         tell the modes apart (e.g. far from
                                         the island)

2. HOW FAR FROM TRAINING DATA is the state?  Nearest-neighbour distance (in
   the same normalised odometry features the policy was trained on) to the
   training samples, compared with how far held-out validation samples are
   from training. Above that calibrated threshold = OOD.
   NOTE: this uses odometry features only, not the camera image.

3. WHAT HAPPENED JUST BEFORE A COLLISION?  The last --window steps of every
   collided trial are reported separately from all steps, since a collision
   is what we are trying to explain.

Verdict: a step counts as averaging / committed / OOD when its condition
holds. The report gives the fraction of pre-collision steps in each class.
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_expert_room as ver   # stubs ROS, exposes expert + clearances

expert = ver.expert


def load_steps(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('trial', 'start_x', 'start_y', 'start_yaw', 'x', 'y', 'yaw',
                  'odom_dx', 'odom_dy', 'odom_yaw', 'pred_v', 'pred_w'):
            r[k] = float(r[k])
        r['trial'] = int(r['trial'])
        r['collided'] = str(r.get('collided', '')).lower() == 'true'
    return rows


def expert_turn_rates(row, obstacles, goal, args, side):
    """Expert (v, w) at this state if it were following the `side` route
    from this trial's start - the same controller as expert_demo_room."""
    wps = [(row['start_x'], row['start_y'])] + expert.build_waypoints(
        side, obstacles, goal, clearance=args.clearance,
        transition_margin=args.transition_margin)
    follower = expert.PathFollower(wps, lookahead=args.lookahead)
    tx, ty, _ = follower.target_and_progress(row['x'], row['y'])
    return expert.steer_to_point(row['x'], row['y'], row['yaw'], (tx, ty),
                                 args.cruise_speed, args.k_heading)


def nn_distance(points, train_pts, chunk=512):
    """Distance from each point to its nearest training point (brute force)."""
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        d = np.linalg.norm(points[i:i + chunk, None, :] - train_pts[None],
                           axis=2)
        out[i:i + chunk] = d.min(axis=1)
    return out


def classify(pred_w, wL, wR, tol_frac=0.25):
    """Label by ROUTE NAME (left/right), not by the sign of the turn rate."""
    span = abs(wL - wR)
    if span < 0.5:
        return 'indeterminate'
    tol = tol_frac * span
    dL, dR = abs(pred_w - wL), abs(pred_w - wR)
    mid = 0.5 * (wL + wR)
    if dL <= tol and dL < dR:
        return 'committed-left'
    if dR <= tol and dR < dL:
        return 'committed-right'
    if abs(pred_w - mid) <= tol:
        return 'averaging'
    return 'other'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--steps-csv', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--config', default=None)
    p.add_argument('--window', type=int, default=10)
    p.add_argument('--train-split', default='train',
                   help='which dataset arrays to use as the training reference')
    cli = p.parse_args()

    config = cli.config or expert.default_config_path()
    cfg = yaml.safe_load(open(config))
    obstacles, goal = cfg['obstacles'], cfg['goal']
    xargs = expert.build_parser().parse_args([])   # same defaults as live node

    rows = load_steps(cli.steps_csv)
    if not rows:
        raise SystemExit("no steps in that log")

    # ---- 1. mode classification --------------------------------------------
    for r in rows:
        wl = expert_turn_rates(r, obstacles, goal, xargs, 'left')[1]
        wr = expert_turn_rates(r, obstacles, goal, xargs, 'right')[1]
        r['wL'], r['wR'] = wl, wr
        r['mode'] = classify(r['pred_w'], wl, wr)

    # ---- 2. OOD vs training data ------------------------------------------
    d = np.load(cli.dataset, allow_pickle=True)
    odom_mean, odom_std = d['odom_mean'], d['odom_std']
    train_pts = (d['X_train_odom'] if cli.train_split == 'train'
                 else d['X_val_odom'])
    val_pts = d['X_val_odom']
    norm = lambda a: (a - odom_mean) / odom_std
    # same feature order as the dataset: (dx, dy, sin, cos)
    feats = np.array([[r['odom_dx'], r['odom_dy'],
                       math.sin(r['odom_yaw']), math.cos(r['odom_yaw'])]
                      for r in rows])
    nn_rollout = nn_distance(norm(feats), train_pts)
    nn_val = nn_distance(val_pts, train_pts)
    ood_thresh = float(np.percentile(nn_val, 95))
    for r, dist in zip(rows, nn_rollout):
        r['nn'] = float(dist)
        r['ood'] = dist > ood_thresh

    # ---- 3. report ----------------------------------------------------------
    def tally(subset, key):
        c = Counter(key(r) for r in subset)
        n = sum(c.values()) or 1
        return {k: (v, 100.0 * v / n) for k, v in c.items()}

    # A collision does not end a trial, so "the last steps" are usually
    # AFTER the crash. The window is the steps leading up to first contact
    # (first step with clearance < 0), per collided trial.
    collided_trials = {r['trial'] for r in rows if r['collided']}
    pre = []
    for t in collided_trials:
        steps = [r for r in rows if r['trial'] == t]
        contact = next((i for i, r in enumerate(steps)
                        if float(r.get('clearance', 0) or 0) < 0.0), None)
        if contact is None:
            contact = len(steps)          # no per-step data: fall back
        pre += steps[max(0, contact - cli.window):contact]

    print(f"\nOOD threshold (95th pct of validation-to-train NN distance): "
          f"{ood_thresh:.3f}   [{len(val_pts)} val states, "
          f"{len(train_pts)} train states]")
    print(f"steps analysed: {len(rows)}   collided trials: "
          f"{len(collided_trials)}\n")

    def show(title, subset):
        print(f" {title}  ({len(subset)} steps)")
        for k, (v, pct) in sorted(tally(subset, lambda r: r['mode']).items(),
                                  key=lambda kv: -kv[1][0]):
            print(f"    mode  {k:18s} {v:5d}  {pct:5.1f}%")
        ood_n = sum(r['ood'] for r in subset)
        print(f"    OOD   {'':18s} {ood_n:5d}  "
              f"{100.0 * ood_n / max(len(subset), 1):5.1f}%")
        print()

    show("ALL STEPS", rows)
    show(f"{cli.window} STEPS BEFORE FIRST CONTACT, PER COLLIDED TRIAL", pre)

    # verdict on the pre-collision window
    if pre:
        n = len(pre)
        avg = sum(r['mode'] == 'averaging' for r in pre) / n
        com = sum(r['mode'].startswith('committed') for r in pre) / n
        ood = sum(r['ood'] for r in pre) / n
        print(" verdict (pre-collision window):")
        if ood > 0.5:
            print(f"   OOD dominates ({100*ood:.0f}% of steps are far from "
                  "training data) -> distribution shift is the main story.")
        if avg > 0.5:
            print(f"   averaging dominates ({100*avg:.0f}% of steps sit "
                  "between the two routes) -> mean-seeking failure.")
        if com > 0.5:
            print(f"   the policy commits to one route ({100*com:.0f}%) -> "
                  "averaging is NOT the cause; it's a single wrong choice.")
        if max(ood, avg, com) <= 0.5:
            print("   no single cause dominates - look at the per-step table.")
        print()
        print(f"   note: the 'committed' label here means the policy's turn "
              "rate matched one route,\n         not that the chosen route "
              "was the one it crashed into.")
