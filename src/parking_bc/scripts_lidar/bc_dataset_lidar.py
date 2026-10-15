#!/usr/bin/env python3
"""
Turn odometry+lidar episodes (recorded with record_demo_lidar.py) into a
training set - the lidar/odometry counterpart of scripts/bc_dataset.py.

    python3 scripts_lidar/bc_dataset_lidar.py \
        --data-dir data_lidar --out dataset_lidar.npz

THE OBSERVATION FORMAT IS UNCHANGED from the original dataset:

    dx, dy          position relative to the bay centre
    sin(yaw), cos(yaw)
    (+ v_prev, w_prev with --history)

The only thing that's different is where (dx, dy) come from. The
original dataset used a bay centre handed over by a config constant.
This one has NO privileged knowledge of where the bay is: for each
episode, lidar_gap.find_gap_local() looks for a near-far-near pattern
(block / open space / block) in the recorded scans, GapEstimator fuses
those detections across the episode using the recorded ODOMETRY pose,
and the resulting sensed bay position is used as the reference instead.

This is done with the full episode in hand (hindsight), exactly like the
original script gets to use its constant for every sample of an episode
uniformly - the bay does not move mid-manoeuvre, so one best estimate per
episode is the right target, not a wobblier one that changes sample to
sample. bc_policy_node_lidar.py runs the SAME detection code but
causally, one scan at a time, since at run time there is no hindsight.

An episode where no clean near-far-near pattern ever appears (robot never
got a view of both blocks) is skipped, not silently patched with the
ground truth - if the robot couldn't have perceived the slot, it isn't
honest training data for a sensor-only policy.

THE SPLIT IS BY EPISODE, NOT BY SAMPLE - see scripts/bc_dataset.py for why.
"""

import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_gap import DEFAULTS, GapEstimator, find_gap_local, local_to_frame


def load_episode(path):
    with open(path, 'rb') as f:
        ep = pickle.load(f)
    return ep['samples'], ep.get('meta', {})


def detect_bay(samples, det_args):
    """Run the causal gap detector over one whole episode -> (bx, by) in
    the odom frame, or None if it never found a clean detection."""
    gap = GapEstimator(lock_count=det_args.lock_count,
                       lock_tol=det_args.lock_tol)
    for s in samples:
        local = find_gap_local(
            s['scan_ranges'], s['scan_angle_min'], s['scan_angle_increment'],
            near_thresh=det_args.near_thresh, min_run=det_args.min_run,
            fov_deg=det_args.fov_deg)
        if local is not None:
            frame_xy = local_to_frame(local[0], local[1],
                                      s['odom_x'], s['odom_y'], s['odom_yaw'])
            gap.update(frame_xy)
    return gap.estimate()


def episode_arrays(samples, bay_xy, use_history):
    """-> obs (T, D), act (T, 2) for one episode, dx/dy relative to the
    SENSED bay_xy (odom frame) instead of a config constant."""
    bx, by = bay_xy
    obs, act = [], []
    v_prev, w_prev = 0.0, 0.0
    for s in samples:
        row = [s['odom_x'] - bx, s['odom_y'] - by,
               math.sin(s['odom_yaw']), math.cos(s['odom_yaw'])]
        if use_history:
            row += [v_prev, w_prev]
        obs.append(row)
        act.append([s['v'], s['w']])
        v_prev, w_prev = s['v'], s['w']
    return np.asarray(obs, np.float32), np.asarray(act, np.float32)


def chunk_actions(act, k):
    T = len(act)
    out = np.empty((T, k, 2), np.float32)
    for i in range(T):
        for j in range(k):
            out[i, j] = act[min(i + j, T - 1)]
    return out


def build(args):
    paths = sorted(glob.glob(os.path.join(args.data_dir, '*.p')))
    if not paths:
        raise SystemExit(f"no episodes found in {args.data_dir}/")

    episodes, skipped_short, skipped_nodetect = [], 0, 0
    for p in paths:
        samples, meta = load_episode(p)
        if len(samples) < args.min_samples:
            print(f"  skipping {os.path.basename(p)}: "
                  f"only {len(samples)} samples")
            skipped_short += 1
            continue
        if 'scan_ranges' not in samples[0] or 'odom_x' not in samples[0]:
            raise SystemExit(
                f"{p} has no odom/scan fields - was it recorded with "
                "record_demo_lidar.py, not the original record_demo.py?")

        bay_xy = detect_bay(samples, args)
        if bay_xy is None:
            print(f"  skipping {os.path.basename(p)}: lidar never saw a "
                  "clean gap in this episode")
            skipped_nodetect += 1
            continue

        obs, act = episode_arrays(samples, bay_xy, args.history)
        episodes.append(dict(
            name=os.path.basename(p),
            label=meta.get('start_label', '?'),
            obs=obs, act=act,
            final=(samples[-1]['odom_x'] - bay_xy[0],
                   samples[-1]['odom_y'] - bay_xy[1]),
            bay_sensed=bay_xy,
            # /odom and the Gazebo world frame don't share an origin, so
            # the sensed and ground-truth bay can't be compared directly -
            # only as offsets from this episode's own known start pose,
            # in each frame respectively. Used by the sanity check below.
            start_odom=(samples[0]['odom_x'], samples[0]['odom_y']),
            start_truth=(samples[0]['x'], samples[0]['y']),
        ))

    if not episodes:
        raise SystemExit("nothing usable - every episode was too short or "
                         "never gave the lidar a clean view of the gap")

    # ---- split by episode, stratified over start labels ------------------
    rng = np.random.default_rng(args.seed)
    by_label = {}
    for i, e in enumerate(episodes):
        by_label.setdefault(e['label'], []).append(i)

    train_idx, val_idx = [], []
    for label in sorted(by_label):
        idx = np.array(by_label[label])
        rng.shuffle(idx)
        n_val = max(1, int(round(args.val_frac * len(idx))))
        if len(idx) - n_val < 1:
            n_val = len(idx) - 1
        val_idx += list(idx[:n_val])
        train_idx += list(idx[n_val:])

    def stack(indices):
        o = np.concatenate([episodes[i]['obs'] for i in indices])
        a = np.concatenate([
            chunk_actions(episodes[i]['act'], args.chunk)
            if args.chunk > 1 else episodes[i]['act']
            for i in indices])
        return o, a

    Xtr, Ytr = stack(train_idx)
    Xva, Yva = stack(val_idx)

    obs_mean = Xtr.mean(0)
    obs_std = Xtr.std(0)
    obs_std[obs_std < 1e-6] = 1.0

    flat = Ytr.reshape(-1, 2)
    act_mean = flat.mean(0)
    act_std = flat.std(0)
    act_std[act_std < 1e-6] = 1.0

    np.savez(
        args.out,
        X_train=(Xtr - obs_mean) / obs_std,
        Y_train=(Ytr - act_mean) / act_std,
        X_val=(Xva - obs_mean) / obs_std,
        Y_val=(Yva - act_mean) / act_std,
        obs_mean=obs_mean, obs_std=obs_std,
        act_mean=act_mean, act_std=act_std,
        chunk=args.chunk, history=args.history,
        train_episodes=np.array([episodes[i]['name'] for i in train_idx]),
        val_episodes=np.array([episodes[i]['name'] for i in val_idx]),
        # detector settings baked in, so bc_policy_node_lidar.py can be
        # pointed at this checkpoint and warned if its own flags disagree
        det_near_thresh=args.near_thresh, det_min_run=args.min_run,
        det_fov_deg=args.fov_deg, det_lock_count=args.lock_count,
        det_lock_tol=args.lock_tol,
    )

    # ---- report ----------------------------------------------------------
    print(f"\n{len(episodes)} usable episodes from {args.data_dir}/ "
          f"({skipped_short} too short, {skipped_nodetect} no clean "
          "lidar detection)")
    for label in sorted(by_label):
        n = len(by_label[label])
        tr = sum(1 for i in train_idx if episodes[i]['label'] == label)
        print(f"   start {label}: {n} episodes  ({tr} train, {n - tr} val)")

    obs_desc = (' (dx, dy, sin, cos, v_prev, w_prev)' if args.history
                else ' (dx, dy, sin, cos)')
    act_desc = ('(%d, 2) chunk' % args.chunk if args.chunk > 1
                else '2 (v, w)')
    print(f"\nobservation: {Xtr.shape[1]} dims{obs_desc}"
          "   [dx,dy now relative to the SENSED bay, not a config constant]")
    print(f"action:      {act_desc}")
    print(f"\nsamples:  train {len(Xtr)}   val {len(Xva)}")

    # sanity check: sensed bay vs ground truth, per episode, measured as
    # an odometry-frame offset from this episode's own start (the two
    # frames don't share an origin, but this comparison is fair because
    # everything below is measured relative to that same episode's start).
    print("\nsensed-bay sanity check (sensed vs ground truth, per episode, "
          "relative to that episode's own start pose):")
    errs = []
    for e in episodes:
        sox, soy = e['start_odom']
        stx, sty = e['start_truth']
        bx_odom_rel = e['bay_sensed'][0] - sox
        by_odom_rel = e['bay_sensed'][1] - soy
        gt_bx, gt_by = 0.0, 0.16   # nominal bay in the world/ground-truth frame
        bx_truth_rel = gt_bx - stx
        by_truth_rel = gt_by - sty
        err = math.hypot(bx_odom_rel - bx_truth_rel, by_odom_rel - by_truth_rel)
        errs.append(err)
    errs = np.array(errs)
    print(f"   mean {errs.mean():.3f} m   worst {errs.max():.3f} m   "
          f"(large values mean the detector is finding the wrong thing - "
          "retune --near-thresh/--fov-deg/--min-run)")

    print(f"\nwrote {args.out}\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_lidar')
    p.add_argument('--out', default='dataset_lidar.npz')
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--chunk', type=int, default=1)
    p.add_argument('--history', action='store_true')
    p.add_argument('--min-samples', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--near-thresh', type=float, default=DEFAULTS['near_thresh'])
    p.add_argument('--min-run', type=int, default=DEFAULTS['min_run'])
    p.add_argument('--fov-deg', type=float, default=DEFAULTS['fov_deg'])
    p.add_argument('--lock-count', type=int, default=DEFAULTS['lock_count'])
    p.add_argument('--lock-tol', type=float, default=DEFAULTS['lock_tol'])
    return p


if __name__ == '__main__':
    build(build_parser().parse_args())
