#!/usr/bin/env python3
"""
Build a training set where the network learns from RAW lidar + odometry
ONLY - no hand-coded gap detector, no knowledge of where the bay is.
Reuses the same episodes recorded by record_demo_lidar.py; nothing needs
to be re-recorded (they already logged scan_ranges and odom_x/y/yaw).

    python3 scripts_lidar/bc_dataset_lidar_raw.py \
        --data-dir data_lidar --out dataset_lidar_raw.npz

OBSERVATION (this is the part that's genuinely different from every other
dataset in this package):

    lidar_bins[0..K-1]   min range in each of K equal angular sectors of
                         the full 360 degree scan (near_thresh/fov/etc
                         from lidar_gap.py are NOT used here - nothing is
                         thresholded or classified, every bin is just a
                         number: "closest thing in that direction")
    dx, dy               odometry position relative to THIS EPISODE'S OWN
                         START - the only position reference available,
                         since the network is never told where the bay is
    sin(yaw), cos(yaw)    odometry heading
    (+ v_prev, w_prev with --history)

Whatever the network learns about "where the gap is" has to come purely
from the correlation, across your demonstrations, between (these lidar
bins, this much travelled since start) and (the action you took) - it is
never given the answer directly, unlike dataset_lidar.npz (sensed bay) or
the original dataset.npz (config bay). Expect this to need more/cleaner
demonstrations than the other two to learn well - the input is bigger
and the target is only implicit in it.

Every recorded episode is usable (there's no detector to fail here), so
nothing gets skipped for "no clean lidar detection" the way
bc_dataset_lidar.py's does.

THE SPLIT IS BY EPISODE, NOT BY SAMPLE - see scripts/bc_dataset.py for why.
"""

import argparse
import glob
import math
import os
import pickle

import numpy as np


def load_episode(path):
    with open(path, 'rb') as f:
        ep = pickle.load(f)
    return ep['samples'], ep.get('meta', {})


def bin_scan(ranges, angle_min, angle_increment, n_bins, range_max):
    """360 raw beams -> n_bins sector minimums, each clipped to
    [0, range_max]. A sector that saw nothing reads range_max - "nothing
    within sensor range in that direction" is itself informative, not a
    missing value, so it's a plain number, not inf/NaN."""
    two_pi = 2.0 * math.pi
    bins = [range_max] * n_bins
    width = two_pi / n_bins
    for i, r in enumerate(ranges):
        if not math.isfinite(r) or r <= 0.0:
            r = range_max
        else:
            r = min(r, range_max)
        angle = (angle_min + i * angle_increment) % two_pi
        b = int(angle / width) % n_bins
        if r < bins[b]:
            bins[b] = r
    return bins


def episode_arrays(samples, n_bins, use_history):
    """-> obs (T, D), act (T, 2) for one episode."""
    ox0 = samples[0]['odom_x']
    oy0 = samples[0]['odom_y']
    obs, act = [], []
    v_prev, w_prev = 0.0, 0.0
    for s in samples:
        bins = bin_scan(s['scan_ranges'], s['scan_angle_min'],
                        s['scan_angle_increment'], n_bins,
                        s['scan_range_max'])
        row = list(bins) + [
            s['odom_x'] - ox0, s['odom_y'] - oy0,
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

    episodes, skipped = [], 0
    for p in paths:
        samples, meta = load_episode(p)
        if len(samples) < args.min_samples:
            print(f"  skipping {os.path.basename(p)}: "
                  f"only {len(samples)} samples")
            skipped += 1
            continue
        if 'scan_ranges' not in samples[0] or 'odom_x' not in samples[0]:
            raise SystemExit(
                f"{p} has no odom/scan fields - was it recorded with "
                "record_demo_lidar.py, not the original record_demo.py?")

        obs, act = episode_arrays(samples, args.lidar_bins, args.history)
        episodes.append(dict(
            name=os.path.basename(p),
            label=meta.get('start_label', '?'),
            obs=obs, act=act,
        ))

    if not episodes:
        raise SystemExit("nothing usable - every episode was too short")

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
        lidar_bins=args.lidar_bins,
        train_episodes=np.array([episodes[i]['name'] for i in train_idx]),
        val_episodes=np.array([episodes[i]['name'] for i in val_idx]),
    )

    # ---- report ----------------------------------------------------------
    print(f"\n{len(episodes)} usable episodes from {args.data_dir}/ "
          f"({skipped} too short)")
    for label in sorted(by_label):
        n = len(by_label[label])
        tr = sum(1 for i in train_idx if episodes[i]['label'] == label)
        print(f"   start {label}: {n} episodes  ({tr} train, {n - tr} val)")

    obs_desc = (f' ({args.lidar_bins} lidar bins, dx, dy, sin, cos, '
                'v_prev, w_prev)' if args.history else
                f' ({args.lidar_bins} lidar bins, dx, dy, sin, cos)')
    act_desc = ('(%d, 2) chunk' % args.chunk if args.chunk > 1
                else '2 (v, w)')
    print(f"\nobservation: {Xtr.shape[1]} dims{obs_desc}"
          "   [raw lidar+odometry only - no gap detector, no bay knowledge]")
    print(f"action:      {act_desc}")
    print(f"\nsamples:  train {len(Xtr)}   val {len(Xva)}")
    print(f"\nwrote {args.out}\n")
    print("NOTE: this observation is much higher-dimensional than "
          f"dataset.npz/dataset_lidar.npz ({Xtr.shape[1]} vs 4-6 dims). "
          "With only a few thousand samples the network has to both find "
          "the gap AND learn to drive from the same data - if val loss "
          "looks poor, more/cleaner episodes (or fewer --lidar-bins) will "
          "help more here than it would for the other two datasets.\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_lidar')
    p.add_argument('--out', default='dataset_lidar_raw.npz')
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--chunk', type=int, default=1)
    p.add_argument('--history', action='store_true')
    p.add_argument('--min-samples', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lidar-bins', type=int, default=36,
                   help='360 raw beams collapsed into this many angular '
                        'sectors (default 36, i.e. one every 10 degrees)')
    return p


if __name__ == '__main__':
    build(build_parser().parse_args())
