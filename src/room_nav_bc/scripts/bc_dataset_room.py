#!/usr/bin/env python3
"""
Build a training set from room-navigation episodes (camera + odometry
ONLY - no map, no ground truth, no gap/obstacle knowledge of any kind),
for either regression or generative (flow-matching) behavioural cloning.

    python3 bc_dataset_room.py --data-dir data_room_expert --out dataset_room.npz

Works on episodes from EITHER expert_demo_room.py or record_demo_room.py
(or both, pass --data-dir multiple times) - they share the exact same
per-sample schema, so a scripted-expert episode and a human-driven one are
interchangeable here.

OBSERVATION (two modalities, fed to the network separately - see
bc_train_room.py's CNNPolicy):

    image            the recorded camera frame, resized to --img-size x
                     --img-size, kept as uint8 (normalised to [0,1] at
                     train time, not here, to keep this file smaller)
    odom             [dx, dy, sin(yaw), cos(yaw)] - odometry position
                     relative to THAT EPISODE'S OWN START (the only
                     reference available; there is no bay/goal constant
                     here the way parking_bc's dataset has) [+ v_prev,
                     w_prev with --history]

ACTION: v, w (+ --chunk K for the next K actions, same as every other
dataset builder in this project).

THE LEFT/RIGHT CHOICE IS NEVER IN THE OBSERVATION.  Whether an episode
went left or right of the island shows up only in the recorded actions,
never as an explicit input feature - exactly like the two-bay/double-gap
problem in parking_bc and CS224R's Flappy Bird hard mode. A regression
(MSE) policy trained on a balanced left/right dataset is expected to
average the two and drive into the island; that is the point of this
dataset, not a bug to fix here.

THE SPLIT IS BY EPISODE, NOT BY SAMPLE - see parking_bc's bc_dataset.py
for why (near-duplicate adjacent samples at 10 Hz).
"""

import argparse
import glob
import math
import os
import pickle

import cv2
import numpy as np


def load_episode(path):
    with open(path, 'rb') as f:
        ep = pickle.load(f)
    return ep['samples'], ep.get('meta', {})


def decode_image(jpeg_bytes, img_size):
    arr = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    arr = cv2.resize(arr, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return arr[:, :, ::-1]   # BGR (cv2) -> RGB, for sanity when visualised


def episode_arrays(samples, img_size, use_history):
    """-> images (T,H,W,3) uint8, odom_obs (T,D) float32, act (T,2) float32."""
    ox0, oy0 = samples[0]['odom_x'], samples[0]['odom_y']
    images, odom_obs, act = [], [], []
    v_prev, w_prev = 0.0, 0.0
    for s in samples:
        images.append(decode_image(s['img_jpeg'], img_size))
        row = [s['odom_x'] - ox0, s['odom_y'] - oy0,
               math.sin(s['odom_yaw']), math.cos(s['odom_yaw'])]
        if use_history:
            row += [v_prev, w_prev]
        odom_obs.append(row)
        act.append([s['v'], s['w']])
        v_prev, w_prev = s['v'], s['w']
    return (np.asarray(images, np.uint8),
            np.asarray(odom_obs, np.float32),
            np.asarray(act, np.float32))


def chunk_actions(act, k):
    T = len(act)
    out = np.empty((T, k, 2), np.float32)
    for i in range(T):
        for j in range(k):
            out[i, j] = act[min(i + j, T - 1)]
    return out


def build(args):
    paths = []
    for d in args.data_dir:
        paths += sorted(glob.glob(os.path.join(d, '*.p')))
    if not paths:
        raise SystemExit(f"no episodes found in {args.data_dir}")

    episodes, skipped = [], 0
    for p in paths:
        samples, meta = load_episode(p)
        if len(samples) < args.min_samples:
            print(f"  skipping {os.path.basename(p)}: "
                  f"only {len(samples)} samples")
            skipped += 1
            continue
        if 'img_jpeg' not in samples[0] or 'odom_x' not in samples[0]:
            raise SystemExit(
                f"{p} has no camera/odom fields - was it recorded with "
                "record_demo_room.py or expert_demo_room.py?")

        images, odom_obs, act = episode_arrays(samples, args.img_size,
                                               args.history)
        # Samples from DAgger rounds carry human=True/False (True = a human
        # was driving that step). Recorded episodes have no flag: all human.
        human = np.array([bool(s.get('human', True)) for s in samples])
        episodes.append(dict(
            name=os.path.basename(p),
            label=meta.get('start_label', '?'),
            side=meta.get('side', '?'),
            source=meta.get('source', 'human'),
            images=images, odom_obs=odom_obs, act=act, human=human,
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

    def keep_rows(e):
        # --human-only drops steps where the policy was driving (DAgger
        # rounds). The full trial was still used above for the odometry
        # origin, so the observation means the same thing for every row.
        return e['human'] if args.human_only else np.ones(len(e['human']), bool)

    def stack(indices):
        imgs = np.concatenate([episodes[i]['images'][keep_rows(episodes[i])]
                               for i in indices])
        odom = np.concatenate([episodes[i]['odom_obs'][keep_rows(episodes[i])]
                               for i in indices])
        a = np.concatenate([
            (chunk_actions(episodes[i]['act'], args.chunk)
             if args.chunk > 1 else episodes[i]['act'])[keep_rows(episodes[i])]
            for i in indices])
        return imgs, odom, a

    Xtr_img, Xtr_odom, Ytr = stack(train_idx)
    Xva_img, Xva_odom, Yva = stack(val_idx)

    # normalise only the odometry features (images are normalised at
    # train time, straight from the stored uint8)
    odom_mean = Xtr_odom.mean(0)
    odom_std = Xtr_odom.std(0)
    odom_std[odom_std < 1e-6] = 1.0

    flat = Ytr.reshape(-1, 2)
    act_mean = flat.mean(0)
    act_std = flat.std(0)
    act_std[act_std < 1e-6] = 1.0

    np.savez(
        args.out,
        X_train_img=Xtr_img,
        X_train_odom=(Xtr_odom - odom_mean) / odom_std,
        Y_train=(Ytr - act_mean) / act_std,
        X_val_img=Xva_img,
        X_val_odom=(Xva_odom - odom_mean) / odom_std,
        Y_val=(Yva - act_mean) / act_std,
        odom_mean=odom_mean, odom_std=odom_std,
        act_mean=act_mean, act_std=act_std,
        chunk=args.chunk, history=args.history, img_size=args.img_size,
        train_episodes=np.array([episodes[i]['name'] for i in train_idx]),
        val_episodes=np.array([episodes[i]['name'] for i in val_idx]),
    )

    # ---- report ----------------------------------------------------------
    size_mb = os.path.getsize(args.out) / 1e6
    by_side = {}
    for e in episodes:
        by_side[e['side']] = by_side.get(e['side'], 0) + 1
    print(f"\n{len(episodes)} usable episodes from {args.data_dir} "
          f"({skipped} too short)")
    print(f"   by side: {dict(sorted(by_side.items()))}")
    for label in sorted(by_label):
        n = len(by_label[label])
        tr = sum(1 for i in train_idx if episodes[i]['label'] == label)
        sides = [episodes[i]['side'] for i in by_label[label]]
        print(f"   start {label}: {n} episodes  ({tr} train, {n - tr} val)  "
              f"sides={dict((s, sides.count(s)) for s in set(sides))}")

    odom_desc = (' (dx, dy, sin, cos, v_prev, w_prev)' if args.history
                else ' (dx, dy, sin, cos)')
    act_desc = ('(%d, 2) chunk' % args.chunk if args.chunk > 1
                else '2 (v, w)')
    print(f"\nimage obs:  {args.img_size}x{args.img_size}x3 uint8")
    print(f"odom obs:   {Xtr_odom.shape[1]} dims{odom_desc}  "
          "[relative to each episode's OWN start - no map, no goal "
          "constant]")
    print(f"action:     {act_desc}")
    print(f"\nsamples:  train {len(Xtr_img)}   val {len(Xva_img)}")
    print(f"\nwrote {args.out}  ({size_mb:.1f} MB)")

    if len(episodes) < 20:
        print(f"\nNOTE: only {len(episodes)} episodes. A CNN+odometry "
              "policy has a lot more to learn\n      from the same amount "
              "of data than the small MLP policies elsewhere in this\n"
              "      project did (4-6 input dims vs an image) - if the "
              "trained policy drives\n      poorly, more episodes from "
              "expert_demo_room.py (it's free, no human needed)\n"
              "      is the first thing to try, before assuming the "
              "method itself is wrong.")

    still = sum(int(np.sum(np.abs(e['act']).sum(1) < 1e-3)) for e in episodes)
    total = sum(len(e['act']) for e in episodes)
    if still > 0.25 * total:
        print(f"\nstationary samples: {still}/{total} "
              f"({100.0 * still / total:.0f}%) - NOTE: a lot of idle time "
              "in the data, the policy will learn to stop early.")
    print()


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', nargs='+', default=['data_room_expert'],
                   help='one or more directories of recorded episodes '
                        '(default: data_room_expert)')
    p.add_argument('--out', default='dataset_room.npz')
    p.add_argument('--img-size', type=int, default=64,
                   help='camera frames are resized to this x this '
                        '(default 64)')
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--chunk', type=int, default=1)
    p.add_argument('--history', action='store_true')
    p.add_argument('--min-samples', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--human-only', action='store_true',
                   help='train only on steps where a human was driving '
                        '(for DAgger rounds; recorded episodes count as human)')
    return p


if __name__ == '__main__':
    build(build_parser().parse_args())
