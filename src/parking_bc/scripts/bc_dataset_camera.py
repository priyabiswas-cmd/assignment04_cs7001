#!/usr/bin/env python3
"""
Turn recorded camera episodes into a training set for image-based BC.

    python3 scripts/bc_dataset_camera.py --data-dir data_camera --out dataset_camera.npz

Observation: the camera IMAGE only. x, y, yaw are loaded from each episode's
meta for plotting/diagnostics, but are never placed in X_train / X_val -
that would defeat the entire point of switching to a camera.

Action: v, w (or the next K actions with --chunk).

THE SPLIT IS BY EPISODE, NOT BY FRAME, same reasoning as bc_dataset.py:
consecutive frames are near-duplicates, so a random per-frame split would
leak validation data into training.

Images are kept as uint8 (not normalised to float here) to keep the file
small - a normalisation script commonly stores raw pixels and normalises
per-batch during training instead. Per-channel mean/std ARE computed here
from the training images and saved, so bc_train_camera.py normalises
consistently without recomputing statistics itself.
"""

import argparse
import glob
import os
import pickle

import numpy as np


def load_episode(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def chunk_actions(act, k):
    """(T, 2) -> (T, k, 2): next k actions from each step, padded at the end."""
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

    episodes = []
    total_noisy_frames, total_frames = 0, 0
    for p in paths:
        ep = load_episode(p)
        meta = ep.get('meta', {})
        n = meta.get('n_samples', len(ep['actions']))
        if n < args.min_samples:
            print(f"  skipping {os.path.basename(p)}: only {n} samples")
            continue
        episodes.append(dict(
            name=os.path.basename(p),
            label=meta.get('start_label', '?'),
            images=ep['images'],
            actions=ep['actions'].astype(np.float32),
            final=(float(meta['x'][-1]), float(meta['y'][-1])),
        ))
        if 'noisy' in meta:
            total_noisy_frames += int(np.sum(meta['noisy']))
            total_frames += len(meta['noisy'])

    if not episodes:
        raise SystemExit("every episode was too short - nothing to build")

    img_size = episodes[0]['images'].shape[1]
    for e in episodes:
        if e['images'].shape[1:3] != (img_size, img_size):
            raise SystemExit(
                f"{e['name']} has image size {e['images'].shape[1:3]}, "
                f"expected {(img_size, img_size)} - mixed --img-size runs?")

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
        imgs = np.concatenate([episodes[i]['images'] for i in indices])
        if args.chunk > 1:
            acts = np.concatenate([
                chunk_actions(episodes[i]['actions'], args.chunk)
                for i in indices])
        else:
            acts = np.concatenate([episodes[i]['actions'] for i in indices])
        return imgs, acts

    Ximg_tr, Ytr = stack(train_idx)
    Ximg_va, Yva = stack(val_idx)

    # ---- action normalisation, from TRAIN split only ----------------------
    flat = Ytr.reshape(-1, 2)
    act_mean = flat.mean(0)
    act_std = flat.std(0)
    act_std[act_std < 1e-6] = 1.0

    # ---- image normalisation stats (pixels/255, per channel), TRAIN only --
    sample = Ximg_tr[:: max(1, len(Ximg_tr) // 2000)].astype(np.float32) / 255.0
    img_mean = sample.mean(axis=(0, 1, 2))
    img_std = sample.std(axis=(0, 1, 2))
    img_std[img_std < 1e-6] = 1.0

    np.savez(
        args.out,
        X_train_img=Ximg_tr, Y_train=(Ytr - act_mean) / act_std,
        X_val_img=Ximg_va, Y_val=(Yva - act_mean) / act_std,
        img_mean=img_mean, img_std=img_std,
        act_mean=act_mean, act_std=act_std,
        img_size=img_size, chunk=args.chunk,
        train_episodes=np.array([episodes[i]['name'] for i in train_idx]),
        val_episodes=np.array([episodes[i]['name'] for i in val_idx]),
    )

    # ---- report ------------------------------------------------------------
    print(f"\n{len(episodes)} episodes from {args.data_dir}/")
    for label in sorted(by_label):
        n = len(by_label[label])
        tr = sum(1 for i in train_idx if episodes[i]['label'] == label)
        print(f"   start {label}: {n} episodes  ({tr} train, {n - tr} val)")

    print(f"\nimage:  {img_size}x{img_size}x3 uint8")
    act_desc = ('(%d, 2) chunk' % args.chunk if args.chunk > 1
                else '2 (v, w)')
    print(f"action: {act_desc}")
    print(f"\nframes:  train {len(Ximg_tr)}   val {len(Ximg_va)}")

    print(f"\naction  mean {np.array2string(act_mean, precision=3)}")
    print(f"        std  {np.array2string(act_std, precision=3)}")
    print(f"image   per-channel mean {np.array2string(img_mean, precision=3)} "
          f"(0-1 scale)")
    print(f"        per-channel std  {np.array2string(img_std, precision=3)}")

    finals = np.array([e['final'] for e in episodes])
    dx = finals[:, 0] - args.bay_x
    dy = finals[:, 1] - args.bay_y
    print(f"\nfinal positions relative to bay centre:")
    print(f"     dx  {dx.mean():+.3f} +- {dx.std():.3f}")
    print(f"     dy  {dy.mean():+.3f} +- {dy.std():.3f}")
    if np.hypot(dx.std(), dy.std()) > 0.10:
        print("     NOTE: final positions vary a lot between episodes - "
              "check the expert\n           is reaching the bay "
              "consistently before training on this.")

    if total_frames:
        print(f"\nframes recorded under trajectory noise: "
              f"{total_noisy_frames}/{total_frames} "
              f"({100.0*total_noisy_frames/total_frames:.0f}%)")

    print(f"\nwrote {args.out}\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_camera')
    p.add_argument('--out', default='dataset_camera.npz')
    p.add_argument('--val-frac', type=float, default=0.2,
                   help='fraction of episodes per start held out (0.2)')
    p.add_argument('--chunk', type=int, default=1,
                   help='predict the next K actions (default 1)')
    p.add_argument('--min-samples', type=int, default=15)
    p.add_argument('--bay-x', type=float, default=0.0)
    p.add_argument('--bay-y', type=float, default=0.16)
    p.add_argument('--seed', type=int, default=0)
    return p


if __name__ == '__main__':
    build(build_parser().parse_args())
