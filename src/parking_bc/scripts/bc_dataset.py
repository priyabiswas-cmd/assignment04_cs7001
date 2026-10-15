#!/usr/bin/env python3
"""
Turn recorded episodes into a training set for behavioural cloning.

    python3 scripts/bc_dataset.py --data-dir data --out dataset.npz

Observation (what the policy sees), by default 4 numbers:

    dx, dy          position relative to the bay centre
    sin(yaw), cos(yaw)

    yaw is split into sin/cos so the network never sees the
    discontinuity at +-pi.

With --history the previous applied action (v, w) is appended, giving 6
inputs.  That lets an otherwise memoryless policy tell apart phases of the
manoeuvre that look identical from pose alone - useful if the same (dx, dy,
yaw) appears in your demonstrations with two different correct actions.

Action (what the policy predicts):

    v, w            linear and angular velocity

With --chunk K the target becomes the next K actions (K, 2) instead of one.
Predicting a short sequence keeps the policy internally consistent when it
drifts slightly off the demonstrated states, which is the main failure mode
of single-step behavioural cloning.

THE SPLIT IS BY EPISODE, NOT BY SAMPLE.  At 10 Hz two consecutive samples
are almost the same point; splitting at random would put near-duplicates on
both sides and report a validation loss that means nothing.
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


def episode_arrays(samples, use_history):
    """-> obs (T, D), act (T, 2) for one episode."""
    obs, act = [], []
    v_prev, w_prev = 0.0, 0.0
    for s in samples:
        row = [s['bay_x'], s['bay_y'],
               math.sin(s['yaw']), math.cos(s['yaw'])]
        if use_history:
            row += [v_prev, w_prev]
        obs.append(row)
        act.append([s['v'], s['w']])
        v_prev, w_prev = s['v'], s['w']
    return np.asarray(obs, np.float32), np.asarray(act, np.float32)


def chunk_actions(act, k):
    """
    (T, 2) -> (T, k, 2): the next k actions from each step.

    Near the end of an episode there are fewer than k actions left, so the
    last one is repeated.  The robot is stationary there, so padding with
    "keep doing the final action" is the honest choice.
    """
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
    for p in paths:
        samples, meta = load_episode(p)
        if len(samples) < args.min_samples:
            print(f"  skipping {os.path.basename(p)}: "
                  f"only {len(samples)} samples")
            continue
        obs, act = episode_arrays(samples, args.history)
        episodes.append(dict(
            name=os.path.basename(p),
            label=meta.get('start_label', '?'),
            obs=obs, act=act,
            final=(samples[-1]['bay_x'], samples[-1]['bay_y']),
        ))

    if not episodes:
        raise SystemExit("every episode was too short - nothing to build")

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
        if len(idx) - n_val < 1:            # never leave a label unseen
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

    # ---- normalisation, from the TRAIN split only ------------------------
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
    )

    # ---- report ----------------------------------------------------------
    print(f"\n{len(episodes)} episodes from {args.data_dir}/")
    for label in sorted(by_label):
        n = len(by_label[label])
        tr = sum(1 for i in train_idx if episodes[i]['label'] == label)
        print(f"   start {label}: {n} episodes  ({tr} train, {n - tr} val)")

    obs_desc = (' (dx, dy, sin, cos, v_prev, w_prev)' if args.history
                else ' (dx, dy, sin, cos)')
    act_desc = ('(%d, 2) chunk' % args.chunk if args.chunk > 1
                else '2 (v, w)')
    print(f"\nobservation: {Xtr.shape[1]} dims{obs_desc}")
    print(f"action:      {act_desc}")
    print(f"\nsamples:  train {len(Xtr)}   val {len(Xva)}")
    print(f"\nobs  mean {np.array2string(obs_mean, precision=3)}")
    print(f"     std  {np.array2string(obs_std, precision=3)}")
    print(f"act  mean {np.array2string(act_mean, precision=3)}")
    print(f"     std  {np.array2string(act_std, precision=3)}")

    finals = np.array([e['final'] for e in episodes])
    print("\nfinal positions relative to bay centre:")
    print(f"     dx  {finals[:, 0].mean():+.3f} +- {finals[:, 0].std():.3f}")
    print(f"     dy  {finals[:, 1].mean():+.3f} +- {finals[:, 1].std():.3f}")
    spread = np.hypot(finals[:, 0].std(), finals[:, 1].std())
    if spread > 0.10:
        print("     NOTE: final positions vary a lot between episodes. "
              "Inconsistent\n           demonstrations give the policy "
              "contradictory targets.")

    still = sum(int(np.sum(np.abs(e['act']).sum(1) < 1e-3)) for e in episodes)
    total = sum(len(e['act']) for e in episodes)
    print(f"\nstationary samples: {still}/{total} "
          f"({100.0 * still / total:.0f}%)")
    if still > 0.25 * total:
        print("     NOTE: a lot of the data is the robot not moving. "
              "The policy will\n           learn to stop early. Trim idle "
              "time or drive more decisively.")

    print(f"\nwrote {args.out}\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out', default='dataset.npz')
    p.add_argument('--val-frac', type=float, default=0.2,
                   help='fraction of episodes per start held out (0.2)')
    p.add_argument('--chunk', type=int, default=1,
                   help='predict the next K actions (default 1)')
    p.add_argument('--history', action='store_true',
                   help='append the previous action to the observation')
    p.add_argument('--min-samples', type=int, default=20,
                   help='ignore episodes shorter than this')
    p.add_argument('--seed', type=int, default=0)
    return p


if __name__ == '__main__':
    build(build_parser().parse_args())
