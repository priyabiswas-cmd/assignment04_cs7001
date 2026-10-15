#!/usr/bin/env python3
"""
How much does the same pose disagree with itself across the dataset?

For each sample, find other samples at a nearly identical pose and measure
how much their actions differ.  That spread is a floor on the achievable
loss: a deterministic policy sees one input and must emit one output, so
whatever disagreement exists in the labels is error it can never remove.

    python3 diagnose_labels.py --data-dir data
"""
import argparse, glob, math, os, pickle
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--data-dir', default='data')
ap.add_argument('--radius', type=float, default=0.05,
                help='poses within this distance (m) count as "the same"')
ap.add_argument('--yaw-tol', type=float, default=15.0, help='degrees')
a = ap.parse_args()

obs, act, epi = [], [], []
for i, p in enumerate(sorted(glob.glob(os.path.join(a.data_dir, '*.p')))):
    s = pickle.load(open(p, 'rb'))['samples']
    for r in s:
        obs.append([r['bay_x'], r['bay_y'], r['yaw']])
        act.append([r['v'], r['w']])
        epi.append(i)
obs = np.array(obs); act = np.array(act); epi = np.array(epi)
print(f"{len(obs)} samples from {epi.max()+1} episodes\n")

rng = np.random.default_rng(0)
probe = rng.choice(len(obs), size=min(600, len(obs)), replace=False)
yaw_tol = math.radians(a.yaw_tol)

spreads_v, spreads_w, neigh_counts, cross_ep = [], [], [], []
for i in probe:
    d = np.hypot(obs[:, 0] - obs[i, 0], obs[:, 1] - obs[i, 1])
    dy = np.abs(np.arctan2(np.sin(obs[:, 2] - obs[i, 2]),
                           np.cos(obs[:, 2] - obs[i, 2])))
    m = (d < a.radius) & (dy < yaw_tol)
    if m.sum() < 3:
        continue
    neigh_counts.append(m.sum())
    cross_ep.append(len(np.unique(epi[m])))
    spreads_v.append(act[m, 0].std())
    spreads_w.append(act[m, 1].std())

sv, sw = np.mean(spreads_v), np.mean(spreads_w)
print(f"neighbourhoods: {len(spreads_v)} probed, "
      f"median {np.median(neigh_counts):.0f} samples each, "
      f"spanning {np.median(cross_ep):.0f} episodes\n")
print(f"action disagreement WITHIN a neighbourhood (this is the floor):")
print(f"   v : {sv:.4f} m/s      w : {sw:.4f} rad/s")
print(f"overall spread of the actions (what a constant predictor faces):")
print(f"   v : {act[:,0].std():.4f} m/s      w : {act[:,1].std():.4f} rad/s")
print(f"\nfraction of the variance that is irreducible:")
print(f"   v : {sv/act[:,0].std():.0%}      w : {sw/act[:,1].std():.0%}")

still = np.abs(act).sum(1) < 1e-3
print(f"\nstationary samples: {still.sum()}/{len(act)} ({still.mean():.0%})")
mv = ~still
if mv.sum():
    print("excluding them, the floor becomes:")
    sv2, sw2 = [], []
    for i in probe:
        if still[i]:
            continue
        d = np.hypot(obs[:,0]-obs[i,0], obs[:,1]-obs[i,1])
        dyw = np.abs(np.arctan2(np.sin(obs[:,2]-obs[i,2]),
                                np.cos(obs[:,2]-obs[i,2])))
        m = (d < a.radius) & (dyw < yaw_tol) & mv
        if m.sum() >= 3:
            sv2.append(act[m,0].std()); sw2.append(act[m,1].std())
    if sv2:
        print(f"   v : {np.mean(sv2):.4f} m/s      w : {np.mean(sw2):.4f} rad/s")
        print(f"   ({np.mean(sv2)/act[mv,0].std():.0%} and "
              f"{np.mean(sw2)/act[mv,1].std():.0%} of the moving-only spread)")

rev = np.sum(np.diff(np.sign(act[:,0][act[:,0]!=0])) != 0)
print(f"\ndirection reversals (forward<->reverse): {rev}")
