#!/usr/bin/env python3
"""
Plot what a learned room policy actually did, from its rollout log.

    python3 plot_policy_room.py --steps-csv logs/centre_test_steps.csv --out figures/policy_centre.png

Two panels:
  1. Map: every logged position, drawn over the room layout, coloured by the
     turn rate the policy commanded there (red = turned right, blue = left).
     The island, gap boxes and goal are drawn from start_goal.yaml.
  2. Histogram of the commanded turn rate, with the expert's left-route and
     right-route turn rates at the same states overlaid. A policy that
     averages the two modes puts its mass between the two expert peaks.
"""

import argparse
import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse_rollouts_room as an   # reuses the expert route controllers


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('x', 'y', 'yaw', 'start_x', 'start_y', 'start_yaw',
                  'odom_dx', 'odom_dy', 'odom_yaw', 'pred_v', 'pred_w'):
            r[k] = float(r[k])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps-csv', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    config = args.config or an.expert.default_config_path()
    cfg = yaml.safe_load(open(config))
    obstacles, goal, room = cfg['obstacles'], cfg['goal'], cfg['room']
    xargs = an.expert.build_parser().parse_args([])

    rows = load(args.steps_csv)
    if not rows:
        raise SystemExit('no steps in that log')

    xs = np.array([r['x'] for r in rows])
    ys = np.array([r['y'] for r in rows])
    pw = np.array([r['pred_w'] for r in rows])

    wl, wr = [], []
    for r in rows:
        wl.append(an.expert_turn_rates(r, obstacles, goal, xargs, 'left')[1])
        wr.append(an.expert_turn_rates(r, obstacles, goal, xargs, 'right')[1])
    wl, wr = np.array(wl), np.array(wr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   gridspec_kw={'width_ratios': [1, 1.1]})

    # ---- panel 1: map ---------------------------------------------------
    ax1.add_patch(patches.Rectangle((room['x_min'], room['y_min']),
                                    room['x_max'] - room['x_min'],
                                    room['y_max'] - room['y_min'],
                                    fill=False, lw=2, color='black'))
    colours = {'island': '#8b5a2b', 'gap_left': '#b33a3a', 'gap_right': '#3a5fb3'}
    for name, b in obstacles.items():
        c = colours.get(name, '#999999')
        ax1.add_patch(patches.Rectangle((b['x'] - b['w'] / 2, b['y'] - b['h'] / 2),
                                        b['w'], b['h'], color=c, alpha=0.6))
    ax1.add_patch(patches.Rectangle((goal['x'] - goal['w'] / 2, goal['y'] - goal['h'] / 2),
                                    goal['w'], goal['h'], color='gold', alpha=0.5))
    sc = ax1.scatter(xs, ys, c=pw, cmap='coolwarm', s=6,
                     vmin=-max(abs(pw)), vmax=max(abs(pw)))
    fig.colorbar(sc, ax=ax1, label='commanded turn rate w (rad/s)')
    ax1.set_xlim(room['x_min'] - 0.1, room['x_max'] + 0.1)
    ax1.set_ylim(room['y_min'] - 0.1, room['y_max'] + 0.1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Where the policy drove, coloured by turn rate')

    # ---- panel 2: distribution -------------------------------------------
    bins = np.linspace(min(pw.min(), wl.min(), wr.min()) - 0.1,
                       max(pw.max(), wl.max(), wr.max()) + 0.1, 40)
    ax2.hist(pw, bins=bins, color='#444444', alpha=0.8, label='policy')
    ax2.axvline(np.median(wl), color='#3a5fb3', lw=2, ls='--',
                label=f'expert left route (median {np.median(wl):+.2f})')
    ax2.axvline(np.median(wr), color='#b33a3a', lw=2, ls='--',
                label=f'expert right route (median {np.median(wr):+.2f})')
    ax2.axvline(0, color='black', lw=1, alpha=0.5)
    ax2.set_xlabel('commanded turn rate w (rad/s)')
    ax2.set_ylabel('number of control steps')
    ax2.set_title(f'Turn-rate distribution ({len(rows)} steps)')
    ax2.legend(fontsize=9)

    fig.suptitle(os.path.basename(args.steps_csv))
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f'wrote {args.out}')
    print(f'  policy turn rate:  mean {pw.mean():+.2f}  std {pw.std():.2f}')
    print(f'  expert left/right: median {np.median(wl):+.2f} / {np.median(wr):+.2f}')


if __name__ == '__main__':
    main()
