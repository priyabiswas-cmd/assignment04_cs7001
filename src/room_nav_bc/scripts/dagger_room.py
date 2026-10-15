#!/usr/bin/env python3
"""
Human-in-the-loop DAgger for the room-navigation policy (HG-DAgger style).

The current policy drives. When it goes wrong, you take over, drive it
back on track, and hand control back. Every step is saved with the
command that was actually executed, plus a flag saying whether you were
driving. Training later uses only your steps (bc_dataset_room.py
--human-only), so the policy learns the correction it needed in the
states it actually got itself into - that is the DAgger idea.

    ros2 run room_nav_bc dagger_room.py --policy policy_room_human.pt \
        --starts train --trials 5 --round 1

KEYS
    SPACE          take over / hand back to the policy (toggles)
    w / x          drive forward / backward   (only while you have control)
    a / d          turn left / right          (only while you have control)
    s              stop                       (only while you have control)
    n              skip this trial (saved as 'skipped')
    q              quit and save the steps so far

Each trial is saved as one episode in <out-dir>/round_<k>/. Episodes contain
the full trial (camera, odometry, executed command, human flag); the
dataset builder decides what to train on.

Only your steps are labelled by you, so a round with no takeovers adds no
training data. Watch the intervention rate printed at the end of each trial:
it should fall from round to round if the policy is improving.
"""

import argparse
import os
import pickle
import time

import rclpy

from bc_policy_node_room import (PolicyRunner, build_parser as policy_parser,
                                 default_config_path, load_policy, load_poses)
from record_demo_room import (KeyReader, encode_frame, LIN_STEP, ANG_STEP,
                              MAX_LIN, MAX_ANG)
import yaml


class DaggerRunner(PolicyRunner):

    def __init__(self, args, model, ck, poses, obstacles, goal, keys):
        # set our own state BEFORE the base class starts the first trial
        self.keys = keys
        self.human_mode = False
        self.h_lin = 0.0
        self.h_ang = 0.0
        self.ep_samples = []
        self.round_dir = os.path.join(args.out_dir, f'round_{args.round:02d}')
        os.makedirs(self.round_dir, exist_ok=True)
        self.saved = 0
        self.intervention = []           # per-trial human fraction
        super().__init__(args, model, ck, poses, obstacles, goal)
        self.override = self._takeover
        self.sample_hook = self._record
        self.create_timer(0.02, self._on_keys)

    # -- takeover -------------------------------------------------------

    def _takeover(self, pv, pw):
        if self.human_mode:
            return self.h_lin, self.h_ang, True
        return pv, pw, False

    def _record(self, d):
        if not self.running or d['frame'] is None:
            return
        self.ep_samples.append(dict(
            x=d['x'], y=d['y'], yaw=d['yaw'],
            odom_x=d['odom_x'], odom_y=d['odom_y'], odom_yaw=d['odom_yaw'],
            img_jpeg=encode_frame(d['frame']),
            v=d['v'], w=d['w'], human=d['human'],
        ))

    # -- keys -----------------------------------------------------------

    def _on_keys(self):
        k = self.keys.get()
        if k is None:
            return
        if k == ' ':
            self.human_mode = not self.human_mode
            self.h_lin = self.h_ang = 0.0
            self.publish(0.0, 0.0)
            print(f"\n  >> {'YOU have control' if self.human_mode else 'policy has control'}")
        elif self.human_mode and k == 'w':
            self.h_lin = min(MAX_LIN, self.h_lin + LIN_STEP * 2)
        elif self.human_mode and k == 'x':
            self.h_lin = max(-MAX_LIN, self.h_lin - LIN_STEP * 2)
        elif self.human_mode and k == 'a':
            self.h_ang = min(MAX_ANG, self.h_ang + ANG_STEP)
        elif self.human_mode and k == 'd':
            self.h_ang = max(-MAX_ANG, self.h_ang - ANG_STEP)
        elif self.human_mode and k == 's':
            self.h_lin = self.h_ang = 0.0
        elif k == 'n' and self.running:
            x, y, yaw = self.pose if self.pose else (0.0, 0.0, 0.0)
            self.finish_trial('skipped', x, y, yaw)
        elif k == 'q':
            self.publish(0.0, 0.0)
            self.save_steps()
            raise KeyboardInterrupt

    # -- trial boundaries ------------------------------------------------

    def finish_trial(self, outcome, x, y, yaw):
        n = len(self.ep_samples)
        human_frac = (sum(s['human'] for s in self.ep_samples) / n) if n else 0.0
        if n:
            label = self.cur_trial['label']
            idx = self.cur_trial['idx']
            path = os.path.join(self.round_dir,
                                f'start{label}_{idx:03d}.p')
            with open(path, 'wb') as f:
                pickle.dump(dict(
                    samples=self.ep_samples,
                    meta=dict(source='dagger', start_label=label,
                              outcome=outcome, human_fraction=human_frac,
                              round=self.args.round, recorded_at=time.strftime(
                                  '%Y-%m-%d %H:%M:%S')),
                ), f)
            self.saved += 1
        self.intervention.append(human_frac)
        print(f"     [{outcome}] {n} steps, you drove {100 * human_frac:.0f}% of them")
        self.human_mode = False
        self.h_lin = self.h_ang = 0.0
        self.ep_samples = []
        super().finish_trial(outcome, x, y, yaw)

    def report(self):
        super().report()
        if self.intervention:
            mean = 100 * sum(self.intervention) / len(self.intervention)
            print(f"\nround {self.args.round}: saved {self.saved} episodes to "
                  f"{self.round_dir}")
            print(f"intervention rate (share of steps you drove): {mean:.1f}%")
            print("next: bc_dataset_room.py --human-only, retrain, run the next round\n")


def main():
    p = policy_parser()
    p.add_argument('--out-dir', default='data_dagger')
    p.add_argument('--round', type=int, default=1)
    args = p.parse_args()
    if args.config is None:
        args.config = default_config_path()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    poses = load_poses(args.config, args.starts)
    room = cfg['room']
    args.room_x_min, args.room_x_max = room['x_min'], room['x_max']
    args.room_y_min, args.room_y_max = room['y_min'], room['y_max']
    model, ck = load_policy(args.policy)

    rclpy.init()
    with KeyReader() as keys:
        node = DaggerRunner(args, model, ck, poses, cfg['obstacles'],
                            cfg['goal'], keys)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, RuntimeError):
            pass
        finally:
            try:
                node.publish(0.0, 0.0)
                node.report()
            except Exception:
                pass
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
