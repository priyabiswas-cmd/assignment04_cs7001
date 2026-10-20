#!/usr/bin/env python3
"""
Run a trained REINFORCE policy deterministically (the Gaussian's mean, no
sampling) and score it - the policy-gradient counterpart of
bc_policy_node_room.py. Same grading convention as every other test node
in this project: ground truth decides success/collision, the policy only
ever sees odometry.

    ros2 run room_nav_bc pg_policy_node_room.py --policy policy_pg.pt \
        --starts test --trials 5
"""

import argparse
import math
import os
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch not found - install it to run a policy.")

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_policy_node_room import (  # noqa: E402
    MAX_ANG, MAX_LIN, default_config_path, in_goal, load_poses,
    quaternion_from_yaw, worst_clearance, wrap_angle, yaw_from_quaternion,
)
from pg_networks import GaussianPolicy  # noqa: E402
from pg_train_room import RunningNormalizer  # noqa: E402


def load_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = GaussianPolicy(ck['obs_dim'], ck['act_dim'], tuple(ck['hidden']))
    model.load_state_dict(ck['state_dict'])
    model.eval()
    normalizer = RunningNormalizer(ck['obs_dim'])
    normalizer.load_state_dict(ck['normalizer'])
    return model, normalizer, ck


class PolicyRunner(Node):

    def __init__(self, args, model, normalizer, poses, obstacles, goal, room):
        super().__init__('pg_policy_room')
        self.args = args
        self.model = model
        self.normalizer = normalizer
        self.poses = poses
        self.obstacles, self.goal, self.room = obstacles, goal, room
        self.rng = np.random.default_rng(args.jitter_seed)

        self.pose = None
        self.odom = None
        self.odom_origin = None

        self.trial = 0
        self.pose_i = 0
        self.rep = 0
        self.results = []
        self.t_start = None
        self.still_since = None
        self.running = False
        self.min_clear = None
        self.min_clear_obs = None

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self.on_states, qos)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        self.pub = self.create_publisher(Twist, args.cmd_out, 10)
        self.reset_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(1.0 / args.rate, self.step)

        print(f"\npolicy  {os.path.basename(args.policy)}  "
              "(REINFORCE, odometry only, deterministic mean action)")
        print(f"starts  '{args.starts}'  x {len(poses)}  x {args.trials} trials\n")
        self.begin_trial()

    # -- plumbing ------------------------------------------------------

    def on_states(self, msg):
        try:
            i = msg.name.index(self.args.robot)
        except ValueError:
            return
        p = msg.pose[i]
        self.pose = (p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                         p.orientation.z, p.orientation.w))

    def on_odom(self, msg):
        p = msg.pose.pose
        self.odom = (p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                         p.orientation.z, p.orientation.w))

    def publish(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub.publish(m)

    def teleport(self, x, y, yaw):
        if not self.reset_cli.service_is_ready():
            self.reset_cli.wait_for_service(timeout_sec=3.0)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        req = SetEntityState.Request()
        req.state.name = self.args.robot
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.01
        req.state.pose.orientation.x = qx
        req.state.pose.orientation.y = qy
        req.state.pose.orientation.z = qz
        req.state.pose.orientation.w = qw
        self.reset_cli.call_async(req)

    # -- trials ------------------------------------------------------------

    def begin_trial(self):
        if self.pose_i >= len(self.poses):
            self.report()
            raise KeyboardInterrupt

        label, x0, y0, yaw0 = self.poses[self.pose_i]
        x = x0 + self.rng.uniform(-self.args.jitter_xy, self.args.jitter_xy)
        y = y0 + self.rng.uniform(-self.args.jitter_xy, self.args.jitter_xy)
        yaw = yaw0 + math.radians(
            self.rng.uniform(-self.args.jitter_yaw_deg, self.args.jitter_yaw_deg))
        self.publish(0.0, 0.0)
        self.teleport(x, y, yaw)
        self.pose = None
        self.odom = None
        self.odom_origin = None

        self.t_start = time.time()
        self.still_since = None
        self.running = True
        self.min_clear = float('inf')
        self.min_clear_obs = None
        print(f"  trial {self.trial + 1:2d}  start {label} "
              f"(rep {self.rep + 1}/{self.args.trials}, jittered to "
              f"x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.0f}) ... ",
              end='', flush=True)

    def finish_trial(self, outcome, x, y, yaw):
        self.publish(0.0, 0.0)
        self.running = False

        success = outcome == 'stopped' and in_goal(x, y, self.goal)
        collided = self.min_clear < 0.0

        self.results.append(dict(
            label=self.poses[self.pose_i][0], outcome=outcome,
            success=success, x=x, y=y,
            min_clear=self.min_clear, min_clear_obs=self.min_clear_obs,
        ))
        note = f"  [HIT {self.min_clear_obs}]" if collided else ""
        print(f"{'OK ' if success else 'no '} [{outcome}] "
              f"pos=({x:+.2f},{y:+.2f})  "
              f"clearance={self.min_clear:+.3f}m vs {self.min_clear_obs}{note}")

        self.trial += 1
        self.rep += 1
        if self.rep >= self.args.trials:
            self.rep = 0
            self.pose_i += 1
        time.sleep(0.4)
        self.begin_trial()

    # -- control loop --------------------------------------------------------

    def step(self):
        if not self.running or self.pose is None or self.odom is None:
            return
        if self.odom_origin is None:
            self.odom_origin = self.odom

        x, y, yaw = self.pose
        c, which = worst_clearance(x, y, self.obstacles, self.room)
        if c < self.min_clear:
            self.min_clear, self.min_clear_obs = c, which
        elapsed = time.time() - self.t_start

        ox, oy, oyaw = self.odom
        ox0, oy0, _ = self.odom_origin
        obs_raw = np.array([ox - ox0, oy - oy0, math.sin(oyaw), math.cos(oyaw)])
        obs_n = self.normalizer.normalize(obs_raw).astype(np.float32)

        with torch.no_grad():
            v, w = self.model.mean(torch.tensor(obs_n).unsqueeze(0)).squeeze(0).numpy()
        v = float(np.clip(v, -MAX_LIN, MAX_LIN))
        w = float(np.clip(w, -MAX_ANG, MAX_ANG))
        self.publish(v, w)

        moving = abs(v) > 0.01 or abs(w) > 0.05
        if not moving:
            if self.still_since is None:
                self.still_since = time.time()
            elif time.time() - self.still_since > self.args.settle:
                self.finish_trial('stopped', x, y, yaw)
                return
        else:
            self.still_since = None

        if elapsed > self.args.timeout:
            self.finish_trial('timeout', x, y, yaw)
        elif not (self.args.room_x_min - self.args.room_margin <= x
                 <= self.args.room_x_max + self.args.room_margin
                 and self.args.room_y_min - self.args.room_margin <= y
                 <= self.args.room_y_max + self.args.room_margin):
            self.finish_trial('left arena', x, y, yaw)

    # -- report ------------------------------------------------------------

    def report(self):
        r = self.results
        if not r:
            return
        n = len(r)
        ok = sum(x['success'] for x in r)
        hit = sum(1 for x in r if x['min_clear'] < 0.0)

        print("\n" + "=" * 58)
        print(f" REINFORCE policy: success {ok}/{n}  ({100.0 * ok / n:.0f}%)")
        print(f" collided with something: {hit}/{n}  ({100.0 * hit / n:.0f}%)")

        by = {}
        for x in r:
            by.setdefault(x['label'], []).append(x)
        print("\n per start:")
        for label in sorted(by):
            g = by[label]
            s = sum(z['success'] for z in g)
            h = sum(1 for z in g if z['min_clear'] < 0.0)
            print(f"   {label}:  {s}/{len(g)} success, {h}/{len(g)} collided")

        outcomes = {}
        for x in r:
            outcomes[x['outcome']] = outcomes.get(x['outcome'], 0) + 1
        print("\n outcomes: " + ", ".join(
            f"{k} {v}" for k, v in sorted(outcomes.items())))
        print("=" * 58 + "\n")

        if self.args.csv:
            import csv
            os.makedirs(os.path.dirname(self.args.csv) or '.', exist_ok=True)
            with open(self.args.csv, 'w', newline='') as f:
                wri = csv.DictWriter(f, fieldnames=list(r[0].keys()))
                wri.writeheader()
                wri.writerows(r)
            print(f"wrote {self.args.csv}\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', default='policy_pg.pt')
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='test')
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=20.0)
    p.add_argument('--settle', type=float, default=1.0)
    p.add_argument('--robot', default='burger')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--csv', default='')
    p.add_argument('--room-margin', type=float, default=0.3)
    p.add_argument('--jitter-xy', type=float, default=0.08)
    p.add_argument('--jitter-yaw-deg', type=float, default=8.0)
    p.add_argument('--jitter-seed', type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    poses = load_poses(args.config, args.starts)
    room = cfg['room']
    args.room_x_min, args.room_x_max = room['x_min'], room['x_max']
    args.room_y_min, args.room_y_max = room['y_min'], room['y_max']
    model, normalizer, ck = load_policy(args.policy)
    print(f"  loaded checkpoint from episode {ck.get('episode', '?')}")

    rclpy.init()
    node = PolicyRunner(args, model, normalizer, poses, cfg['obstacles'],
                        cfg['goal'], room)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
