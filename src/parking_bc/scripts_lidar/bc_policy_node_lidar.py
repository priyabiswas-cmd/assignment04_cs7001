#!/usr/bin/env python3
"""
Run a trained odometry+lidar policy in closed loop and score it - the
sensor-only counterpart of scripts/bc_policy_node.py.

    ros2 run parking_bc bc_policy_node_lidar.py --policy policy_lidar.pt \
        --starts test --trials 5

WHAT THE ROBOT SEES vs WHAT GRADES IT.  The policy's observation
(dx, dy, sin, cos[, v, w]) is built from ODOMETRY and the SENSED bay
position (lidar_gap.py, same detector as bc_dataset_lidar.py, run here
causally/online instead of with hindsight) - never from ground truth.
Success/failure, at the end of each trial, is still graded against the
TRUE bay position from Gazebo, exactly like the original policy node.
That's not cheating the robot - it's how you'd grade a real robot from
outside even though it doesn't know the answer either. It's what makes
the success rate comparable, apples to apples, with the original policy.

Before the lidar has ever seen a clean gap, the network has nothing to
condition its action on, so instead of guessing the robot creeps straight
forward at a fixed --search-speed (no network involved) until a detection
appears, then hands off to the trained policy. This mirrors the
demonstrations themselves - the human drove forward first and only saw a
clean near/far/near reading once close enough. Start poses sit farther
from the blocks than --near-thresh, by design, so this search phase is
expected to run for a second or two, not be instant. If a trial times out
still searching, that's useful information (retune --near-thresh/--fov-deg
or --search-speed for that start pose), not a training-data problem.

Same start poses, same --starts set (e.g. "test") as the original policy
node - config/start_poses.yaml is shared, unchanged, between both.
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
from sensor_msgs.msg import LaserScan

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch not found - install it to run a policy.")

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_gap import DEFAULTS, GapEstimator, find_gap_local, local_to_frame


# -- the same architecture bc_train_lidar.py saved (unchanged from
# bc_train.py - see bc_train_lidar.py's docstring for why) -----------------

class MLPPolicy(nn.Module):

    def __init__(self, obs_dim, chunk=1, hidden=(256, 256)):
        super().__init__()
        self.chunk = chunk
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 2 * chunk))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, self.chunk, 2) if self.chunk > 1 else y


def load_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = MLPPolicy(ck['obs_dim'], ck['chunk'], tuple(ck['hidden']))
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def default_config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, '..', 'config', 'start_poses.yaml'),
              os.path.join(here, '..', 'share', 'parking_bc',
                           'config', 'start_poses.yaml')):
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('parking_bc'),
                        'config', 'start_poses.yaml')


def load_poses(path, set_name):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if set_name not in cfg.get('sets', {}):
        raise SystemExit(f"set '{set_name}' not in {path}. "
                         f"available: {', '.join(sorted(cfg['sets']))}")
    return [(str(e['label']), float(e['x']), float(e['y']),
             math.radians(float(e['yaw']))) for e in cfg['sets'][set_name]]


class TemporalEnsemble:
    """ACT-style blending of overlapping action chunks (unchanged)."""

    def __init__(self, chunk, m=0.1):
        self.k = chunk
        self.m = m
        self.buf = []

    def add(self, chunk_pred):
        self.buf.append(chunk_pred)
        if len(self.buf) > self.k:
            self.buf.pop(0)

    def action(self):
        acts, weights = [], []
        n = len(self.buf)
        for i, pred in enumerate(self.buf):
            step = n - 1 - i
            if step >= self.k:
                continue
            acts.append(pred[step])
            weights.append(math.exp(-self.m * step))
        if not acts:
            return np.zeros(2, np.float32)
        a = np.array(acts, np.float32)
        w = np.array(weights, np.float32)[:, None]
        return (a * w).sum(0) / w.sum()


class PolicyRunner(Node):

    def __init__(self, args, model, ck, poses):
        super().__init__('bc_policy_lidar')
        self.args = args
        self.model = model
        self.ck = ck
        self.poses = poses

        self.obs_mean = np.asarray(ck['obs_mean'], np.float32)
        self.obs_std = np.asarray(ck['obs_std'], np.float32)
        self.act_mean = np.asarray(ck['act_mean'], np.float32)
        self.act_std = np.asarray(ck['act_std'], np.float32)
        self.chunk = int(ck['chunk'])
        self.use_history = bool(ck['history'])

        self.pose = None        # ground truth - grading ONLY
        self.odom = None        # (x, y, yaw) - what the policy senses
        self.scan = None
        self.gap = None         # GapEstimator, reset each trial
        self.v_prev = 0.0
        self.w_prev = 0.0
        self.ens = TemporalEnsemble(self.chunk) if self.chunk > 1 else None

        self.trial = 0
        self.pose_i = 0
        self.rep = 0
        self.results = []
        self.t_start = None
        self.still_since = None
        self.running = False
        self.waited_for_detection = False

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self.on_states, qos)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        self.create_subscription(LaserScan, args.scan_topic, self.on_scan, qos)
        self.pub = self.create_publisher(Twist, args.cmd_out, 10)
        self.reset_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(1.0 / args.rate, self.step)

        print(f"\npolicy   {os.path.basename(args.policy)}  (odometry+lidar)")
        print(f"  obs {ck['obs_dim']}  chunk {self.chunk}  "
              f"history {self.use_history}")
        print(f"  val loss {ck.get('val_loss', float('nan')):.5f}")
        print(f"  detector  near<{args.near_thresh:.2f}m  fov+-{args.fov_deg:.0f}deg  "
              f"min_run {args.min_run}  lock {args.lock_count}@{args.lock_tol:.2f}m")
        print(f"starts   '{args.starts}'  x {len(poses)}  "
              f"x {args.trials} trials  (same set/config as the original "
              "policy node)\n")
        self.begin_trial()

    # -- plumbing ----------------------------------------------------------

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

    def on_scan(self, msg):
        self.scan = msg

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

        label, x, y, yaw = self.poses[self.pose_i]
        self.publish(0.0, 0.0)
        self.teleport(x, y, yaw)

        self.v_prev = self.w_prev = 0.0
        self.ens = TemporalEnsemble(self.chunk) if self.chunk > 1 else None
        self.gap = GapEstimator(lock_count=self.args.lock_count,
                                lock_tol=self.args.lock_tol)
        self.t_start = time.time()
        self.still_since = None
        self.running = True
        self.waited_for_detection = False
        print(f"  trial {self.trial + 1:2d}  start {label} "
              f"(rep {self.rep + 1}/{self.args.trials}) ... ", end='',
              flush=True)

    def finish_trial(self, outcome, dx, dy, yaw):
        self.publish(0.0, 0.0)
        self.running = False

        in_bay = (abs(dx) <= self.args.bay_len / 2.0
                  and -0.02 <= dy <= self.args.bay_depth)
        success = in_bay and outcome == 'stopped'
        off = min(abs(wrap_angle(yaw)), abs(wrap_angle(yaw - math.pi)))

        self.results.append(dict(
            label=self.poses[self.pose_i][0], outcome=outcome,
            success=success, dx=dx, dy=dy,
            off_parallel_deg=math.degrees(off),
            dist=math.hypot(dx, dy - self.args.bay_depth / 2.0)))

        print(f"{'OK ' if success else 'no '} "
              f"[{outcome}] dx={dx:+.3f} dy={dy:+.3f} "
              f"yaw_off={math.degrees(off):4.0f}d")

        self.trial += 1
        self.rep += 1
        if self.rep >= self.args.trials:
            self.rep = 0
            self.pose_i += 1
        time.sleep(0.4)
        self.begin_trial()

    def _forward_blocked(self):
        """True if something is closer than --stop-dist within a narrow
        cone straight ahead - used only to brake the blind search creep,
        never consulted once the trained policy is driving."""
        if self.scan is None:
            return False
        fov = math.radians(self.args.stop_fov_deg)
        s = self.scan
        n = len(s.ranges)
        for i in range(n):
            angle = wrap_angle(s.angle_min + i * s.angle_increment)
            if abs(angle) <= fov:
                r = s.ranges[i]
                if math.isfinite(r) and 0.0 < r < self.args.stop_dist:
                    return True
        return False

    # -- control loop --------------------------------------------------------

    def step(self):
        if not self.running or self.pose is None or self.odom is None:
            return

        # -- perception: update the sensed bay estimate from THIS scan ----
        if self.scan is not None:
            local = find_gap_local(
                self.scan.ranges, self.scan.angle_min,
                self.scan.angle_increment,
                near_thresh=self.args.near_thresh,
                min_run=self.args.min_run, fov_deg=self.args.fov_deg)
            if local is not None:
                ox, oy, oyaw = self.odom
                self.gap.update(local_to_frame(local[0], local[1], ox, oy, oyaw))

        # IMPORTANT: only act on a LOCKED estimate, never a single hit or a
        # still-drifting running mean. Before it locks, the estimate moves
        # as the robot moves (it's built from whatever's briefly in view),
        # so dx computed against it stays roughly constant no matter where
        # the robot actually is - the network never sees a real position
        # error to correct and just keeps doing "drive forward", which is
        # exactly the "goes straight into whatever's ahead" bug. bc_dataset
        # _lidar.py never has this problem because it only ever uses the
        # FINAL, converged estimate for a whole episode - this is the
        # online equivalent of waiting for that same convergence.
        est = self.gap.locked

        # -- grading uses ground truth, never the estimate -----------------
        x, y, yaw = self.pose
        dx_true = x - self.args.bay_x
        dy_true = y - self.args.bay_y

        elapsed = time.time() - self.t_start

        if est is None:
            # Not locked on yet - the network has nothing trustworthy to
            # condition on. Start poses are placed too far from the blocks
            # for "near" to ever trigger while standing still (this is
            # exactly why your test starts stalled before this fix), so
            # creep straight forward - no network involved yet - until a
            # LOCKED detection appears, then hand off to the trained
            # policy. This mirrors how the demonstrations themselves
            # worked: the human drove forward first and only got a clean,
            # stable view once closer.
            if not self.waited_for_detection:
                print("(searching for the gap - creeping forward) ",
                      end='', flush=True)
                self.waited_for_detection = True
            # Blind creep has no idea what's ahead of it - if a start pose
            # happens to point roughly at a block instead of the gap (some
            # of yours do), driving forward with no check WILL ram it
            # before a lock ever forms. Brake using the raw scan itself
            # (not the network - this is a dumb safety net, not skill) so
            # the worst case is "stopped short, still searching" instead
            # of a collision. Detection keeps running off whatever the
            # scan sees peripherally even while stopped.
            blocked = self._forward_blocked()
            self.publish(0.0 if blocked else self.args.search_speed, 0.0)
            if elapsed > self.args.timeout:
                self.finish_trial('timeout', dx_true, dy_true, yaw)
            return

        ox, oy, oyaw = self.odom
        obs = [ox - est[0], oy - est[1], math.sin(oyaw), math.cos(oyaw)]
        if self.use_history:
            obs += [self.v_prev, self.w_prev]
        z = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std

        with torch.no_grad():
            out = self.model(torch.tensor(z).unsqueeze(0)).squeeze(0).numpy()

        if self.chunk > 1:
            self.ens.add(out)
            a_n = self.ens.action()
        else:
            a_n = out

        v, w = a_n * self.act_std + self.act_mean
        v = float(np.clip(v, -0.22, 0.22))
        w = float(np.clip(w, -2.84, 2.84))
        self.publish(v, w)
        self.v_prev, self.w_prev = v, w

        # -- termination (identical thresholds to the original) -----------
        moving = abs(v) > 0.01 or abs(w) > 0.05

        if not moving:
            if self.still_since is None:
                self.still_since = time.time()
            elif time.time() - self.still_since > self.args.settle:
                self.finish_trial('stopped', dx_true, dy_true, yaw)
                return
        else:
            self.still_since = None

        if elapsed > self.args.timeout:
            self.finish_trial('timeout', dx_true, dy_true, yaw)
        elif abs(dx_true) > 3.0 or dy_true > 3.0 or dy_true < -0.5:
            self.finish_trial('left arena', dx_true, dy_true, yaw)

    # -- report ------------------------------------------------------------

    def report(self):
        r = self.results
        if not r:
            return
        n = len(r)
        ok = sum(x['success'] for x in r)

        print("\n" + "=" * 58)
        print(f" odometry+lidar policy: success {ok}/{n}  "
              f"({100.0 * ok / n:.0f}%)")

        by = {}
        for x in r:
            by.setdefault(x['label'], []).append(x)
        print("\n per start:")
        for label in sorted(by):
            g = by[label]
            s = sum(z['success'] for z in g)
            print(f"   {label}:  {s}/{len(g)}")

        outcomes = {}
        for x in r:
            outcomes[x['outcome']] = outcomes.get(x['outcome'], 0) + 1
        print("\n outcomes: " + ", ".join(
            f"{k} {v}" for k, v in sorted(outcomes.items())))

        d = np.array([x['dist'] for x in r])
        print(f"\n distance to bay centre  "
              f"mean {d.mean():.3f} m   best {d.min():.3f} m")
        if ok:
            sd = np.array([x['dist'] for x in r if x['success']])
            print(f" (successful trials only) mean {sd.mean():.3f} m")
        print("=" * 58 + "\n")
        print(" compare this against the original ground-truth policy's "
              "report from\n scripts/bc_policy_node.py on the same "
              f"'--starts {self.args.starts}' set.\n")

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
    p.add_argument('--policy', default='policy_lidar.pt')
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='test',
                   help="pose set from start_poses.yaml (default: test - "
                        "same set/positions the original policy is tested on)")
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--settle', type=float, default=1.5)
    p.add_argument('--bay-x', type=float, default=0.0,
                   help='ground truth, for GRADING only - never given to '
                        'the policy')
    p.add_argument('--bay-y', type=float, default=0.16)
    p.add_argument('--bay-len', type=float, default=0.4416)
    p.add_argument('--bay-depth', type=float, default=0.32)
    p.add_argument('--robot', default='burger')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--scan-topic', default='/scan')
    p.add_argument('--csv', default='')
    p.add_argument('--search-speed', type=float, default=0.08,
                   help='m/s to creep forward, no network involved, while '
                        'waiting for the first gap detection (default 0.08)')
    p.add_argument('--stop-dist', type=float, default=0.25,
                   help='m - brake the blind search creep if anything gets '
                        'this close straight ahead (default 0.25)')
    p.add_argument('--stop-fov-deg', type=float, default=20.0,
                   help='+/- degrees of straight-ahead checked for the '
                        'search-creep braking (default 20)')
    p.add_argument('--near-thresh', type=float, default=DEFAULTS['near_thresh'])
    p.add_argument('--min-run', type=int, default=DEFAULTS['min_run'])
    p.add_argument('--fov-deg', type=float, default=DEFAULTS['fov_deg'])
    p.add_argument('--lock-count', type=int, default=DEFAULTS['lock_count'])
    p.add_argument('--lock-tol', type=float, default=DEFAULTS['lock_tol'])
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    poses = load_poses(args.config, args.starts)
    model, ck = load_policy(args.policy)

    rclpy.init()
    node = PolicyRunner(args, model, ck, poses)
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
