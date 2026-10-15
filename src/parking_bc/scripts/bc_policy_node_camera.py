#!/usr/bin/env python3
"""
Run a camera-based policy in closed loop and score it.

    ros2 run parking_bc bc_policy_node_camera.py --policy policy_cam.pt
    ros2 run parking_bc bc_policy_node_camera.py --policy policy_cam.pt \
        --starts test --trials 5

For each trial, the robot is teleported to a start pose and the policy
drives using ONLY the camera image - the true pose is read here purely to
know when a trial has ended and to score it, exactly like the pose-based
evaluator. The policy itself never sees it.

THIS IS THE NUMBER THAT MATTERS, not the training loss. A camera policy
can look excellent offline and still fail closed loop, because its own
small errors put it in front of views the expert never showed it.
"""

import argparse
import math
import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import yaml

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch not found - install it to run a policy.")

from bc_train_camera import CNNPolicy
from record_expert_camera import decode_image, resize


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


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


def load_policy(path):
    """
    Reconstruct the model to exactly match what was saved.

    Whether dropout was used at train time changes the layer indices
    inside `head` (an extra Dropout shifts everything after it), so
    building the model with the wrong dropout setting fails to load with
    a confusing "head.3 vs head.4" key mismatch. The actual VALUE of
    dropout is irrelevant here - it is switched off in eval() mode
    regardless - so this detects the right shape directly from the
    checkpoint's own keys rather than trusting a separately recorded
    setting that could drift out of sync with the weights.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    had_dropout = any(k.startswith('head.4') for k in ck['state_dict'])
    model = CNNPolicy(int(ck['img_size']), ck['img_mean'], ck['img_std'],
                      chunk=int(ck['chunk']),
                      dropout=0.1 if had_dropout else 0.0)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck


class TemporalEnsemble:
    """ACT-style blending of overlapping action-chunk predictions."""

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


class CameraPolicyRunner(Node):

    def __init__(self, args, model, ck, poses):
        super().__init__('bc_policy_camera')
        self.args = args
        self.model = model
        self.ck = ck
        self.poses = poses
        self.img_size = int(ck['img_size'])
        self.chunk = int(ck['chunk'])

        self.pose = None
        self.img = None
        self.ens = TemporalEnsemble(self.chunk) if self.chunk > 1 else None

        self.trial = 0
        self.pose_i = 0
        self.rep = 0
        self.results = []
        self.t_start = None
        self.still_since = None
        self.running = False

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self.on_states, qos)
        self.create_subscription(
            Image, args.camera_topic, self.on_image, qos)
        self.pub = self.create_publisher(Twist, args.cmd_out, 10)
        self.reset_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(1.0 / args.rate, self.step)

        print(f"\npolicy   {os.path.basename(args.policy)}")
        print(f"  image {self.img_size}x{self.img_size}  chunk {self.chunk}")
        print(f"  val loss {ck.get('val_loss', float('nan')):.5f}")
        print(f"starts   '{args.starts}'  x {len(poses)}  "
              f"x {args.trials} trials\n")
        self.begin_trial()

    # -- plumbing ------------------------------------------------------

    def on_states(self, msg):
        try:
            i = msg.name.index(self.args.robot)
        except ValueError:
            return
        p = msg.pose[i]
        yaw = math.atan2(2.0 * (p.orientation.w * p.orientation.z
                                +p.orientation.x * p.orientation.y),
                         1.0 - 2.0 * (p.orientation.y ** 2
                                     +p.orientation.z ** 2))
        self.pose = (p.position.x, p.position.y, yaw)

    def on_image(self, msg):
        try:
            img = decode_image(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        self.img = resize(img, self.img_size)

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

    # -- trials ----------------------------------------------------------

    def begin_trial(self):
        if self.pose_i >= len(self.poses):
            self.report()
            raise KeyboardInterrupt

        label, x, y, yaw = self.poses[self.pose_i]
        self.publish(0.0, 0.0)
        self.pose = None                 # wait for a FRESH pose post-teleport
        self.teleport(x, y, yaw)

        self.ens = TemporalEnsemble(self.chunk) if self.chunk > 1 else None
        self.t_start = time.time()
        self.still_since = None
        self.running = True
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
              f"[{outcome}] dx={dx:+.3f} dy={dy:+.3f}")

        self.trial += 1
        self.rep += 1
        if self.rep >= self.args.trials:
            self.rep = 0
            self.pose_i += 1
        time.sleep(0.4)
        self.begin_trial()

    # -- control loop ------------------------------------------------------

    def step(self):
        if not self.running or self.pose is None or self.img is None:
            return

        x, y, yaw = self.pose
        dx = x - self.args.bay_x
        dy = y - self.args.bay_y

        xb = torch.tensor(self.img[None], dtype=torch.float32)
        with torch.no_grad():
            out = self.model(xb).squeeze(0).numpy()

        if self.chunk > 1:
            self.ens.add(out)
            a_n = self.ens.action()
        else:
            a_n = out

        v, w = a_n * self.ck['act_std'] + self.ck['act_mean']
        v = float(np.clip(v, -0.22, 0.22))
        w = float(np.clip(w, -2.84, 2.84))
        self.publish(v, w)

        elapsed = time.time() - self.t_start
        moving = abs(v) > 0.01 or abs(w) > 0.05
        if not moving:
            if self.still_since is None:
                self.still_since = time.time()
            elif time.time() - self.still_since > self.args.settle:
                self.finish_trial('stopped', dx, dy, yaw)
                return
        else:
            self.still_since = None

        if elapsed > self.args.timeout:
            self.finish_trial('timeout', dx, dy, yaw)
        elif abs(dx) > 3.0 or dy > 3.0 or dy < -0.5:
            self.finish_trial('left arena', dx, dy, yaw)

    # -- report ------------------------------------------------------------

    def report(self):
        r = self.results
        if not r:
            return
        n = len(r)
        ok = sum(x['success'] for x in r)

        print("\n" + "=" * 58)
        print(f" success {ok}/{n}  ({100.0 * ok / n:.0f}%)")

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
        print(f"\n distance to bay centre  mean {d.mean():.3f} m")
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
    p.add_argument('--policy', default='policy_cam.pt')
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='right',
                   help="pose set from start_poses.yaml (default: right)")
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=60.0)
    p.add_argument('--settle', type=float, default=1.5)
    p.add_argument('--bay-x', type=float, default=0.0)
    p.add_argument('--bay-y', type=float, default=0.16)
    p.add_argument('--bay-len', type=float, default=0.4416)
    p.add_argument('--bay-depth', type=float, default=0.32)
    p.add_argument('--robot', default='burger')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--camera-topic', default='/camera/image_raw')
    p.add_argument('--csv', default='')
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    poses = load_poses(args.config, args.starts)
    model, ck = load_policy(args.policy)

    rclpy.init()
    node = CameraPolicyRunner(args, model, ck, poses)
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
