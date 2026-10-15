#!/usr/bin/env python3
"""
Auto-generate a camera-based parking dataset using the scripted expert.

No driving needed - the expert steers, this node saves what the CAMERA
sees alongside the action the expert took. Ground-truth pose is read
internally so the expert can compute the correct action, but it is never
what the policy will be trained on.

    # smoke test: one start, a handful of episodes
    ros2 run parking_bc record_expert_camera.py --only A --reps 5

    # full collection across all starts in a set
    ros2 run parking_bc record_expert_camera.py --starts right --reps 10

Each episode is saved as data_camera/start<L>_<NNN>.p, containing:

    images   (T, S, S, C) uint8   - what the camera saw, resized to SxS
    actions  (T, 2)       float32 - the expert's (v, w) at each frame
    meta     dict  - includes x/y/yaw for each frame too, kept ONLY for
                     plotting and debugging. bc_dataset_camera.py must not
                     feed these to the network - that would defeat the
                     point of using a camera at all.
"""

import argparse
import glob
import math
import os
import pickle
import re
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import yaml

from expert import ExpertParams, expert_action, wrap_angle


# -- reused conventions, kept local so this script runs standalone --------

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


def next_index(data_dir, label):
    used = set()
    for p in glob.glob(os.path.join(data_dir, f"start{label}_*.p")):
        m = re.search(rf"start{label}_(\d+)\.p$", os.path.basename(p))
        if m:
            used.add(int(m.group(1)))
    i = 1
    while i in used:
        i += 1
    return i


# -- image decoding ---------------------------------------------------------

_CHANNELS = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}


def decode_image(msg):
    """
    sensor_msgs/Image -> (H, W, C) uint8 numpy array.

    Channel ORDER is left exactly as the camera published it (rgb8 vs bgr8
    is not resolved here) - a CNN does not care which channel is which, as
    long as it is the same for every frame. The encoding is recorded in the
    episode's metadata for anyone who wants to visualise a frame correctly
    later (cv2 expects bgr8; matplotlib expects rgb8).
    """
    c = _CHANNELS.get(msg.encoding)
    if c is None:
        raise ValueError(f"unsupported camera encoding '{msg.encoding}' - "
                         f"add it to _CHANNELS if you need it")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    arr = arr.reshape(msg.height, msg.step)      # step may include padding
    arr = arr[:, :msg.width * c].reshape(msg.height, msg.width, c)
    return arr


def resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------

class CameraExpertRecorder(Node):

    def __init__(self, args, starts):
        super().__init__('record_expert_camera')
        self.args = args
        self.starts = starts
        self.params = ExpertParams(bay_x=args.bay_x, bay_y=args.bay_y)

        self.pose = None
        self.img = None
        self.img_encoding = None

        self.idx = 0
        self.rep = 0
        self.buf_images = []
        self.buf_actions = []
        self.buf_x, self.buf_y, self.buf_yaw = [], [], []
        self.t0 = None
        self.active = False
        self.saved = 0
        self.skipped = 0
        self.rng = np.random.default_rng(args.seed)

        # Trajectory noise: a short burst added to the EXECUTED steering,
        # refreshed periodically. The LOGGED action is always the clean
        # expert output recomputed at wherever the robot actually is -
        # never the noisy one. This teaches recovery for free: once
        # perturbed, the feedback expert automatically shows the correct
        # way back, with no special-casing needed.
        self.noise_w = 0.0
        self.noise_until = 0.0

        os.makedirs(args.data_dir, exist_ok=True)

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

        print(f"\nrecording camera + expert-action pairs")
        print(f"  starts   {[s[0] for s in starts]}  x {args.reps} reps")
        print(f"  image    {args.img_size}x{args.img_size}")
        print(f"  data ->  {args.data_dir}/\n")
        self.begin_episode()

    # -- callbacks -----------------------------------------------------

    def on_states(self, msg):
        try:
            i = msg.name.index(self.args.robot)
        except ValueError:
            return
        p = msg.pose[i]
        self.pose = (p.position.x, p.position.y,
                     yaw_from_quaternion(p.orientation.x, p.orientation.y,
                                         p.orientation.z, p.orientation.w))

    def on_image(self, msg):
        try:
            img = decode_image(msg)
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        self.img = resize(img, self.args.img_size)
        self.img_encoding = msg.encoding

    def publish(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub.publish(m)

    # -- episode lifecycle -----------------------------------------------

    def begin_episode(self):
        if self.idx >= len(self.starts):
            self.report()
            raise KeyboardInterrupt

        label, nx, ny, nyaw = self.starts[self.idx]

        # Jitter around the nominal pose so repeated episodes from the same
        # labelled start are not literally identical. Without this, a
        # deterministic expert plus a fixed start produces the exact same
        # image sequence every time - the network would have nothing to
        # generalise from, just one trajectory to memorise.
        x = nx + self.rng.uniform(-self.args.jitter_x, self.args.jitter_x)
        y = ny + self.rng.uniform(-self.args.jitter_y, self.args.jitter_y)
        yaw = nyaw + math.radians(
            self.rng.uniform(-self.args.jitter_yaw, self.args.jitter_yaw))

        self.publish(0.0, 0.0)

        # Forget the last known pose. Otherwise, if the teleport below has
        # not been processed by Gazebo yet when the next control tick
        # fires, step() would read the OLD position - which, right after a
        # successful episode, is sitting exactly at the goal, and the
        # expert would instantly report "done" with zero samples logged.
        # Setting this to None makes step() wait for a fresh pose that
        # actually reflects the teleport, instead of guessing a delay.
        self.pose = None

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

        self.buf_images, self.buf_actions = [], []
        self.buf_x, self.buf_y, self.buf_yaw = [], [], []
        self.buf_noisy = []
        self.noise_w, self.noise_until = 0.0, 0.0
        self.t0 = time.time()
        self.active = True
        print(f"  [{label}] rep {self.rep + 1}/{self.args.reps}  "
              f"(start x={x:+.3f} y={y:+.3f} "
              f"yaw={math.degrees(yaw):+.0f}) ... ", end='', flush=True)

    def step(self):
        if not self.active or self.pose is None or self.img is None:
            return

        x, y, yaw = self.pose
        v, w, done = expert_action(x, y, yaw, self.params)   # clean - LOGGED
        dist_to_goal = math.hypot(x - self.args.bay_x, y - self.args.bay_y)

        v_exec, w_exec = v, w
        if self.args.traj_noise > 0 and dist_to_goal > self.args.noise_stop_radius:
            now = time.time()
            if now > self.noise_until:
                if self.rng.uniform() < self.args.traj_noise_prob:
                    self.noise_w = self.rng.uniform(
                        -self.args.traj_noise, self.args.traj_noise)
                else:
                    self.noise_w = 0.0
                self.noise_until = now + self.rng.uniform(0.3, 0.6)
            w_exec = max(-self.params.max_w,
                        min(self.params.max_w, w + self.noise_w))

        elapsed = time.time() - self.t0
        timeout = elapsed > self.args.timeout
        left_arena = (abs(x - self.args.bay_x) > 3.0
                      or y - self.args.bay_y > 3.0
                      or y - self.args.bay_y < -0.5)

        if not (done or timeout or left_arena):
            self.buf_images.append(self.img.copy())
            self.buf_actions.append((v, w))          # clean label, always
            self.buf_x.append(x)
            self.buf_y.append(y)
            self.buf_yaw.append(yaw)
            self.buf_noisy.append(abs(w_exec - w) > 1e-6)
            self.publish(v_exec, w_exec)              # what actually happens
            return

        self.publish(0.0, 0.0)
        self.active = False
        outcome = 'done' if done else ('timeout' if timeout else 'left arena')
        self.finish_episode(outcome)

    def finish_episode(self, outcome):
        label = self.starts[self.idx][0]
        n = len(self.buf_actions)

        if n < self.args.min_samples or outcome != 'done':
            print(f"skipped ({outcome}, {n} samples)")
            self.skipped += 1
        else:
            i = next_index(self.args.data_dir, label)
            out = os.path.join(self.args.data_dir, f"start{label}_{i:03d}.p")
            episode = dict(
                images=np.stack(self.buf_images).astype(np.uint8),
                actions=np.array(self.buf_actions, np.float32),
                meta=dict(
                    start_label=label,
                    start_pose_nominal=self.starts[self.idx][1:],
                    x=np.array(self.buf_x, np.float32),
                    y=np.array(self.buf_y, np.float32),
                    yaw=np.array(self.buf_yaw, np.float32),
                    noisy=np.array(self.buf_noisy, bool),
                    traj_noise=self.args.traj_noise,
                    bay_centre=(self.args.bay_x, self.args.bay_y),
                    img_size=self.args.img_size,
                    camera_encoding=self.img_encoding,
                    camera_topic=self.args.camera_topic,
                    rate=self.args.rate,
                    n_samples=n,
                    duration_s=self.buf_x and time.time() - self.t0 or 0.0,
                    outcome=outcome,
                    recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                ),
            )
            with open(out, 'wb') as f:
                pickle.dump(episode, f)
            self.saved += 1
            pct = 100.0 * sum(self.buf_noisy) / max(n, 1)
            print(f"saved {os.path.basename(out)}  "
                  f"({n} frames, {pct:.0f}% under noise)")

        self.rep += 1
        if self.rep >= self.args.reps:
            self.rep = 0
            self.idx += 1
        time.sleep(0.3)
        self.begin_episode()

    def report(self):
        print("\n" + "=" * 50)
        print(f" saved {self.saved} episodes, skipped {self.skipped}")
        for label, *_ in self.starts:
            n = len(glob.glob(os.path.join(
                self.args.data_dir, f"start{label}_*.p")))
            print(f"   start {label}: {n} on disk")
        print("=" * 50 + "\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_camera')
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='right',
                   help="pose set from start_poses.yaml (default: right)")
    p.add_argument('--only', metavar='LABEL', default=None,
                   help='record just this one start - use for the smoke '
                        'test, e.g. --only A')
    p.add_argument('--reps', type=int, default=10,
                   help='episodes per start (default 10)')
    p.add_argument('--img-size', type=int, default=64)
    p.add_argument('--jitter-x', type=float, default=0.05,
                   help='+/- start-pose jitter in x, metres (default 0.05)')
    p.add_argument('--jitter-y', type=float, default=0.03,
                   help='+/- start-pose jitter in y, metres (default 0.03)')
    p.add_argument('--jitter-yaw', type=float, default=8.0,
                   help='+/- start-pose jitter in yaw, degrees (default 8)')
    p.add_argument('--traj-noise', type=float, default=0.20,
                   help='burst magnitude added to steering (rad/s), '
                        '0 disables (default 0.20)')
    p.add_argument('--traj-noise-prob', type=float, default=0.6,
                   help='chance a new burst starts when the previous one '
                        'ends (default 0.6)')
    p.add_argument('--noise-stop-radius', type=float, default=0.25,
                   help='disable noise within this distance of the bay, '
                        'so the final approach is not perturbed (default 0.25)')
    p.add_argument('--seed', type=int, default=0,
                   help='RNG seed for jitter, for reproducible datasets')
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=40.0)
    p.add_argument('--min-samples', type=int, default=15,
                   help='discard episodes shorter than this (default 15)')
    p.add_argument('--robot', default='burger')
    p.add_argument('--bay-x', type=float, default=0.0)
    p.add_argument('--bay-y', type=float, default=0.16)
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--camera-topic', default='/camera/image_raw')
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    starts = load_poses(args.config, args.starts)
    if args.only:
        starts = [s for s in starts if s[0] == args.only]
        if not starts:
            raise SystemExit(f"no start labelled '{args.only}' in "
                             f"set '{args.starts}'")

    rclpy.init()
    node = CameraExpertRecorder(args, starts)
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
