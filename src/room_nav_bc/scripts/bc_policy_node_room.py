#!/usr/bin/env python3
"""
Run a trained camera+odometry policy in closed loop and score it.

    ros2 run room_nav_bc bc_policy_node_room.py --policy policy_room.pt \
        --starts test --trials 5

WHAT THE ROBOT SEES vs WHAT GRADES IT.  The policy's observation is the
live camera frame (resized to whatever it was trained on) plus odometry
relative to THIS TRIAL's own start - never ground truth, never told where
the island/gap/goal are. Success/failure and the clearance diagnostic are
computed from ground truth (/gazebo/model_states), exactly like every
other policy node in this project - the robot doesn't get to see the
answer, the grader does.

THE CLEARANCE CHECK IS AGAINST EVERYTHING, NOT JUST THE ISLAND.  Every
obstacle in the room (island, both gap boxes, both decorative landmark
pairs) AND the four walls are checked every step, robot radius included,
so "what did it hit" is an honest answer rather than only ever being able
to say "island" or nothing. If it actually collided, the report names the
specific thing (e.g. "gap_left", "wall") instead of silently missing it.

WHAT TO EXPECT.  If bc_train_room.py trained a plain regression policy on
the balanced left/right dataset, watch for it heading toward the island
and either stalling or driving into it near the choice point - that is
the averaging failure mode this whole exercise exists to demonstrate, not
necessarily a bug in this script. The final report's clearance numbers
quantify exactly how close it got and to what.
"""

import argparse
import math
import os
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch not found - install it to run a policy.")

import yaml


# -- the same architecture bc_train_room.py saved ---------------------------

class CNNPolicy(nn.Module):

    def __init__(self, odom_dim, img_size=64, chunk=1, img_embed=128,
                hidden=(128, 64)):
        super().__init__()
        self.chunk = chunk
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        conv_hw = img_size // 8
        self.img_fc = nn.Sequential(
            nn.Linear(32 * conv_hw * conv_hw, img_embed), nn.ReLU())
        layers, d = [], img_embed + odom_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 2 * chunk))
        self.head = nn.Sequential(*layers)

    def forward(self, img, odom):
        z = self.conv(img).flatten(1)
        z = self.img_fc(z)
        z = torch.cat([z, odom], dim=1)
        y = self.head(z)
        return y.view(-1, self.chunk, 2) if self.chunk > 1 else y


def load_policy(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = CNNPolicy(ck['odom_dim'], ck['img_size'], ck['chunk'],
                      ck['img_embed'], tuple(ck['hidden']))
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
    for c in (os.path.join(here, '..', 'config', 'start_goal.yaml'),
              os.path.join(here, '..', 'share', 'room_nav_bc',
                           'config', 'start_goal.yaml')):
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('room_nav_bc'),
                        'config', 'start_goal.yaml')


def load_poses(path, set_name):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if set_name not in cfg.get('sets', {}):
        raise SystemExit(f"set '{set_name}' not in {path}. "
                         f"available: {', '.join(sorted(cfg['sets']))}")
    return [(str(e['label']), float(e['x']), float(e['y']),
             math.radians(float(e['yaw']))) for e in cfg['sets'][set_name]]


def in_goal(x, y, goal):
    return (abs(x - goal['x']) <= goal['w'] / 2.0
            and abs(y - goal['y']) <= goal['h'] / 2.0)


def point_box_clearance(px, py, box):
    dx = max(abs(px - box['x']) - box['w'] / 2.0, 0.0)
    dy = max(abs(py - box['y']) - box['h'] / 2.0, 0.0)
    return math.hypot(dx, dy)


MAX_LIN = 0.22
MAX_ANG = 2.84

# TurtleBot3 Burger footprint 0.138 x 0.178 m -> circumscribed radius.
# Same figure used in verify_expert_room.py.
ROBOT_RADIUS = math.hypot(0.178 / 2.0, 0.138 / 2.0)   # ~0.113 m


def worst_clearance(x, y, obstacles, room):
    """Surface clearance (robot radius already subtracted) to the closest
    of every obstacle AND the four room walls - not just one obstacle.
    -> (clearance, name). Negative means an actual collision."""
    best, which = float('inf'), None
    for name, box in obstacles.items():
        c = point_box_clearance(x, y, box) - ROBOT_RADIUS
        if c < best:
            best, which = c, name
    wall_c = min(x - room['x_min'], room['x_max'] - x,
                y - room['y_min'], room['y_max'] - y) - ROBOT_RADIUS
    if wall_c < best:
        best, which = wall_c, 'wall'
    return best, which


class PolicyRunner(Node):

    def __init__(self, args, model, ck, poses, obstacles, goal):
        super().__init__('bc_policy_room')
        self.args = args
        self.model = model
        self.ck = ck
        self.poses = poses
        self.obstacles = obstacles
        self.goal = goal
        self.room = dict(x_min=args.room_x_min, x_max=args.room_x_max,
                         y_min=args.room_y_min, y_max=args.room_y_max)
        self.bridge = CvBridge()

        self.odom_mean = np.asarray(ck['odom_mean'], np.float32)
        self.odom_std = np.asarray(ck['odom_std'], np.float32)
        self.act_mean = np.asarray(ck['act_mean'], np.float32)
        self.act_std = np.asarray(ck['act_std'], np.float32)
        self.img_size = ck['img_size']
        self.chunk = int(ck['chunk'])

        self.pose = None        # ground truth - grading ONLY
        self.odom = None        # what the policy senses
        self.frame = None
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
        self.step_log = []          # every control step, all trials
        self.override = None        # fn(pv, pw) -> (v, w, human) for takeover
        self.sample_hook = None     # fn(sample dict) called every control step
        self.trial_outcomes = {}    # trial idx -> outcome dict
        self.rng = np.random.default_rng(args.jitter_seed)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self.on_states, qos)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        self.create_subscription(
            Image, args.image_topic, self.on_image, qos)
        self.pub = self.create_publisher(Twist, args.cmd_out, 10)
        self.reset_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(1.0 / args.rate, self.step)

        print(f"\npolicy  {os.path.basename(args.policy)}  "
              "(camera + odometry, no map)")
        print(f"  odom {ck['odom_dim']}  img {self.img_size}x{self.img_size}"
              f"  chunk {self.chunk}")
        print(f"  val loss {ck.get('val_loss', float('nan')):.5f}")
        print(f"starts  '{args.starts}'  x {len(poses)}  "
              f"x {args.trials} trials\n")
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

    def on_image(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print(f"!! image conversion failed: {e}")

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
        # Never test from the exact recorded/config pose - jitter it a
        # little each trial, so this is always a nearby-but-different
        # start, not a memorised one.
        x = x0 + self.rng.uniform(-self.args.jitter_xy, self.args.jitter_xy)
        y = y0 + self.rng.uniform(-self.args.jitter_xy, self.args.jitter_xy)
        yaw = yaw0 + math.radians(
            self.rng.uniform(-self.args.jitter_yaw_deg, self.args.jitter_yaw_deg))
        self.publish(0.0, 0.0)
        self.teleport(x, y, yaw)
        # see expert_demo_room.py's begin_episode() for why this matters:
        # avoids acting on a pose/odom/frame left over from the end of the
        # PREVIOUS trial before a fresh post-teleport reading arrives.
        self.pose = None
        self.odom = None
        self.frame = None
        self.odom_origin = None

        self.t_start = time.time()
        self.still_since = None
        self.running = True
        self.min_clear = float('inf')
        self.min_clear_obs = None
        self.cur_trial = dict(idx=self.trial, label=label,
                              start_x=x, start_y=y, start_yaw=yaw)
        print(f"  trial {self.trial + 1:2d}  start {label} "
              f"(rep {self.rep + 1}/{self.args.trials}, jittered to "
              f"x={x:+.2f} y={y:+.2f} yaw={math.degrees(yaw):+.0f}) ... ",
              end='',
              flush=True)

    def finish_trial(self, outcome, x, y, yaw):
        self.publish(0.0, 0.0)
        self.running = False

        success = outcome == 'stopped' and in_goal(x, y, self.goal)
        collided = self.min_clear < 0.0   # actual contact, robot radius included

        self.results.append(dict(
            label=self.poses[self.pose_i][0], outcome=outcome,
            success=success, x=x, y=y,
            min_clear=self.min_clear, min_clear_obs=self.min_clear_obs,
        ))
        self.trial_outcomes[self.cur_trial['idx']] = dict(
            outcome=outcome, success=success, collided=collided,
            min_clear=self.min_clear, min_clear_obs=self.min_clear_obs)

        note = f"  [HIT {self.min_clear_obs}]" if collided else ""
        print(f"{'OK ' if success else 'no '} [{outcome}] "
              f"pos=({x:+.2f},{y:+.2f})  "
              f"clearance={self.min_clear:+.3f}m vs {self.min_clear_obs}"
              f"{note}")

        self.trial += 1
        self.rep += 1
        if self.rep >= self.args.trials:
            self.rep = 0
            self.pose_i += 1
        time.sleep(0.4)
        self.begin_trial()

    # -- control loop --------------------------------------------------------

    def step(self):
        if not self.running or self.pose is None or self.odom is None \
                or self.frame is None:
            return

        if self.odom_origin is None:
            self.odom_origin = self.odom

        x, y, yaw = self.pose
        c, which = worst_clearance(x, y, self.obstacles, self.room)
        if c < self.min_clear:
            self.min_clear, self.min_clear_obs = c, which

        elapsed = time.time() - self.t_start

        img = cv2.resize(self.frame, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)[:, :, ::-1]
        img_t = torch.as_tensor(img.copy(), dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0) / 255.0

        ox, oy, oyaw = self.odom
        ox0, oy0, _ = self.odom_origin
        odom_obs = np.asarray(
            [ox - ox0, oy - oy0, math.sin(oyaw), math.cos(oyaw)], np.float32)
        odom_n = (odom_obs - self.odom_mean) / self.odom_std
        odom_t = torch.as_tensor(odom_n).unsqueeze(0)

        with torch.no_grad():
            out = self.model(img_t, odom_t).squeeze(0).numpy()
        a_n = out[0] if self.chunk > 1 else out

        pv, pw = a_n * self.act_std + self.act_mean
        pv = float(np.clip(pv, -MAX_LIN, MAX_LIN))
        pw = float(np.clip(pw, -MAX_ANG, MAX_ANG))

        # Optional takeover hook (used by dagger_room.py): a human can
        # replace the policy's command for this step. Default None = pure
        # policy, so nothing changes for the other test scripts.
        v, w, human = pv, pw, False
        if self.override is not None:
            v, w, human = self.override(pv, pw)
        self.publish(v, w)

        if self.sample_hook is not None:
            self.sample_hook(dict(
                x=x, y=y, yaw=yaw, odom_x=ox, odom_y=oy, odom_yaw=oyaw,
                frame=self.frame, v=v, w=w, human=human))

        # per-step log for offline analysis (bc diagnosis): where it was,
        # what odometry/camera-derived obs it had, what it commanded
        self.step_log.append(dict(
            trial=self.cur_trial['idx'], label=self.cur_trial['label'],
            start_x=self.cur_trial['start_x'],
            start_y=self.cur_trial['start_y'],
            start_yaw=self.cur_trial['start_yaw'],
            x=x, y=y, yaw=yaw,
            odom_dx=ox - ox0, odom_dy=oy - oy0, odom_yaw=oyaw,
            pred_v=pv, pred_w=pw, clearance=c,
            v=v, w=w, human=human,
        ))

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

    def save_steps(self):
        """Write every control step, joined with its trial's outcome, for
        offline diagnosis (see analyse_rollouts_room.py)."""
        if not self.args.steps_csv or not self.step_log:
            return
        os.makedirs(os.path.dirname(self.args.steps_csv) or '.', exist_ok=True)
        fields = list(self.step_log[0].keys()) + [
            'outcome', 'success', 'collided', 'min_clear', 'min_clear_obs']
        with open(self.args.steps_csv, 'w', newline='') as f:
            import csv
            wri = csv.DictWriter(f, fieldnames=fields)
            wri.writeheader()
            for row in self.step_log:
                out = self.trial_outcomes.get(row['trial'], {})
                wri.writerow({**row, **{k: out.get(k, '') for k in fields
                                        if k not in row}})
        print(f"wrote {len(self.step_log)} steps -> {self.args.steps_csv}")

    def report(self):
        self.save_steps()
        r = self.results
        if not r:
            return
        n = len(r)
        ok = sum(x['success'] for x in r)
        hit = sum(1 for x in r if x['min_clear'] < 0.0)

        print("\n" + "=" * 58)
        print(f" camera+odometry BC policy: success {ok}/{n}  "
              f"({100.0 * ok / n:.0f}%)")
        print(f" collided with something: {hit}/{n}  "
              f"({100.0 * hit / n:.0f}%)")

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

        if hit:
            by_obs = {}
            for x in r:
                if x['min_clear'] < 0.0:
                    by_obs[x['min_clear_obs']] = by_obs.get(x['min_clear_obs'], 0) + 1
            print("\n what it hit: " + ", ".join(
                f"{k} {v}" for k, v in sorted(by_obs.items())))

        clears = np.array([x['min_clear'] for x in r])
        print(f"\n closest approach to ANYTHING (island, gap boxes, decor, "
              f"walls)\n   mean {clears.mean():+.3f} m   worst "
              f"{clears.min():+.3f} m")
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
    p.add_argument('--policy', default='policy_room.pt')
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='test')
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=40.0)
    p.add_argument('--settle', type=float, default=1.5)
    p.add_argument('--robot', default='burger')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--image-topic', default='/camera/image_raw')
    p.add_argument('--csv', default='')
    p.add_argument('--steps-csv', default='',
                   help='write every control step here for offline analysis')
    p.add_argument('--room-margin', type=float, default=0.3)
    p.add_argument('--jitter-xy', type=float, default=0.08,
                   help='m - uniform +/- random offset applied to x and y '
                        'each trial, so it never starts from the exact '
                        'recorded/config pose (default 0.08)')
    p.add_argument('--jitter-yaw-deg', type=float, default=8.0,
                   help='deg - uniform +/- random heading offset each '
                        'trial (default 8)')
    p.add_argument('--jitter-seed', type=int, default=None,
                   help='fix this for a reproducible jitter sequence '
                        '(default: random each run)')
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    poses = load_poses(args.config, args.starts)
    obstacles = cfg['obstacles']
    goal = cfg['goal']
    room = cfg['room']
    args.room_x_min, args.room_x_max = room['x_min'], room['x_max']
    args.room_y_min, args.room_y_max = room['y_min'], room['y_max']

    model, ck = load_policy(args.policy)

    rclpy.init()
    node = PolicyRunner(args, model, ck, poses, obstacles, goal)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        try:
            node.publish(0.0, 0.0)
            node.save_steps()
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
