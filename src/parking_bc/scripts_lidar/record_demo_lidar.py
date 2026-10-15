#!/usr/bin/env python3
"""
Drive the robot and record parking demonstrations - odometry+lidar variant.

Identical teleop and controls to scripts/record_demo.py (this IS that
node, plus two extra subscriptions). The ground-truth pose is still
logged too, exactly as before - it costs nothing and is invaluable for
debugging the lidar gap-detector later - but it is never fed to the
policy trained on this data. Only odometry and the raw lidar scan are.

    # terminal 1 - simulator (same launch file, unchanged)
    ros2 launch parking_bc parking_demo.launch.py

    # terminal 2 - this node
    ros2 run parking_bc record_demo_lidar.py --data-dir data_lidar

Controls, same as before:

           w                 w / x   forward / backward
      a    s    d            a / d   turn left / right
           x                 s       stop

    ENTER   save the episode, advance to the next start pose
    SPACE   discard the episode, retry the same start pose
    r       reset the robot to the current start pose
    n / p   next / previous start pose
    l       list start poses and episode counts
    q       quit

Writes to --data-dir (default "data_lidar", NOT "data" - the original
ground-truth dataset in data/ is never touched by this script).
"""

import argparse
import glob
import math
import os
import pickle
import re
import select
import sys
import termios
import time
import tty

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


# -- TurtleBot3 Burger limits and step sizes (same as turtlebot3_teleop) ----
MAX_LIN = 0.22
MAX_ANG = 2.84
LIN_STEP = 0.01
ANG_STEP = 0.1


def default_config_path():
    """../config/start_poses.yaml - the SAME file scripts/record_demo.py
    reads, so both variants always drive from identical start poses."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, '..', 'config', 'start_poses.yaml'),
        os.path.join(here, '..', 'share', 'parking_bc',
                     'config', 'start_poses.yaml'),
    ):
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('parking_bc'),
                            'config', 'start_poses.yaml')
    except Exception:
        return os.path.normpath(
            os.path.join(here, '..', 'config', 'start_poses.yaml'))


def load_poses(path, set_name):
    """-> [(label, x, y, yaw_radians), ...]"""
    if not os.path.exists(path):
        raise SystemExit(f"start-pose config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    sets = cfg.get('sets', {})
    if set_name not in sets:
        raise SystemExit(f"set '{set_name}' not in {path}. "
                         f"available: {', '.join(sorted(sets))}")
    return [(str(e['label']), float(e['x']), float(e['y']),
             math.radians(float(e['yaw']))) for e in sets[set_name]]


# --------------------------------------------------------------------------
# Pure helpers (no ROS - unit tested)
# --------------------------------------------------------------------------

def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def approach(current, target, step):
    """Move `current` toward `target` by at most `step` (teleop smoothing)."""
    if current < target:
        return min(target, current + step)
    if current > target:
        return max(target, current - step)
    return target


def clamp_turn_rate(v, w, r_min, v_eps=1e-3):
    """Impose a minimum turning radius: |w| <= |v| / r_min."""
    if r_min <= 0.0:
        return w, False
    if abs(v) < v_eps:
        return 0.0, abs(w) > v_eps
    w_max = abs(v) / r_min
    if abs(w) > w_max:
        return math.copysign(w_max, w), True
    return w, False


def to_bay_frame(x, y, yaw, bay_x=0.0, bay_y=0.0):
    return x - bay_x, y - bay_y, wrap_angle(yaw)


def next_index(data_dir, label):
    used = set()
    for path in glob.glob(os.path.join(data_dir, f"start{label}_*.p")):
        m = re.search(rf"start{label}_(\d+)\.p$", os.path.basename(path))
        if m:
            used.add(int(m.group(1)))
    i = 1
    while i in used:
        i += 1
    return i


def count_episodes(data_dir, label):
    return len(glob.glob(os.path.join(data_dir, f"start{label}_*.p")))


# --------------------------------------------------------------------------

class KeyReader:
    """Non-blocking single-key reads, restoring the terminal on exit."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None

    def __enter__(self):
        try:
            self.saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        except termios.error:
            self.saved = None
        return self

    def __exit__(self, *exc):
        self.restore()

    def restore(self):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None

    def get(self):
        if self.saved is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# --------------------------------------------------------------------------

class DemoSessionLidar(Node):

    def __init__(self, args, keys):
        super().__init__('record_demo_lidar')
        self.args = args
        self.keys = keys
        self.starts = load_poses(args.config, args.starts)
        self.idx = 0

        # teleop state
        self.target_lin = 0.0
        self.target_ang = 0.0
        self.lin = 0.0
        self.ang = 0.0
        self.clamped = False

        self.pose = None          # ground truth (Gazebo), logged, not sensed
        self.odom = None          # (x, y, yaw) from /odom - what the robot senses
        self.scan = None          # last LaserScan message
        self.samples = []
        self.recording = False
        self.t0 = None
        self.saved_count = 0
        self.discarded_count = 0

        os.makedirs(args.data_dir, exist_ok=True)

        qos_best_effort = QoSProfile(depth=10)
        qos_best_effort.reliability = ReliabilityPolicy.BEST_EFFORT

        self.create_subscription(
            ModelStates, '/gazebo/model_states', self.on_states,
            qos_best_effort)
        self.create_subscription(
            Odometry, args.odom_topic, self.on_odom, 10)
        self.create_subscription(
            LaserScan, args.scan_topic, self.on_scan, qos_best_effort)
        self.pub = self.create_publisher(Twist, args.cmd_out, 10)

        if args.external_teleop:
            self.create_subscription(
                Twist, args.cmd_in, self.on_external_cmd, 10)
        self.reset_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(0.05, self.on_drive)        # 20 Hz control
        self.create_timer(1.0 / args.rate, self.on_log)
        self.create_timer(0.02, self.on_keys)

        self.banner()
        self.reset_robot()

    # -- output ------------------------------------------------------------

    def say(self, msg=""):
        sys.stdout.write(msg + "\r\n")
        sys.stdout.flush()

    def banner(self):
        a = self.args
        clamp_txt = (f"R_min = {a.r_min:.2f} m"
                     if a.r_min > 0 else "no clamp (plain teleop)")
        self.say()
        self.say("=" * 64)
        self.say("  ODOMETRY + LIDAR variant - logs /odom and "
                 f"{a.scan_topic}")
        self.say("  (ground truth pose is still logged too, for "
                 "debugging only)")
        if a.external_teleop:
            self.say(f"  driving: external teleop on {a.cmd_in}")
        else:
            self.say("        w             w/x  forward / backward")
            self.say("   a    s    d        a/d  turn left / right")
            self.say("        x             s    stop")
        self.say("")
        self.say("   ENTER save    SPACE discard    r reset")
        self.say("   n/p next/prev start    l list    q quit")
        self.say("-" * 64)
        self.say(f" logging {a.rate:.0f} Hz | {clamp_txt} | "
                 f"data -> {a.data_dir}/")
        self.say(f" starts: set '{a.starts}' from "
                 f"{os.path.basename(a.config)} "
                 f"({len(self.starts)} poses)  -- SAME file/positions as "
                 "the original recorder")
        self.say("=" * 64)

    def status(self):
        label, x, y, yaw = self.starts[self.idx]
        n = count_episodes(self.args.data_dir, label)
        ready = "" if (self.odom is not None and self.scan is not None) \
            else "  ** waiting for /odom and lidar... **"
        self.say(f"[start {label}] x={x:+.2f} y={y:+.2f} "
                 f"yaw={math.degrees(yaw):+.0f} deg  "
                 f"({n} recorded)  -- drive when ready{ready}")

    # -- robot -------------------------------------------------------------

    def publish(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

    def stop(self):
        self.target_lin = self.target_ang = 0.0
        self.lin = self.ang = 0.0
        self.publish(0.0, 0.0)

    def reset_robot(self):
        label, x, y, yaw = self.starts[self.idx]
        self.stop()

        if not self.reset_cli.service_is_ready():
            self.reset_cli.wait_for_service(timeout_sec=2.0)
        if not self.reset_cli.service_is_ready():
            self.say("!! /gazebo/set_entity_state unavailable - "
                     "reposition the robot yourself")
        else:
            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            req = SetEntityState.Request()
            req.state.name = self.args.robot
            req.state.pose.position.x = float(x)
            req.state.pose.position.y = float(y)
            req.state.pose.position.z = 0.01
            req.state.pose.orientation.x = qx
            req.state.pose.orientation.y = qy
            req.state.pose.orientation.z = qz
            req.state.pose.orientation.w = qw
            self.reset_cli.call_async(req)

        self.samples = []
        self.recording = False
        self.status()

    # -- callbacks ---------------------------------------------------------

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

    def on_external_cmd(self, msg):
        self.target_lin = self.lin = msg.linear.x
        self.target_ang = self.ang = msg.angular.z

    def on_drive(self):
        if self.args.external_teleop:
            w_out, self.clamped = clamp_turn_rate(
                self.lin, self.ang, self.args.r_min)
            self.publish(self.lin, w_out)
            self.w_applied = w_out
            return
        self.lin = approach(self.lin, self.target_lin, LIN_STEP / 2.0)
        self.ang = approach(self.ang, self.target_ang, ANG_STEP / 2.0)
        w_out, self.clamped = clamp_turn_rate(
            self.lin, self.ang, self.args.r_min)
        self.publish(self.lin, w_out)
        self.w_applied = w_out

    def on_log(self):
        # Need all three: ground truth (for the logged debug fields),
        # odometry and a scan (what the new dataset actually trains on).
        if self.pose is None or self.odom is None or self.scan is None:
            return
        w_applied = getattr(self, 'w_applied', self.ang)
        moving = abs(self.lin) > 1e-3 or abs(w_applied) > 1e-3

        if not self.recording:
            if not moving:
                return
            self.recording = True
            self.t0 = time.time()
            self.samples = []
            self.say("  recording...")

        x, y, yaw = self.pose
        bx, by, byaw = to_bay_frame(x, y, yaw,
                                    self.args.bay_x, self.args.bay_y)
        ox, oy, oyaw = self.odom
        s = self.scan
        self.samples.append(dict(
            t=time.time() - self.t0,
            # ground truth - logged for debugging/plotting only
            x=x, y=y, yaw=yaw,
            bay_x=bx, bay_y=by, bay_yaw=byaw,
            # what the lidar+odom policy actually gets to see
            odom_x=ox, odom_y=oy, odom_yaw=oyaw,
            scan_ranges=list(s.ranges),
            scan_angle_min=s.angle_min,
            scan_angle_increment=s.angle_increment,
            scan_range_max=s.range_max,
            # action
            v=self.lin, w=w_applied,
            w_raw=self.ang, clamped=bool(self.clamped),
        ))

        if len(self.samples) % 3 == 0:
            sys.stdout.write(
                f"\r  {self.samples[-1]['t']:5.1f}s  "
                f"v {self.lin:+.2f}  w {w_applied:+.2f}   "
                f"odom ({ox:+.3f},{oy:+.3f})  "
                f"dx {bx:+.3f}  dy {by:+.3f}      ")
            sys.stdout.flush()

    # -- keys --------------------------------------------------------------

    def on_keys(self):
        k = self.keys.get()
        if k is None:
            return

        if k in ('w', 'x', 'a', 'd', 's') and self.args.external_teleop:
            if k == 's':
                self.stop()
            return
        if k == 'w':
            self.target_lin = clamp(self.target_lin + LIN_STEP,
                                    -MAX_LIN, MAX_LIN)
        elif k == 'x':
            self.target_lin = clamp(self.target_lin - LIN_STEP,
                                    -MAX_LIN, MAX_LIN)
        elif k == 'a':
            self.target_ang = clamp(self.target_ang + ANG_STEP,
                                    -MAX_ANG, MAX_ANG)
        elif k == 'd':
            self.target_ang = clamp(self.target_ang - ANG_STEP,
                                    -MAX_ANG, MAX_ANG)
        elif k == 's':
            self.stop()

        elif k in ('\r', '\n'):
            self.save_episode()
        elif k == ' ':
            self.discard_episode()
        elif k == 'r':
            self.say()
            self.say("  reset")
            self.reset_robot()
        elif k in ('n', 'p'):
            if self.samples:
                self.say()
                self.say("  episode in progress - ENTER to save, "
                         "SPACE to discard")
                return
            self.idx = ((self.idx + 1) if k == 'n' else (self.idx - 1)) \
                % len(self.starts)
            self.say()
            self.reset_robot()
        elif k == 'l':
            self.list_starts()
        elif k in ('q', '\x03'):
            self.stop()
            self.say()
            self.summary()
            raise KeyboardInterrupt

    def list_starts(self):
        self.say()
        for i, (label, x, y, yaw) in enumerate(self.starts):
            n = count_episodes(self.args.data_dir, label)
            mark = ">" if i == self.idx else " "
            self.say(f" {mark} {label}  x={x:+.2f} y={y:+.2f} "
                     f"yaw={math.degrees(yaw):+.0f}  {n} episodes")
        self.say()

    # -- episodes ----------------------------------------------------------

    def discard_episode(self):
        self.say()
        if not self.samples:
            self.say("  nothing to discard")
        else:
            self.say(f"  discarded ({len(self.samples)} samples)")
            self.discarded_count += 1
        self.reset_robot()

    def save_episode(self):
        self.say()
        if not self.samples:
            self.say("  nothing recorded yet")
            return

        label, sx, sy, syaw = self.starts[self.idx]
        i = next_index(self.args.data_dir, label)
        out = os.path.join(self.args.data_dir, f"start{label}_{i:03d}.p")

        first, last = self.samples[0], self.samples[-1]
        episode = dict(
            samples=self.samples,
            meta=dict(
                sensors=['odom', 'scan'],
                start_label=label,
                start_pose_nominal=(sx, sy, syaw),
                start_pose=(first['x'], first['y'], first['yaw']),
                final_pose=(last['x'], last['y'], last['yaw']),
                r_min=self.args.r_min,
                rate=self.args.rate,
                robot=self.args.robot,
                bay_centre=(self.args.bay_x, self.args.bay_y),
                preset=self.args.preset,
                duration_s=last['t'],
                n_samples=len(self.samples),
                n_clamped=sum(s['clamped'] for s in self.samples),
                recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'),
            ),
        )
        with open(out, 'wb') as f:
            pickle.dump(episode, f)

        self.saved_count += 1
        self.say(f"  saved {os.path.basename(out)}  "
                 f"{len(self.samples)} samples, {last['t']:.1f}s  "
                 f"final dx={last['bay_x']:+.3f} dy={last['bay_y']:+.3f}")
        if last['t'] > 60.0:
            self.say("        NOTE: long episode - shorter, more decisive "
                     "runs make better training data")

        self.idx = (self.idx + 1) % len(self.starts)
        self.reset_robot()

    def summary(self):
        self.say("=" * 64)
        self.say(f" saved {self.saved_count}, discarded "
                 f"{self.discarded_count} this session")
        for label, _, _, _ in self.starts:
            self.say(f"   start {label}: "
                     f"{count_episodes(self.args.data_dir, label)} on disk")
        self.say("=" * 64)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_lidar',
                   help='separate from the original data/ (default: '
                        'data_lidar)')
    p.add_argument('--config', default=None,
                   help='path to start_poses.yaml (default: the package '
                        'one - the SAME file the original recorder uses)')
    p.add_argument('--starts', default='right',
                   help="which set in the YAML (default: right)")
    p.add_argument('--r-min', type=float, default=0.0,
                   help='minimum turning radius in m; 0 = no clamp (default)')
    p.add_argument('--rate', type=float, default=10.0,
                   help='logging rate in Hz (default: 10)')
    p.add_argument('--robot', default='burger')
    p.add_argument('--bay-x', type=float, default=0.0,
                   help='ground truth, logged for debugging only - never '
                        'fed to the policy')
    p.add_argument('--bay-y', type=float, default=0.16)
    p.add_argument('--preset', default='default')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--cmd-in', default='/cmd_vel_raw')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--scan-topic', default='/scan')
    p.add_argument('--external-teleop', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    rclpy.init()
    with KeyReader() as keys:
        node = DemoSessionLidar(args, keys)
        try:
            rclpy.spin(node)
        except (KeyboardInterrupt, RuntimeError):
            pass
        finally:
            keys.restore()
            for cleanup in (lambda: node.publish(0.0, 0.0),
                            node.destroy_node):
                try:
                    cleanup()
                except Exception:
                    pass
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    main()
