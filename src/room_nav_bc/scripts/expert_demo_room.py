#!/usr/bin/env python3
"""
Generate BALANCED expert demonstrations for the room-navigation task - no
human teleop, fully autonomous. This is the privileged scripted expert
discussed for DAgger, reused here for its other job: producing a clean,
exactly-balanced left/right dataset instead of hoping human driving
happens to split evenly.

    ros2 run room_nav_bc expert_demo_room.py --episodes-per-combo 5

For every (start pose x side) combination - each of the --starts set's
poses, crossed with BOTH going left and right of the island - it runs
--episodes-per-combo episodes, teleporting the robot, driving it with a
deterministic waypoint-following controller (ground truth pose only -
this is the privileged expert, not what the learned policy will see),
and logging camera + odometry + action exactly like record_demo_room.py
so the two are interchangeable to the dataset builder.

THE CONTROLLER.  A handful of waypoints per episode: offset past the
island on the chosen side, through the (always centred) tight gap, then
the goal, tracked with lookahead-based path following (see PathFollower).
Fully deterministic - no action noise - since the only thing that matters
for an expert used in BC/DAgger is that it's COMPETENT and DETERMINISTIC
given (state, chosen side), not that it's optimal.

Fully deterministic also means repeats of the same (start, side) combo
are bit-for-bit identical - there's no within-mode variation to gain from
--episodes-per-combo > 1 any more. Use --episodes-per-combo 1 (8 episodes
total for the default 4-start set) unless you add some other source of
variety (e.g. --starts a set with more poses).

Only episodes that actually reach the goal are saved, so what you get is
a clean dataset, exactly N left + N right per start pose.
"""

import argparse
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record_demo_room import (  # noqa: E402
    MAX_ANG, MAX_LIN, default_config_path, encode_frame, in_goal, load_goal,
    load_obstacles, load_poses, load_room, next_index, quaternion_from_yaw,
    wrap_angle, yaw_from_quaternion,
)

import pickle


def build_waypoints(side, obstacles, goal, clearance=0.25,
                    transition_margin=0.05):
    """[(x,y), ...]: past the island on `side`, through the gap, the goal.

    FIVE waypoints, not three - a straight line from "abeam the island"
    directly to "gap centre" cuts the island's far corner and the gap
    box's near corner, both with less clearance than the robot's own
    radius (verified with verify_expert_room.py - don't go back to 3).
    The fix is to stay offset until fully PAST the island's y-extent,
    then come back to the centreline before the gap's y-extent begins, so
    every straight segment's nearest obstacle is entirely to one side of
    it, not cutting a corner:

        1) bypass_x, just before the island starts (y)
        2) bypass_x, just past the island's far edge (y)
        3) centreline (x=0), just before the gap boxes start (y)
        4) gap centre
        5) goal

    Reads island/gap/goal positions straight from start_goal.yaml rather
    than hardcoding them again, so this stays correct if the room layout
    is ever regenerated with different numbers.
    """
    island = obstacles['island']
    gap_l, gap_r = obstacles['gap_left'], obstacles['gap_right']
    bypass_x = (island['w'] / 2.0 + clearance) * (1.0 if side == 'right'
                                                   else -1.0)
    gap_x = (gap_l['x'] + gap_r['x']) / 2.0
    gap_y = (gap_l['y'] + gap_r['y']) / 2.0

    island_near_y = island['y'] - island['h'] / 2.0 - transition_margin
    island_far_y = island['y'] + island['h'] / 2.0 + transition_margin
    gap_near_y = gap_y - gap_r['h'] / 2.0 - transition_margin
    # Midpoint of the island->gap transition: a pure-pursuit controller
    # beelining straight from "offset, just past the island" to
    # "centreline, just before the gap" cuts the island's far corner (its
    # forward speed outruns how fast it can turn before the lateral move
    # is done - verified empirically, see verify_expert_room.py). Splitting
    # that transition across an extra waypoint halves the heading change
    # demanded per segment.
    mid_y = (island_far_y + gap_near_y) / 2.0
    mid_x = bypass_x * 0.5
    # Drive straight up the centreline until well before the island, and
    # only start moving toward the chosen side after that. Until this
    # point the left and right routes give IDENTICAL observations (same
    # camera view, odometry offset ~0), so a regression policy trained on
    # both must average them there. Without this the lateral move starts
    # at the start pose and the sides are separable almost immediately.
    lateral_start_y = island_near_y - 0.5

    return [
        (0.0, lateral_start_y),
        (bypass_x, island_near_y),
        (bypass_x, island_far_y),
        (mid_x, mid_y),
        (0.0, gap_near_y),
        (gap_x, gap_y),
        (goal['x'], goal['y']),
    ]


def steer_to_point(x, y, yaw, target, v_cruise, k_heading):
    """One proportional step toward a point. -> (v, w)."""
    dx, dy = target[0] - x, target[1] - y
    heading_err = wrap_angle(math.atan2(dy, dx) - yaw)
    w = max(-MAX_ANG, min(MAX_ANG, k_heading * heading_err))
    # scale forward speed down (to zero, never negative) as the heading
    # error grows - turn toward the target before driving at it
    v = v_cruise * max(0.0, math.cos(heading_err))
    v = max(0.0, min(MAX_LIN, v))
    return v, w


def _closest_point_on_segment(x, y, p0, p1):
    """-> (proj_x, proj_y, t in [0,1], segment_length)."""
    ax, ay = p0
    bx, by = p1
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    if seg_len2 < 1e-12:
        return ax, ay, 0.0, 0.0
    t = ((x - ax) * abx + (y - ay) * aby) / seg_len2
    t = max(0.0, min(1.0, t))
    return ax + t * abx, ay + t * aby, t, math.sqrt(seg_len2)


class PathFollower:
    """
    Lookahead-based tracking along a polyline - classic pure-pursuit ON A
    PATH (used in real autonomous vehicle path tracking), not point-to-
    point waypoint chasing.

    Point-to-point chasing (aim at waypoint N, switch to N+1 once within
    some tolerance) does NOT bound how far the trajectory can stray from
    the planned line, especially right after a switch or through a sharp
    turn - verified empirically with verify_expert_room.py to cut corners
    by several centimetres, enough to clip the obstacles this path is
    specifically trying to avoid. Parametrising the whole polyline by arc
    length and always steering at a fixed small distance AHEAD of the
    robot's own closest point on the path keeps the trajectory hugging the
    line through corners instead, with tracking error bounded by
    (roughly) the lookahead distance rather than left to chance.
    """

    def __init__(self, waypoints, lookahead=0.15):
        self.wps = waypoints
        self.lookahead = lookahead
        self.cum = [0.0]
        for i in range(1, len(waypoints)):
            self.cum.append(self.cum[-1] + math.hypot(
                waypoints[i][0] - waypoints[i - 1][0],
                waypoints[i][1] - waypoints[i - 1][1]))

    def _closest_arclength(self, x, y):
        best_s, best_d = 0.0, float('inf')
        for i in range(len(self.wps) - 1):
            px, py, t, seg_len = _closest_point_on_segment(
                x, y, self.wps[i], self.wps[i + 1])
            d = math.hypot(x - px, y - py)
            if d < best_d:
                best_d, best_s = d, self.cum[i] + t * seg_len
        return best_s

    def _point_at_arclength(self, s):
        s = max(0.0, min(self.cum[-1], s))
        for i in range(len(self.wps) - 1):
            if self.cum[i] <= s <= self.cum[i + 1]:
                seg_len = self.cum[i + 1] - self.cum[i]
                t = 0.0 if seg_len < 1e-9 else (s - self.cum[i]) / seg_len
                return (self.wps[i][0] + t * (self.wps[i + 1][0] - self.wps[i][0]),
                        self.wps[i][1] + t * (self.wps[i + 1][1] - self.wps[i][1]))
        return self.wps[-1]

    def target_and_progress(self, x, y):
        """-> (target_x, target_y, distance_remaining_to_the_very_end)."""
        s = self._closest_arclength(x, y)
        target = self._point_at_arclength(min(s + self.lookahead, self.cum[-1]))
        remaining = self.cum[-1] - s
        return target[0], target[1], remaining


class ExpertRunner(Node):

    def __init__(self, args):
        super().__init__('expert_demo_room')
        self.args = args
        self.goal = load_goal(args.config)
        self.obstacles = load_obstacles(args.config)
        self.room = load_room(args.config)
        self.bridge = CvBridge()

        starts = load_poses(args.config, args.starts)
        self.plan = [
            (label, x, y, yaw, side)
            for (label, x, y, yaw) in starts
            for side in ('left', 'right')
            for _ in range(args.episodes_per_combo)
        ]
        self.plan_i = 0

        self.pose = None       # ground truth - the ONLY thing the expert uses
        self.odom = None       # logged for the dataset, not used by the expert
        self.frame = None

        self.follower = None
        self.samples = []
        self.t_start = None
        self.running = False
        self.saved, self.failed = 0, 0

        os.makedirs(args.data_dir, exist_ok=True)

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

        print(f"\nplan: {len(starts)} starts x 2 sides x "
              f"{args.episodes_per_combo} reps = {len(self.plan)} episodes")
        print(f"data -> {args.data_dir}/\n")
        self.begin_episode()

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

    # -- episodes --------------------------------------------------------

    def begin_episode(self):
        if self.plan_i >= len(self.plan):
            self.report()
            raise KeyboardInterrupt

        label, x, y, yaw, side = self.plan[self.plan_i]
        self.publish(0.0, 0.0)
        self.teleport(x, y, yaw)
        # The teleport service call is fire-and-forget, and /gazebo/
        # model_states keeps publishing in the background - without this,
        # step() can run on a pose still left over from the END of the
        # PREVIOUS episode (which was AT the goal) before a fresh
        # post-teleport reading arrives, making the new episode look
        # "already arrived" after 0-1 samples. Forces step() to wait (it
        # already skips while pose/odom/frame are None) for a confirmed
        # fresh reading instead of trusting a stale one.
        self.pose = None
        self.odom = None
        self.frame = None

        # The start pose itself must be the path's first point, not just
        # an unconstrained "chase wp0" phase before the defined polyline
        # begins - verified empirically that omitting it lets the initial
        # approach cut the island's corner right where it joins wp0.
        waypoints = [(x, y)] + build_waypoints(
            side, self.obstacles, self.goal,
            clearance=self.args.clearance,
            transition_margin=self.args.transition_margin)
        self.follower = PathFollower(waypoints, lookahead=self.args.lookahead)
        self.samples = []
        self.t_start = time.time()
        self.running = True
        print(f"  [{self.plan_i + 1:3d}/{len(self.plan)}] start {label} "
              f"side={side:5s} ... ", end='', flush=True)

    def finish_episode(self, success):
        self.publish(0.0, 0.0)
        self.running = False
        label, x, y, yaw, side = self.plan[self.plan_i]

        if success and len(self.samples) >= self.args.min_samples:
            out = os.path.join(
                self.args.data_dir,
                f"start{label}_{side}_{next_index(self.args.data_dir, label + '_' + side):03d}.p")
            last = self.samples[-1]
            episode = dict(
                samples=self.samples,
                meta=dict(
                    sensors=['camera', 'odom'],
                    source='scripted_expert',
                    side=side,
                    start_label=label,
                    start_pose_nominal=(x, y, yaw),
                    start_pose=(self.samples[0]['x'], self.samples[0]['y'],
                               self.samples[0]['yaw']),
                    final_pose=(last['x'], last['y'], last['yaw']),
                    reached_goal=True,
                    goal=self.goal,
                    rate=self.args.rate,
                    robot=self.args.robot,
                    duration_s=last['t'],
                    n_samples=len(self.samples),
                    recorded_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                ),
            )
            with open(out, 'wb') as f:
                pickle.dump(episode, f)
            self.saved += 1
            print(f"OK  {len(self.samples)} samples, {last['t']:.1f}s "
                  f"-> {os.path.basename(out)}")
        else:
            self.failed += 1
            print(f"FAILED ({'timeout/out-of-bounds' if not success else 'too short'})")

        self.plan_i += 1
        time.sleep(0.2)
        self.begin_episode()

    # -- control loop ------------------------------------------------------

    def step(self):
        if not self.running or self.pose is None or self.odom is None \
                or self.frame is None:
            return

        x, y, yaw = self.pose
        elapsed = time.time() - self.t_start

        # safety: out of the room entirely -> something went wrong, abort
        r = self.room
        if not (r['x_min'] - 0.3 <= x <= r['x_max'] + 0.3
                and r['y_min'] - 0.3 <= y <= r['y_max'] + 0.3):
            self.finish_episode(success=False)
            return
        if elapsed > self.args.timeout:
            self.finish_episode(success=False)
            return

        tx, ty, remaining = self.follower.target_and_progress(x, y)
        if remaining < self.args.goal_tol:
            self.finish_episode(success=in_goal(x, y, self.goal))
            return
        v, w = steer_to_point(x, y, yaw, (tx, ty), self.args.cruise_speed,
                              self.args.k_heading)
        v = max(-MAX_LIN, min(MAX_LIN, v))
        w = max(-MAX_ANG, min(MAX_ANG, w))
        self.publish(v, w)

        ox, oy, oyaw = self.odom
        img_jpeg = encode_frame(self.frame)
        self.samples.append(dict(
            t=elapsed,
            x=x, y=y, yaw=yaw,
            odom_x=ox, odom_y=oy, odom_yaw=oyaw,
            img_jpeg=img_jpeg,
            v=v, w=w,
        ))

    def report(self):
        print(f"\n{'=' * 58}")
        print(f" saved {self.saved}  failed {self.failed}  "
              f"(of {len(self.plan)} planned)")
        print(f"{'=' * 58}\n")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_room_expert')
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='train')
    p.add_argument('--episodes-per-combo', type=int, default=5,
                   help='episodes per (start x side) - default 5')
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=40.0)
    p.add_argument('--min-samples', type=int, default=10)
    p.add_argument('--cruise-speed', type=float, default=0.15)
    p.add_argument('--k-heading', type=float, default=2.5)
    p.add_argument('--lookahead', type=float, default=0.15,
                   help='m - how far ahead along the path to steer '
                        '(default 0.15) - see PathFollower')
    p.add_argument('--goal-tol', type=float, default=0.15,
                   help='m - remaining path distance counted as "arrived"')
    p.add_argument('--clearance', type=float, default=0.25,
                   help='m - how far past the island edge to aim when '
                        'bypassing it (default 0.25)')
    p.add_argument('--transition-margin', type=float, default=0.05,
                   help='m - how far past/before an obstacle edge to stay '
                        'offset before cutting back to centreline '
                        '(default 0.05) - see build_waypoints()')
    p.add_argument('--robot', default='burger')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--image-topic', default='/camera/image_raw')
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    rclpy.init()
    node = ExpertRunner(args)
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
