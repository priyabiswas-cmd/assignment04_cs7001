#!/usr/bin/env python3
"""
Offline safety verification for the scripted expert - NO Gazebo needed.

Simulates the exact same kinematics and lookahead path-following as
expert_demo_room.py's step(), for every (start x side) combo, and checks
the robot's ACTUAL FOOTPRINT (not just its centre point) against every
obstacle at every timestep. This is the check that was missing before:
reaching the goal eventually is not the same as never touching an
obstacle along the way.

The controller is fully deterministic (no action noise), so one run per
combo is exhaustive - there's nothing a second run with a different seed
would show you that the first didn't.

    python3 verify_expert_room.py            # all starts x both sides
    python3 verify_expert_room.py -v         # + per-step trace

Exits non-zero if any run collides or fails to reach the goal, so this can
also be dropped into a CI-style check after any future change to the
controller or the room layout.
"""

import argparse
import math
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# expert_demo_room.py only imports rclpy/cv_bridge/etc. at module scope for
# the live ROS node - stub them out so this script runs with plain python3,
# no ROS environment needed, which is the whole point of an offline check.
import types
for _mod in ('rclpy', 'rclpy.node', 'rclpy.qos', 'cv_bridge',
            'gazebo_msgs', 'gazebo_msgs.msg', 'gazebo_msgs.srv',
            'geometry_msgs', 'geometry_msgs.msg',
            'nav_msgs', 'nav_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg'):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
sys.modules['rclpy.node'].Node = object
sys.modules['rclpy.qos'].QoSProfile = object
sys.modules['rclpy.qos'].ReliabilityPolicy = object
sys.modules['cv_bridge'].CvBridge = object
sys.modules['gazebo_msgs.msg'].ModelStates = object
sys.modules['gazebo_msgs.srv'].SetEntityState = object
sys.modules['geometry_msgs.msg'].Twist = object
sys.modules['nav_msgs.msg'].Odometry = object
sys.modules['sensor_msgs.msg'].Image = object

import expert_demo_room as expert  # noqa: E402

# TurtleBot3 Burger footprint 0.138 x 0.178 m -> circumscribed radius.
ROBOT_RADIUS = math.hypot(0.178 / 2.0, 0.138 / 2.0)   # ~0.113 m
SAFETY_MARGIN = 0.05                                   # extra buffer


def point_box_clearance(px, py, box):
    """Distance from (px,py) to the nearest edge of an axis-aligned box
    (box: dict with x,y,w,h). 0 or negative-equivalent means inside."""
    dx = max(abs(px - box['x']) - box['w'] / 2.0, 0.0)
    dy = max(abs(py - box['y']) - box['h'] / 2.0, 0.0)
    return math.hypot(dx, dy)


def simulate(x0, y0, yaw0, side, obstacles, goal, args):
    """Faithful re-implementation of ExpertRunner.step()'s control logic,
    minus the ROS plumbing, returning (reached, min_clearance, worst_obstacle,
    trajectory)."""
    # start pose must be the path's own first point - see the matching
    # comment in ExpertRunner.begin_episode()
    wps = [(x0, y0)] + expert.build_waypoints(
        side, obstacles, goal, clearance=args.clearance,
        transition_margin=args.transition_margin)
    follower = expert.PathFollower(wps, lookahead=args.lookahead)
    x, y, yaw = x0, y0, yaw0
    dt = 1.0 / args.rate
    traj = [(x, y)]
    min_clear, worst_obs = float('inf'), None

    def check(px, py):
        nonlocal min_clear, worst_obs
        for name, box in obstacles.items():
            c = point_box_clearance(px, py, box) - ROBOT_RADIUS
            if c < min_clear:
                min_clear, worst_obs = c, name

    check(x, y)
    n_steps = int(args.timeout * args.rate)
    for _ in range(n_steps):
        tx, ty, remaining = follower.target_and_progress(x, y)
        if remaining < args.goal_tol:
            return True, min_clear, worst_obs, traj
        v, w = expert.steer_to_point(x, y, yaw, (tx, ty),
                                     args.cruise_speed, args.k_heading)
        v = max(-expert.MAX_LIN, min(expert.MAX_LIN, v))
        w = max(-expert.MAX_ANG, min(expert.MAX_ANG, w))

        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = expert.wrap_angle(yaw + w * dt)
        traj.append((x, y))
        check(x, y)

    return False, min_clear, worst_obs, traj


def build_parser():
    p = expert.build_parser()   # reuse the SAME defaults as the live node
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    config = args.config or expert.default_config_path()
    cfg = yaml.safe_load(open(config))
    obstacles, goal = cfg['obstacles'], cfg['goal']
    starts = expert.load_poses(config, args.starts)

    # min_clear from simulate()/check() is already SURFACE-to-surface
    # clearance (robot radius subtracted out there) - the bar to clear
    # here is just the extra safety margin on top of that, not
    # ROBOT_RADIUS + SAFETY_MARGIN again (that would subtract the robot's
    # radius twice over).
    required = SAFETY_MARGIN
    print(f"robot radius {ROBOT_RADIUS:.3f} m (already subtracted out "
          f"below) + safety margin {SAFETY_MARGIN:.3f} m = {required:.3f} m "
          "minimum SURFACE clearance required\n")

    total, failures = 0, []
    worst_overall = float('inf')
    for label, x, y, yaw in starts:
        for side in ('left', 'right'):
            reached, min_clear, worst_obs, traj = simulate(
                x, y, yaw, side, obstacles, goal, args)
            total += 1
            ok = reached and min_clear >= required
            worst_overall = min(worst_overall, min_clear)
            if not ok:
                failures.append((label, side, reached, min_clear, worst_obs))
            status = 'OK' if ok else 'FAIL'
            print(f"start {label} side={side:5s}  reached={reached}  "
                  f"clearance={min_clear:+.3f}m (vs {worst_obs})  [{status}]")
            if args.verbose:
                for i, (px, py) in enumerate(traj):
                    if i % 10 == 0:
                        print(f"    step {i:3d}  ({px:+.3f},{py:+.3f})")

    print(f"\n{'=' * 64}")
    print(f" {total - len(failures)}/{total} runs OK "
          f"(reached goal AND cleared every obstacle by >= "
          f"{required:.3f} m)")
    print(f" worst clearance seen across ALL runs: {worst_overall:+.3f} m")
    if failures:
        print(f"\n {len(failures)} FAILURES:")
        for label, side, reached, min_clear, worst_obs in failures:
            reason = "never reached goal" if not reached else \
                f"clearance {min_clear:+.3f}m vs {worst_obs} (< {required:.3f}m)"
            print(f"   start {label} side={side}: {reason}")
        print(f"{'=' * 64}\n")
        sys.exit(1)
    else:
        print(f"{'=' * 64}\n")
        sys.exit(0)


if __name__ == '__main__':
    main()
