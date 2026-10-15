#!/usr/bin/env python3
"""
Visualise start poses from config/start_goal.yaml in Gazebo - the
room_nav_bc counterpart of parking_bc's start_poses.py.

    ros2 run room_nav_bc show_starts_room.py --starts train --show
    ros2 run room_nav_bc show_starts_room.py --starts test  --show
    ros2 run room_nav_bc show_starts_room.py --starts train --hide

Draws a flat coloured disc with a heading spike per start - visual only,
no collision, so it never affects the robot or its sensors.
"""

import argparse
import math
import os

import yaml

MARKER_PREFIX = "start_marker_"
COLOURS = [
    ("0.95 0.75 0.15 1", "amber"),
    ("0.30 0.75 0.45 1", "green"),
    ("0.35 0.55 0.90 1", "blue"),
    ("0.85 0.40 0.75 1", "pink"),
]


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
    sets = cfg.get('sets', {})
    if set_name not in sets:
        raise SystemExit(f"set '{set_name}' not in {path}. "
                         f"available: {', '.join(sorted(sets))}")
    return [(str(e['label']), float(e['x']), float(e['y']),
             math.radians(float(e['yaw']))) for e in sets[set_name]]


def marker_sdf(name, rgba):
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <visual name='disc'>
        <geometry><cylinder><radius>0.07</radius>
          <length>0.004</length></cylinder></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
      <visual name='heading'>
        <pose>0.09 0 0 0 0 0</pose>
        <geometry><box><size>0.10 0.02 0.004</size></box></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
      </visual>
    </link>
  </model>"""


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None)
    ap.add_argument('--starts', default='train')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--hide', action='store_true')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    path = args.config or default_config_path()
    poses = load_poses(path, args.starts)

    if args.list or not (args.show or args.hide):
        print(f"\nset '{args.starts}'   ({path})\n")
        for i, (label, x, y, yaw) in enumerate(poses):
            _, colour = COLOURS[i % len(COLOURS)]
            print(f"  {label}   x = {x:+6.2f}   y = {y:+6.2f}   "
                  f"yaw = {math.degrees(yaw):+6.1f} deg   [{colour}]")
        print()
        if not (args.show or args.hide):
            return

    import rclpy
    from rclpy.node import Node
    from gazebo_msgs.srv import DeleteEntity, SpawnEntity

    rclpy.init()
    node = Node('show_starts_room')
    spawn = node.create_client(SpawnEntity, '/spawn_entity')
    delete = node.create_client(DeleteEntity, '/delete_entity')

    def wait(client, what):
        if client.service_is_ready():
            return True
        client.wait_for_service(timeout_sec=3.0)
        if not client.service_is_ready():
            print(f"!! {what} unavailable - is the simulator running?")
            return False
        return True

    def call(client, req):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        return fut.result()

    if args.hide or args.show:
        if wait(delete, '/delete_entity'):
            for label, *_ in poses:
                req = DeleteEntity.Request()
                req.name = MARKER_PREFIX + label
                call(delete, req)
            if args.hide:
                print(f"removed {len(poses)} marker(s)")

    if args.show and wait(spawn, '/spawn_entity'):
        for i, (label, x, y, yaw) in enumerate(poses):
            rgba, colour = COLOURS[i % len(COLOURS)]
            name = MARKER_PREFIX + label
            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            req = SpawnEntity.Request()
            req.name = name
            req.xml = marker_sdf(name, rgba)
            req.initial_pose.position.x = x
            req.initial_pose.position.y = y
            req.initial_pose.position.z = 0.004
            req.initial_pose.orientation.x = qx
            req.initial_pose.orientation.y = qy
            req.initial_pose.orientation.z = qz
            req.initial_pose.orientation.w = qw
            res = call(spawn, req)
            ok = "ok" if (res and res.success) else "FAILED"
            print(f"  {label}  x={x:+.2f} y={y:+.2f} "
                  f"yaw={math.degrees(yaw):+.0f} deg  {colour:6s} {ok}")

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
