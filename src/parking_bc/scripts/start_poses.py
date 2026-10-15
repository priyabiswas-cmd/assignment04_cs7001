#!/usr/bin/env python3
"""
Manage the demonstration start poses: see them, go to them, change them.

The poses live in config/start_poses.yaml, which the recorder reads too,
so the two can never disagree.

Examples:

    # what is defined
    ros2 run parking_bc start_poses.py --list

    # draw all four in Gazebo as coloured discs with a heading spike
    ros2 run parking_bc start_poses.py --show

    # put the robot on start B
    ros2 run parking_bc start_poses.py --goto B

    # move start C and redraw
    ros2 run parking_bc start_poses.py --set C --x 1.10 --y 0.50 --show

    # clear the markers again
    ros2 run parking_bc start_poses.py --hide

The markers are visual only - no collision - so they cannot affect the
robot or the LiDAR.
"""

import argparse
import math
import os
import re
import sys

import yaml

# rclpy and the gazebo services are imported lazily inside _ros(), so that
# --list and --set work without ROS sourced and without a running simulator.


MARKER_PREFIX = "start_marker_"

# one colour per start, in order
COLOURS = [
    ("0.95 0.75 0.15 1", "amber"),
    ("0.30 0.75 0.45 1", "green"),
    ("0.35 0.55 0.90 1", "blue"),
    ("0.85 0.40 0.75 1", "pink"),
]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

def default_config_path():
    """config/start_poses.yaml, whether installed or run from source."""
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
    with open(path) as f:
        cfg = yaml.safe_load(f)
    sets = cfg.get('sets', {})
    if set_name not in sets:
        raise SystemExit(f"set '{set_name}' not in {path}. "
                         f"available: {', '.join(sorted(sets))}")
    out = []
    for e in sets[set_name]:
        out.append((str(e['label']), float(e['x']), float(e['y']),
                    math.radians(float(e['yaw']))))
    return out


def find_set_block(lines, set_name):
    """
    Locate a set in the raw text.

    Returns (header_index, first_entry_index, end_index, entry_indent).
    end_index is one past the last entry line, so a new entry inserted
    there lands at the bottom of the set.  Returns None if not found.
    """
    header = None
    set_indent = None
    first = None
    last = None
    entry_indent = 4

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        indent = len(line) - len(line.lstrip())

        if header is None:
            if stripped == f"{set_name}:":
                header = i
                set_indent = indent
            continue

        # a key at the same or shallower indent ends the block
        if indent <= set_indent and not stripped.startswith('-'):
            break
        if stripped.startswith('-'):
            entry_indent = indent
            if first is None:
                first = i
            last = i

    if header is None:
        return None
    if first is None:                      # set exists but is empty
        return header, header + 1, header + 1, entry_indent
    return header, first, last + 1, entry_indent


def entry_line(indent, label, x, y, yaw):
    return (f"{' ' * indent}- {{label: {label}, x: {x}, "
            f"y: {y}, yaw: {yaw}}}")


def add_pose(path, set_name, label, x, y, yaw, create_set=False):
    """Append a pose to a set, optionally creating the set."""
    with open(path) as f:
        text = f.read()
    lines = text.split('\n')

    cfg = yaml.safe_load(text) or {}
    sets = cfg.setdefault('sets', {}) or {}

    if set_name in sets and any(str(e['label']) == label
                                for e in (sets[set_name] or [])):
        raise SystemExit(
            f"start '{label}' already exists in set '{set_name}' - "
            f"use --set to change it, or --delete first")

    block = find_set_block(lines, set_name)

    if block is None:
        if not create_set:
            raise SystemExit(
                f"set '{set_name}' does not exist. Add --new-set to create "
                f"it.\navailable: {', '.join(sorted(sets))}")
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ['', f"  {set_name}:",
                  entry_line(4, label, x, y, yaw), '']
    else:
        _, _, end, indent = block
        lines.insert(end, entry_line(indent, label, x, y, yaw))

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def delete_pose(path, set_name, label):
    """Remove one pose from a set."""
    with open(path) as f:
        lines = f.read().split('\n')

    block = find_set_block(lines, set_name)
    if block is None:
        raise SystemExit(f"set '{set_name}' not found in {path}")
    _, first, end, _ = block

    for i in range(first, end):
        if re.search(rf"label:\s*{re.escape(label)}\b", lines[i]):
            del lines[i]
            with open(path, 'w') as f:
                f.write('\n'.join(lines))
            return
    raise SystemExit(f"no start labelled '{label}' in set '{set_name}'")


def save_pose(path, set_name, label, x=None, y=None, yaw_deg=None):
    """
    Update one entry in place.

    Rewritten line by line rather than via yaml.safe_dump, because dumping
    would discard every comment in the file - including the explanation of
    what the sets are for.
    """
    with open(path) as f:
        text = f.read()

    cfg = yaml.safe_load(text)
    if set_name not in cfg.get('sets', {}):
        raise SystemExit(f"set '{set_name}' not in {path}")
    entry = next((e for e in cfg['sets'][set_name]
                  if str(e['label']) == label), None)
    if entry is None:
        raise SystemExit(f"no start labelled '{label}' in set '{set_name}'")

    new_x = entry['x'] if x is None else round(float(x), 4)
    new_y = entry['y'] if y is None else round(float(y), 4)
    new_yaw = entry['yaw'] if yaw_deg is None else round(float(yaw_deg), 2)

    lines = text.split('\n')
    in_set = False
    set_indent = None
    done = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())

        if stripped == f"{set_name}:":
            in_set = True
            set_indent = indent
            continue

        if in_set:
            # a key at the same or shallower indent ends this set's block
            if indent <= set_indent and not stripped.startswith('-'):
                break
            if re.search(rf"label:\s*{re.escape(label)}\b", stripped):
                pad = ' ' * indent
                lines[i] = (f"{pad}- {{label: {label}, x: {new_x}, "
                            f"y: {new_y}, yaw: {new_yaw}}}")
                done = True
                break

    if not done:
        raise SystemExit(f"could not locate '{label}' in the text of {path}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return cfg['sets'][set_name]


# --------------------------------------------------------------------------
# Marker SDF
# --------------------------------------------------------------------------

def marker_sdf(name, rgba):
    """
    A flat disc with a spike pointing along +x of the model frame, so the
    marker shows position AND heading.  Visual only: no collision element.
    """
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <visual name='disc'>
        <geometry><cylinder><radius>0.07</radius>
          <length>0.004</length></cylinder></geometry>
        <material>
          <ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>
        </material>
      </visual>
      <visual name='heading'>
        <pose>0.09 0 0 0 0 0</pose>
        <geometry><box><size>0.10 0.02 0.004</size></box></geometry>
        <material>
          <ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def quaternion_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


# --------------------------------------------------------------------------

def _ros():
    """Import the ROS bits on demand; give a clear error if unavailable."""
    try:
        import rclpy
        from rclpy.node import Node
        from gazebo_msgs.srv import (DeleteEntity, SetEntityState,
                                     SpawnEntity)
        return rclpy, Node, DeleteEntity, SetEntityState, SpawnEntity
    except ImportError as e:
        raise SystemExit(
            f"ROS 2 not available ({e}).\n"
            "Source your workspace first:\n"
            "    source /opt/ros/humble/setup.bash\n"
            "    source ~/colcon_ws/install/setup.bash\n"
            "(--list and --set work without this; --show/--goto do not.)")


def make_tool():
    rclpy, Node, DeleteEntity, SetEntityState, SpawnEntity = _ros()

    class StartPoseTool(Node):

        def __init__(self):
            super().__init__('start_poses')
            self.spawn = self.create_client(SpawnEntity, '/spawn_entity')
            self.delete = self.create_client(DeleteEntity, '/delete_entity')
            self.setstate = self.create_client(
                SetEntityState, '/gazebo/set_entity_state')

        def wait(self, client, what):
            if client.service_is_ready():
                return True
            client.wait_for_service(timeout_sec=3.0)
            if not client.service_is_ready():
                print(f"!! {what} unavailable - is the simulator running?")
                return False
            return True

        def call(self, client, req):
            fut = client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            return fut.result()

        # -- actions -----------------------------------------------------------

        def hide(self, poses):
            if not self.wait(self.delete, '/delete_entity'):
                return
            for label, *_ in poses:
                req = DeleteEntity.Request()
                req.name = MARKER_PREFIX + label
                self.call(self.delete, req)
            print(f"removed {len(poses)} marker(s)")

        def show(self, poses):
            if not self.wait(self.spawn, '/spawn_entity'):
                return
            self.hide(poses)                    # idempotent: clear then redraw
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
                res = self.call(self.spawn, req)

                ok = "ok" if (res and res.success) else "FAILED"
                print(f"  {label}  x={x:+.2f} y={y:+.2f} "
                      f"yaw={math.degrees(yaw):+.0f} deg  {colour:6s} {ok}")

        def goto(self, poses, label, robot):
            match = [p for p in poses if p[0] == label]
            if not match:
                raise SystemExit(f"no start labelled '{label}'")
            _, x, y, yaw = match[0]

            if not self.wait(self.setstate, '/gazebo/set_entity_state'):
                return
            qx, qy, qz, qw = quaternion_from_yaw(yaw)
            req = SetEntityState.Request()
            req.state.name = robot
            req.state.pose.position.x = x
            req.state.pose.position.y = y
            req.state.pose.position.z = 0.01
            req.state.pose.orientation.x = qx
            req.state.pose.orientation.y = qy
            req.state.pose.orientation.z = qz
            req.state.pose.orientation.w = qw
            res = self.call(self.setstate, req)

            if res and res.success:
                print(f"robot at start {label}: x={x:+.2f} y={y:+.2f} "
                      f"yaw={math.degrees(yaw):+.0f} deg")
            else:
                print("!! set_entity_state failed - is the robot name right?")

    return StartPoseTool(), rclpy


def print_list(poses, set_name, path):
    print(f"\nset '{set_name}'   ({path})\n")
    for i, (label, x, y, yaw) in enumerate(poses):
        _, colour = COLOURS[i % len(COLOURS)]
        print(f"  {label}   x = {x:+6.2f}   y = {y:+6.2f}   "
              f"yaw = {math.degrees(yaw):+6.1f} deg   [{colour}]")
    print()


def make_tool_initialised():
    rclpy_mod = _ros()[0]
    rclpy_mod.init()
    node, rclpy_mod = make_tool()
    return node, rclpy_mod


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None,
                   help='path to start_poses.yaml')
    p.add_argument('--starts', default='right',
                   help="which set in the YAML (default: right)")
    p.add_argument('--robot', default='burger')

    p.add_argument('--list', action='store_true',
                   help='print the poses and exit')
    p.add_argument('--list-sets', action='store_true',
                   help='print every set defined in the file')
    p.add_argument('--show', action='store_true',
                   help='draw markers in Gazebo')
    p.add_argument('--hide', action='store_true',
                   help='remove the markers')
    p.add_argument('--goto', metavar='LABEL',
                   help='teleport the robot to this start')

    p.add_argument('--set', metavar='LABEL', dest='set_label',
                   help='edit an existing start, with --x / --y / --yaw')
    p.add_argument('--add', metavar='LABEL', dest='add_label',
                   help='add a new start; needs --x and --y')
    p.add_argument('--delete', metavar='LABEL', dest='del_label',
                   help='remove a start from the set')
    p.add_argument('--new-set', action='store_true',
                   help='with --add, create the set if it does not exist')
    p.add_argument('--x', type=float)
    p.add_argument('--y', type=float)
    p.add_argument('--yaw', type=float, help='degrees')
    return p


def main():
    args = build_parser().parse_args()
    path = args.config or default_config_path()

    if not os.path.exists(path):
        raise SystemExit(f"config not found: {path}")

    if args.list_sets:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"\nsets in {path}:\n")
        for name, entries in (cfg.get('sets') or {}).items():
            labels = ', '.join(str(e['label']) for e in (entries or []))
            print(f"  {name:8s} {len(entries or [])} poses  [{labels}]")
        print()
        return

    if args.del_label:
        delete_pose(path, args.starts, args.del_label)
        print(f"deleted start {args.del_label} from set '{args.starts}'")

    if args.add_label:
        if args.x is None or args.y is None:
            raise SystemExit("--add needs --x and --y (--yaw defaults to 270)")
        add_pose(path, args.starts, args.add_label,
                 round(args.x, 4), round(args.y, 4),
                 round(args.yaw if args.yaw is not None else 270.0, 2),
                 create_set=args.new_set)
        print(f"added start {args.add_label} to set '{args.starts}'")

    if args.set_label:
        if args.x is None and args.y is None and args.yaw is None:
            raise SystemExit("--set needs at least one of --x / --y / --yaw")
        save_pose(path, args.starts, args.set_label,
                  args.x, args.y, args.yaw)
        print(f"updated start {args.set_label} in {path}")

    poses = load_poses(path, args.starts)

    needs_ros = args.show or args.hide or args.goto
    if args.list or not needs_ros:
        print_list(poses, args.starts, path)
        if not needs_ros:
            return

    node, rclpy = make_tool_initialised()
    try:
        if args.hide:
            node.hide(poses)
        if args.show:
            node.show(poses)
        if args.goto:
            node.goto(poses, args.goto, args.robot)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
