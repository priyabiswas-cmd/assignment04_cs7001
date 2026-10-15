#!/usr/bin/env python3
"""
Generate the single-room navigation world (visuomotor BC / DAgger exercise).

    python3 generate_room.py --out room_single.world

Deliberately kept to one room (see the project discussion): fast to
simulate (important once a scripted expert is generating hundreds of
demonstration episodes, and later running many DAgger rollouts), and
simple enough to isolate algorithm behaviour (regression-averaging at the
choice point, drift near the tight passage) rather than confound it with
long-horizon multi-room navigation. A house with several rooms is the
natural next step once this pipeline is validated - just more of the same
shape, not a different design.

Layout (all lengths in metres, origin at the room's south-west corner
area - start near y=0, goal near the far wall):

    - Perimeter walls, fully enclosed.
    - ONE central "choice point" obstacle (the kitchen island) with
      roughly equal open space on both sides - this is where a regression
      BC policy is expected to average left/right and drive into it,
      the vision-space equivalent of the double-gap-pipe / two-bay
      problem.
    - ONE tight passage (two boxes forming a narrow gap) elsewhere on the
      path - this is where small drift compounds, i.e. where DAgger has
      something real to fix.
    - A couple of off-path obstacles purely for visual landmark diversity
      (so the camera has real texture/structure to condition on, not a
      featureless box).
    - A floor-marked goal region near the far wall ("kitchen doorway").

The TurtleBot3 Burger footprint is 0.138 m (L) x 0.178 m (W).
"""

import argparse

ROBOT_LEN = 0.138
ROBOT_WID = 0.178

# Room footprint
ROOM_W = 3.0            # x extent: x in [-ROOM_W/2, +ROOM_W/2]
ROOM_L = 3.9            # y extent: y in [0, ROOM_L]
WALL_T = 0.05
WALL_H = 0.30           # tall enough to bound the camera/lidar view

# Choice-point obstacle (centred, open space on both sides)
ISLAND_Y = 1.8
ISLAND_SIZE = (0.6, 0.6, 0.30)
ISLAND_CLEAR = ROOM_W / 2.0 - ISLAND_SIZE[0] / 2.0   # clearance each side

# Tight passage (two boxes forming one narrow gap).
# GAP_Y leaves 0.6 m of clear y-space between the island's far edge and
# the gap boxes' near edge (island edge at 2.1, gap near edge at 2.4 in
# the original layout left only 0.3 m - verified with
# verify_expert_room.py to be narrower than 2x the robot's own safety
# clearance, i.e. geometrically impossible to cross without touching
# something, independent of how good the controller is). 0.6 m gives a
# comfortable margin instead of exactly the bare minimum.
GAP_Y = 2.9
GAP_WIDTH = 0.45        # robot is 0.178 m wide - real but passable margin
GAP_BOX = (0.4, 0.4, 0.30)

# Off-path decorative obstacles (visual diversity only, clear of the path).
# Mirrored left/right pairs, same size/shape/colour on both sides - a
# demonstrator (or the camera) should never have a reason to prefer one
# side over the other because "that side looks different". Only the
# obstacle TYPE varies between pairs, not the left/right placement of any
# one type, so the room still has two distinct visual landmarks for the
# CNN to key off while staying left-right symmetric overall.
DECOR = [
    # (name, size, abs(x), y, rgba) - instantiated at both +x and -x
    ("lamp",  (0.15, 0.15, 0.50), 1.15, 0.9, "0.90 0.75 0.20 1"),
    ("shelf", (0.5, 0.25, 0.20), 1.05, 3.4, "0.25 0.55 0.55 1"),
]

GOAL_Y = 3.6
GOAL_SIZE = (0.6, 0.5, 0.012)
GOAL_RGBA = "0.95 0.75 0.15 1"

START_POSE = (0.0, 0.4, 90.0)   # x, y, yaw degrees - faces +y, toward the goal


STOCK_LIGHT = "<include><uri>model://sun</uri></include>"

# Overhead light, straight down. The stock sun has a sideways direction
# (-0.5 0.1 -0.9) which lights left- and right-facing surfaces differently,
# a visual cue that breaks the left/right ambiguity. Used by room_overhead.world.
OVERHEAD_LIGHT = """    <!-- Overhead light, straight down (0 0 -1). The stock model://sun has a
         sideways direction (-0.5 0.1 -0.9), which lit left-facing and
         right-facing surfaces differently even though the geometry is
         mirror-symmetric - a visual cue that breaks the left/right
         ambiguity the exercise depends on. Straight down keeps the lighting
         symmetric about the centreline. -->
    <light name='overhead' type='directional'>
      <cast_shadows>false</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>0 0 -1</direction>
    </light>"""


def box_link(name, size, pose, rgba, collide=True):
    """One static box model. collide=False gives a visual-only marking."""
    sx, sy, sz = size
    collision = f"""
      <collision name='c'>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>""" if collide else ""
    return f"""
  <model name='{name}'>
    <static>true</static>
    <pose>{pose}</pose>
    <link name='link'>{collision}
      <visual name='v'>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>{rgba}</ambient>
          <diffuse>{rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>"""


def build_walls():
    parts = []
    wall_rgba = "0.85 0.85 0.82 1"
    cx, cy = 0.0, ROOM_L / 2.0

    # south / north (run along x)
    parts.append(box_link(
        "wall_south", (ROOM_W + 2 * WALL_T, WALL_T, WALL_H),
        f"{cx:.4f} {-WALL_T / 2.0:.4f} {WALL_H / 2.0:.4f} 0 0 0", wall_rgba))
    parts.append(box_link(
        "wall_north", (ROOM_W + 2 * WALL_T, WALL_T, WALL_H),
        f"{cx:.4f} {ROOM_L + WALL_T / 2.0:.4f} {WALL_H / 2.0:.4f} 0 0 0",
        wall_rgba))

    # west / east (run along y)
    parts.append(box_link(
        "wall_west", (WALL_T, ROOM_L + 2 * WALL_T, WALL_H),
        f"{-ROOM_W / 2.0 - WALL_T / 2.0:.4f} {cy:.4f} {WALL_H / 2.0:.4f} 0 0 0",
        wall_rgba))
    parts.append(box_link(
        "wall_east", (WALL_T, ROOM_L + 2 * WALL_T, WALL_H),
        f"{ROOM_W / 2.0 + WALL_T / 2.0:.4f} {cy:.4f} {WALL_H / 2.0:.4f} 0 0 0",
        wall_rgba))
    return parts


def build_room(light="stock"):
    parts = []
    parts.extend(build_walls())

    # Choice-point obstacle - the "kitchen island"
    parts.append(box_link(
        "island", ISLAND_SIZE,
        f"0 {ISLAND_Y:.4f} {ISLAND_SIZE[2] / 2.0:.4f} 0 0 0",
        "0.45 0.30 0.20 1"))

    # Tight passage
    left_x = -(GAP_WIDTH / 2.0 + GAP_BOX[0] / 2.0)
    right_x = +(GAP_WIDTH / 2.0 + GAP_BOX[0] / 2.0)
    parts.append(box_link(
        "gap_box_left", GAP_BOX,
        f"{left_x:.4f} {GAP_Y:.4f} {GAP_BOX[2] / 2.0:.4f} 0 0 0",
        "0.55 0.25 0.25 1"))
    parts.append(box_link(
        "gap_box_right", GAP_BOX,
        f"{right_x:.4f} {GAP_Y:.4f} {GAP_BOX[2] / 2.0:.4f} 0 0 0",
        "0.25 0.35 0.55 1"))

    # Decorative, off-path obstacles
    for name, size, abs_x, y, rgba in DECOR:
        for side, x in (("l", -abs_x), ("r", +abs_x)):
            parts.append(box_link(
                f"{name}_{side}", size,
                f"{x:.4f} {y:.4f} {size[2] / 2.0:.4f} 0 0 0", rgba))

    # Goal marking - visual only, flush with the floor
    parts.append(box_link(
        "goal_mark", GOAL_SIZE,
        f"0 {GOAL_Y:.4f} {GOAL_SIZE[2] / 2.0:.4f} 0 0 0",
        GOAL_RGBA, collide=False))

    body = "".join(parts)
    light_xml = STOCK_LIGHT if light == "stock" else OVERHEAD_LIGHT
    cam_y = ROOM_L / 2.0
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='room'>
    <include><uri>model://ground_plane</uri></include>
{light_xml}

    <physics type='ode'>
      <real_time_update_rate>1000</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
    </physics>

    <scene>
      <ambient>0.5 0.5 0.5 1</ambient>
      <shadows>false</shadows>
    </scene>
    <gui fullscreen='0'>
      <camera name='top_down'>
        <pose>0 {cam_y:.4f} 4.5 0 1.5707 1.5707</pose>
        <projection_type>orthographic</projection_type>
      </camera>
    </gui>
    <plugin name='gazebo_ros_state' filename='libgazebo_ros_state.so'>
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>50</update_rate>
    </plugin>
{body}

  </world>
</sdf>
""".replace("{LIGHT}", light_xml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="room_single.world")
    ap.add_argument("--light", choices=["stock", "overhead"], default="stock",
                    help="stock = the original sun (matches existing recordings); "
                         "overhead = symmetric straight-down light")
    args = ap.parse_args()

    with open(args.out, "w") as f:
        f.write(build_room(args.light))

    print(f"wrote {args.out}")
    print(f"  room {ROOM_W:.2f} x {ROOM_L:.2f} m")
    print(f"  choice-point island at y={ISLAND_Y:.2f}, "
          f"{ISLAND_CLEAR:.3f} m clear on each side "
          f"(robot width {ROBOT_WID:.3f} m)")
    print(f"  tight passage at y={GAP_Y:.2f}, gap {GAP_WIDTH:.3f} m "
          f"(robot width {ROBOT_WID:.3f} m, margin "
          f"{(GAP_WIDTH - ROBOT_WID) / 2:.3f} m each side)")
    print(f"  start  x={START_POSE[0]:+.2f} y={START_POSE[1]:+.2f} "
          f"yaw={START_POSE[2]:+.0f} deg")
    print(f"  goal   centred at y={GOAL_Y:.2f}")


if __name__ == "__main__":
    main()
