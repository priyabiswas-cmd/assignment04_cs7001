#!/usr/bin/env python3
"""
Generate the parallel-parking Gazebo world.

The bay geometry is parameterized so that the same source produces the
training world and the held-out variants used to test generalization.

    python3 generate_world.py --preset default  --out parking_grid.world
    python3 generate_world.py --preset narrow   --out parking_narrow.world
    python3 generate_world.py --preset wide     --out parking_wide.world

All lengths are in metres. The TurtleBot3 Burger footprint is
0.138 m (L) x 0.178 m (W), so the bay length is expressed as a multiple
of the robot length to keep the difficulty ratio explicit.
"""

import argparse

ROBOT_LEN = 0.138
ROBOT_WID = 0.178

# Presets: (bay_length_ratio, bay_depth, car_length, n_bays)
# n_bays=1 (the default) is the original single-gap task. n_bays=2 adds a
# third parked car, creating a SECOND bay of the same size - the "double
# gap" counterpart to CS224R HW1's hard-mode Flappy Bird pipes: a
# demonstrator facing two equally valid bays will genuinely split across
# episodes which one they choose, which is the multimodality that plain
# MSE regression BC is bad at and flow matching / DAgger address two
# different ways.
PRESETS = {
    "default": dict(bay_ratio=3.2, bay_depth=0.32, car_len=0.30, n_bays=1),
    "narrow":  dict(bay_ratio=2.6, bay_depth=0.28, car_len=0.30, n_bays=1),
    "wide":    dict(bay_ratio=4.0, bay_depth=0.36, car_len=0.30, n_bays=1),
    "twobay":  dict(bay_ratio=3.2, bay_depth=0.32, car_len=0.30, n_bays=2),
}

# Arena / grid
ARENA_X = 4.0          # total drivable length along the curb
ARENA_Y = 3.0          # total drivable depth away from the curb
GRID_STEP = 0.25       # visual grid spacing
GRID_W = 0.008         # grid line width
CURB_Y = 0.0           # curb line sits at y = 0; robot drives in y > 0

# Flat markings are lifted clear of the ground plane so the renderer does
# not z-fight with it. 6 mm is below the robot's ground clearance.
MARK_T = 0.010         # thickness of flat visual markings
MARK_Z = 0.006         # centre height of flat visual markings

CAR_HEIGHT = 0.20


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


def build_grid():
    """Visual-only grid lines. No collision: they must not affect the LiDAR."""
    parts = []
    grey = "0.30 0.30 0.33 1"
    n_along = int(round(ARENA_X / GRID_STEP)) + 1
    n_across = int(round(ARENA_Y / GRID_STEP)) + 1

    for i in range(n_along):
        x = -ARENA_X / 2.0 + i * GRID_STEP
        parts.append(box_link(
            f"grid_v_{i}",
            (GRID_W, ARENA_Y, MARK_T),
            f"{x:.4f} {ARENA_Y / 2.0:.4f} {MARK_Z} 0 0 0",
            grey, collide=False))

    for j in range(n_across):
        y = j * GRID_STEP
        parts.append(box_link(
            f"grid_h_{j}",
            (ARENA_X, GRID_W, MARK_T),
            f"0 {y:.4f} {MARK_Z} 0 0 0",
            grey, collide=False))
    return parts


CAR_COLOURS = [
    "0.55 0.20 0.18 1",   # red
    "0.20 0.35 0.55 1",   # blue
    "0.30 0.50 0.25 1",   # green
]


def build_world(bay_ratio, bay_depth, car_len, n_bays=1):
    bay_len = bay_ratio * ROBOT_LEN
    # The parked cars define the bays: their width IS the bay depth, and
    # they sit flush against the curb, so the bay footprint and the cars
    # line up. n_bays bays need n_bays+1 cars (car, bay, car, bay, car...),
    # laid out left to right and then centred on x=0.
    car_wid = bay_depth
    car_cy = bay_depth / 2.0
    n_cars = n_bays + 1

    total_w = n_cars * car_len + n_bays * bay_len
    x = -total_w / 2.0
    car_centres, bay_centres = [], []
    for i in range(n_cars):
        car_centres.append(x + car_len / 2.0)
        x += car_len
        if i < n_bays:
            bay_centres.append(x + bay_len / 2.0)
            x += bay_len

    parts = []

    # Curb wall behind the bays
    parts.append(box_link(
        "curb", (ARENA_X, 0.04, 0.15),
        f"0 {CURB_Y - 0.02:.4f} 0.075 0 0 0", "0.35 0.35 0.38 1"))

    # Parked cars
    for i, cx in enumerate(car_centres):
        name = f"parked_car_{chr(ord('a') + i)}"
        parts.append(box_link(
            name, (car_len, car_wid, CAR_HEIGHT),
            f"{cx:.4f} {car_cy:.4f} {CAR_HEIGHT / 2:.4f} 0 0 0",
            CAR_COLOURS[i % len(CAR_COLOURS)]))

    # Bay outlines: visual only, mark each goal region for the report figures
    edge = 0.012
    for k, bx in enumerate(bay_centres):
        for j, (sx, sy, px, py) in enumerate([
            (bay_len, edge, bx, bay_depth),
            (edge, bay_depth, bx - bay_len / 2.0, bay_depth / 2.0),
            (edge, bay_depth, bx + bay_len / 2.0, bay_depth / 2.0),
        ]):
            parts.append(box_link(
                f"bay_mark_{k}_{j}", (sx, sy, MARK_T),
                f"{px:.4f} {py:.4f} {MARK_Z} 0 0 0",
                "0.95 0.75 0.15 1", collide=False))

    parts.extend(build_grid())

    body = "".join(parts)
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='parking'>
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>

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
        <pose>0 0.16 2.2 0 1.5707 1.5707</pose>
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
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="default", choices=sorted(PRESETS))
    ap.add_argument("--out", default="parking_grid.world")
    args = ap.parse_args()

    cfg = PRESETS[args.preset]
    sdf = build_world(**cfg)
    with open(args.out, "w") as f:
        f.write(sdf)

    bay_len = cfg["bay_ratio"] * ROBOT_LEN
    n_bays = cfg.get("n_bays", 1)
    print(f"wrote {args.out}  [preset={args.preset}]")
    print(f"  bay length {bay_len:.3f} m  = {cfg['bay_ratio']:.2f} x robot length")
    print(f"  bay depth  {cfg['bay_depth']:.3f} m  "
          f"(robot width {ROBOT_WID:.3f} m, margin "
          f"{cfg['bay_depth'] - ROBOT_WID:.3f} m)")
    print(f"  parked car {cfg['car_len']:.3f} x {cfg['bay_depth']:.3f} m, "
          f"flush to curb")
    if n_bays > 1:
        n_cars = n_bays + 1
        total_w = n_cars * cfg['car_len'] + n_bays * bay_len
        x = -total_w / 2.0
        centres = []
        for i in range(n_cars):
            x += cfg['car_len']
            if i < n_bays:
                centres.append(x + bay_len / 2.0)
                x += bay_len
        print(f"  {n_bays} bays, centred at x = "
              + ", ".join(f"{c:+.4f}" for c in centres))


if __name__ == "__main__":
    main()
