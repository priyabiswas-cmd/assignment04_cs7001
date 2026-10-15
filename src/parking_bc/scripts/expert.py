#!/usr/bin/env python3
"""
Scripted parking expert: a deterministic go-to-goal controller.

This is a formula, not a lookup table. Given the robot's current pose it
computes ONE correct action, every time, from any position - which is what
makes it usable both for generating clean demonstrations and for DAgger
relabelling (querying "what should I have done here?" at any state a
learned policy visits).

Strategy: aim at a point on the bay's centerline, a fixed look-ahead
distance ahead of the robot. Steer proportionally toward that point's
bearing, drive forward, slow down near the goal, stop when centred.

Aiming at a point ON THE CENTRELINE (not straight at the bay centre) is
what makes far-off starts curve in and merge onto the centreline well
before reaching the parked cars, rather than cutting a corner into them.
A start close to centre barely needs to turn at all - the same formula
handles both because the heading error it sees is naturally smaller.
"""

import math
from dataclasses import dataclass


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class ExpertParams:
    bay_x: float = 0.0
    bay_y: float = 0.16
    lookahead: float = 0.30      # how far ahead on the centreline to aim
    k_alpha: float = 2.5         # steering gain (rad/s per rad of heading error)
    v_nom: float = 0.12          # cruise speed, m/s
    v_min: float = 0.04          # never crawl slower than this while moving
    slow_radius: float = 0.20    # start slowing down within this distance (m)
    stop_radius: float = 0.05    # call it "arrived" within this distance (m)
    max_w: float = 1.2           # angular velocity cap, rad/s


def expert_action(x, y, yaw, params=ExpertParams()):
    """
    -> (v, w, done)

    done=True means the goal is reached and (v, w) = (0, 0).
    Pure function: no state, no history - safe to call from anywhere,
    including mid-rollout for DAgger relabelling.
    """
    p = params
    dist = math.hypot(x - p.bay_x, y - p.bay_y)
    if dist < p.stop_radius:
        return 0.0, 0.0, True

    # aim point: on the centreline, `lookahead` ahead, but never past the
    # bay centre itself
    aim_y = max(p.bay_y, y - p.lookahead)
    aim_x = p.bay_x

    desired_heading = math.atan2(aim_y - y, aim_x - x)
    alpha = wrap_angle(desired_heading - yaw)

    w = max(-p.max_w, min(p.max_w, p.k_alpha * alpha))

    if dist < p.slow_radius:
        v = p.v_min + (p.v_nom - p.v_min) * (dist / p.slow_radius)
    else:
        v = p.v_nom
    # slow down further when badly misaligned, so it turns before it drives
    v *= max(0.3, math.cos(alpha))

    return v, w, False


# --------------------------------------------------------------------------
# Self-test: simulate the expert from every configured start pose and check
# it (a) reaches the bay and (b) never enters either parked car's footprint.
# --------------------------------------------------------------------------

def simulate(x, y, yaw, params=ExpertParams(), dt=0.05, t_max=40.0):
    """Kinematic unicycle rollout under the expert. Returns the trajectory."""
    traj = [(x, y, yaw)]
    t = 0.0
    while t < t_max:
        v, w, done = expert_action(x, y, yaw, params)
        if done:
            break
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = wrap_angle(yaw + w * dt)
        traj.append((x, y, yaw))
        t += dt
    return traj, done


def car_boxes(bay_ratio=3.2, bay_depth=0.32, car_len=0.30, robot_len=0.138):
    """Same geometry generate_world.py uses - kept in sync by these numbers."""
    bay_len = bay_ratio * robot_len
    car_wid = bay_depth
    ax = -(bay_len / 2 + car_len / 2)
    bx = +(bay_len / 2 + car_len / 2)
    half_l, half_w = car_len / 2, car_wid / 2
    return [
        (ax - half_l, ax + half_l, 0.0, car_wid),   # car A: xmin,xmax,ymin,ymax
        (bx - half_l, bx + half_l, 0.0, car_wid),   # car B
    ]


def min_clearance(traj, boxes, margin=0.10):
    """Smallest distance from the trajectory to either car's footprint."""
    worst = float('inf')
    for x, y, _ in traj:
        for xmin, xmax, ymin, ymax in boxes:
            # distance from point to axis-aligned box (0 if inside)
            dx = max(xmin - x, 0, x - xmax)
            dy = max(ymin - y, 0, y - ymax)
            worst = min(worst, math.hypot(dx, dy))
    return worst


if __name__ == '__main__':
    starts = {
        'A': (0.50, 1.00, math.radians(270)),
        'B': (0.10, 1.00, math.radians(270)),
        'C': (-0.20, 1.00, math.radians(270)),
        'D': (-0.50, 1.00, math.radians(270)),
        'E (test)': (0.30, 1.00, math.radians(270)),
        'F (test)': (-0.05, 1.00, math.radians(270)),
        'G (test)': (-0.35, 1.00, math.radians(270)),
    }
    boxes = car_boxes()
    print(f"{'start':10s} {'reached':>8s} {'steps':>6s} "
          f"{'final dx':>9s} {'final dy':>9s} {'min clearance':>14s}")
    all_ok = True
    for label, (x0, y0, yaw0) in starts.items():
        traj, done = simulate(x0, y0, yaw0)
        fx, fy, _ = traj[-1]
        clr = min_clearance(traj, boxes)
        ok = done and clr > 0.0
        all_ok &= ok
        print(f"{label:10s} {'yes' if done else 'NO':>8s} {len(traj):6d} "
              f"{fx:+9.3f} {fy:+9.3f} {clr:14.3f}"
              f"{'  <-- COLLISION RISK' if clr <= 0.0 else ''}")
    print()
    print("all starts reach the bay without clipping a car:"
          if all_ok else "PROBLEM: check the flagged row(s) above")
