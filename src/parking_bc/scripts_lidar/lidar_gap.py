#!/usr/bin/env python3
"""
Shared "find the gap between two blocks" logic for the odometry+lidar
pipeline in this folder.

Same idea a car's ultrasonic self-parking system uses, done with a 2D
lidar instead of one side sensor. Looking across the beams in front of
the robot, a parking slot shows up as

    near (block A) -> far (open space) -> near (block B)

The two BOUNDARY points (the last near-beam of block A, the first
near-beam of block B) are reliable, finite-range hits on real surfaces -
unlike the "far" beam itself, which may return the lidar's max range or
"no return" for a wide-open gap and is therefore useless as a distance
measurement. The midpoint between those two boundary points is the gap
centre, in the robot's own local (forward, left) frame at the instant of
that one scan.

That local estimate is only good for the instant it was taken - a lidar
scan has no idea where "me" was a second ago. Composing it with the
robot's ODOMETRY pose at that same instant turns "the gap is 0.4 m ahead
and a bit left of me right now" into a position that stays meaningful as
the robot keeps moving. That composing step is the only reason odometry
is needed here.

This module is imported by both bc_dataset_lidar.py (offline, with the
benefit of hindsight over a whole recorded episode) and
bc_policy_node_lidar.py (online, causal, one scan at a time) so the two
never disagree about what "the gap" means.

Tuned for the stock TurtleBot3 LDS-01 (360 beams, 1 degree apart, full
360 degree sweep). If you switch lidar models, re-check `min_run` -
it's a beam COUNT, and here that happens to equal degrees.
"""

import math


DEFAULTS = dict(
    near_thresh=0.5,   # m - closer than this counts as "a block, not open space"
    min_run=3,         # beams - reject runs shorter than this as noise
    fov_deg=100.0,     # only look this far left/right of straight ahead
    lock_count=3,      # consecutive agreeing detections needed to lock
    lock_tol=0.05,     # m - "agreeing" means within this of each other
)


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _local_xy(rng, angle):
    return rng * math.cos(angle), rng * math.sin(angle)


def find_gap_local(ranges, angle_min, angle_increment,
                    near_thresh=DEFAULTS['near_thresh'],
                    min_run=DEFAULTS['min_run'],
                    fov_deg=DEFAULTS['fov_deg']):
    """
    Look for near -> far -> near across the beams within +/-fov_deg of
    straight ahead (0 rad = the robot's own forward direction).

    Returns (x_local, y_local) of the gap centre in the robot's frame
    (x forward, y left), or None if no clean triplet is found.
    """
    n = len(ranges)
    if n == 0:
        return None
    fov = math.radians(fov_deg)

    def angle_of(i):
        return wrap_angle(angle_min + i * angle_increment)

    def is_near(i):
        r = ranges[i]
        return math.isfinite(r) and 0.0 < r < near_thresh

    # Sort beams by their wrapped angle first, THEN filter to the FOV.
    # Raw beam index order jumps from +pi to -pi somewhere in the array;
    # sorting makes "straight ahead" the middle of a contiguous run
    # regardless of where that raw wrap-around seam happens to fall.
    order = sorted(range(n), key=angle_of)
    idx = [i for i in order if abs(angle_of(i)) <= fov]
    m = len(idx)

    i = 0
    while i < m:
        j = i
        while j < m and is_near(idx[j]):
            j += 1
        run1 = j - i
        if run1 < min_run:
            i += 1
            continue
        edge_a = idx[j - 1]                 # last near beam of block A

        k = j
        while k < m and not is_near(idx[k]):
            k += 1
        run2 = k - j
        if run2 < min_run:
            i = j
            continue

        l = k
        while l < m and is_near(idx[l]):
            l += 1
        run3 = l - k
        if run3 < min_run:
            i = k
            continue
        edge_b = idx[k]                     # first near beam of block B

        xa, ya = _local_xy(ranges[edge_a], angle_of(edge_a))
        xb, yb = _local_xy(ranges[edge_b], angle_of(edge_b))
        return (xa + xb) / 2.0, (ya + yb) / 2.0

    return None


def local_to_frame(x_local, y_local, pose_x, pose_y, pose_yaw):
    """Rotate+translate a robot-frame point into whatever frame `pose` is in
    (here: the odometry frame, using the robot's own odometry pose)."""
    c, s = math.cos(pose_yaw), math.sin(pose_yaw)
    return (pose_x + c * x_local - s * y_local,
            pose_y + s * x_local + c * y_local)


class GapEstimator:
    """
    Fuses repeated single-scan gap estimates (each already converted into
    a common frame via `local_to_frame`) into one running estimate, and
    LOCKS it once several agree closely - so one bad detection early in
    an episode can't wreck the whole thing, and the estimate stops
    drifting once it's trustworthy (the bay itself does not move mid
    manoeuvre).
    """

    def __init__(self, lock_count=DEFAULTS['lock_count'],
                 lock_tol=DEFAULTS['lock_tol']):
        self.lock_count = lock_count
        self.lock_tol = lock_tol
        self.hits = []          # [(x, y), ...] in the fusion frame
        self.locked = None

    def update(self, xy):
        """Feed one frame-space estimate (or None if this scan found nothing)."""
        if self.locked is not None or xy is None:
            return
        self.hits.append(xy)
        if len(self.hits) < self.lock_count:
            return
        recent = self.hits[-self.lock_count:]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        if (max(xs) - min(xs) <= self.lock_tol
                and max(ys) - min(ys) <= self.lock_tol):
            self.locked = (sum(xs) / len(xs), sum(ys) / len(ys))

    def estimate(self):
        """Best estimate so far: locked value, else running mean, else None."""
        if self.locked is not None:
            return self.locked
        if self.hits:
            xs = [p[0] for p in self.hits]
            ys = [p[1] for p in self.hits]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return None

    @property
    def has_estimate(self):
        return self.locked is not None or bool(self.hits)
