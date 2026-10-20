#!/usr/bin/env python3
"""
Vanilla policy gradient (REINFORCE) for the room-navigation task - Part 3
of the assignment, and the first ONLINE method here: BC and DAgger both
train offline on logged (state, action) pairs; this one drives the real
robot in Gazebo, collects a reward per step, and updates the policy from
the reward it actually got. No demonstrations at all.

    ros2 run room_nav_bc pg_train_room.py --episodes 20     # smoke test
    ros2 run room_nav_bc pg_train_room.py --episodes 200    # real run

WHY NO CAMERA.  A vision policy learning from scratch with plain REINFORCE
needs a very large number of episodes before reward differences are even
distinguishable from noise - thousands to tens of thousands is typical.
At one real-time Gazebo episode per ~15-20s, that is many hours to days of
wall-clock time, not a feasible assignment turnaround. Using the same
4-dim odometry-only observation as the raw-lidar policy earlier in this
project keeps the network tiny and the problem low-dimensional enough
that a few hundred episodes (a few minutes to an hour) shows a real
learning curve. This mirrors how RL courses sequence it: policy gradient
on low-dimensional control first, pixels later. Run with
--gui false (room_demo.launch.py) for a wall-clock speedup, since the
camera is never read here anyway.

REWARD (computed from ground truth /gazebo/model_states - privileged,
exactly like every grading/reward signal elsewhere in this project; the
POLICY only ever sees odometry):
    + progress toward the goal this step (closer = positive)
    - a small per-step penalty (discourages dawdling)
    - a large penalty and immediate episode end on collision
    + a large bonus on reaching the goal and coming to a stop

THE ALGORITHM (see pg_losses.py for the two TODO-sized functions):
    1. Roll out --episodes-per-update episodes with the CURRENT policy,
       sampling actions from its Gaussian (no gradient tracking - this is
       data collection, not a differentiable step).
    2. For each episode, turn its rewards into discounted returns
       (compute_returns) and normalise them across the whole batch
       (normalize) - this is the simple REINFORCE baseline.
    3. Recompute log pi(a|s) for every (state, action) in the batch WITH
       gradients on, under the (still current - no update has happened
       yet) policy, and take one gradient step on policy_gradient_loss.
    4. Repeat.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

try:
    import torch
except ImportError:
    raise SystemExit("PyTorch not found - install it to train.")

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_policy_node_room import (  # noqa: E402
    MAX_ANG, MAX_LIN, default_config_path, in_goal, load_poses,
    quaternion_from_yaw, worst_clearance, wrap_angle, yaw_from_quaternion,
)
from pg_losses import compute_returns, normalize, policy_gradient_loss  # noqa: E402
from pg_networks import GaussianPolicy  # noqa: E402


class RunningNormalizer:
    """Online mean/std (Welford's algorithm) for the observation, since
    there is no fixed dataset to normalise against here - every episode is
    freshly generated. Saved into the checkpoint so test-time uses the
    same statistics training converged under, not a moving target."""

    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros(dim, np.float64)
        self.m2 = np.zeros(dim, np.float64)

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self):
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(np.maximum(self.m2 / (self.n - 1), 1e-8))

    def normalize(self, x):
        return (np.asarray(x, np.float64) - self.mean) / (self.std + 1e-6)

    def state_dict(self):
        return dict(n=self.n, mean=self.mean.copy(), m2=self.m2.copy())

    def load_state_dict(self, d):
        self.n, self.mean, self.m2 = d['n'], np.array(d['mean']), np.array(d['m2'])


class PGTrainer(Node):

    def __init__(self, args, obstacles, goal, room):
        super().__init__('pg_train_room')
        self.args = args
        self.obstacles, self.goal, self.room = obstacles, goal, room
        self.starts = load_poses(args.config, args.starts)
        self.rng = np.random.default_rng(args.seed)

        self.policy = GaussianPolicy(obs_dim=4, act_dim=2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.normalizer = RunningNormalizer(4)

        self.pose = None          # ground truth - reward/termination ONLY
        self.odom = None          # what the policy senses
        self.odom_origin = None

        self.episode = 0
        self.batch = []           # list of per-episode dicts
        self.cur = None
        self.t_start = None
        self.still_since = None
        self.prev_dist = None
        self.running = False

        self.ep_log = []          # for the CSV / learning curve

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self.on_states, qos)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)
        self.pub = self.create_publisher(Twist, args.cmd_out, 10)
        self.reset_cli = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.create_timer(1.0 / args.rate, self.step)

        print(f"\nREINFORCE  obs=4 (odometry only)  "
              f"episodes={args.episodes}  episodes/update={args.episodes_per_update}  "
              f"gamma={args.gamma}  lr={args.lr}\n")
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

    # -- episodes ------------------------------------------------------

    def begin_episode(self):
        label, x, y, yaw = self.starts[self.rng.integers(len(self.starts))]
        x += self.rng.uniform(-self.args.jitter_xy, self.args.jitter_xy)
        y += self.rng.uniform(-self.args.jitter_xy, self.args.jitter_xy)
        yaw += math.radians(
            self.rng.uniform(-self.args.jitter_yaw_deg, self.args.jitter_yaw_deg))
        self.publish(0.0, 0.0)
        self.teleport(x, y, yaw)
        # see bc_policy_node_room.py / expert_demo_room.py for why: avoid
        # acting on a pose/odom left over from the end of the PREVIOUS
        # episode before a fresh post-teleport reading arrives.
        self.pose = None
        self.odom = None
        self.odom_origin = None

        self.cur = dict(label=label, obs=[], act=[], rew=[])
        self.t_start = time.time()
        self.still_since = None
        self.prev_dist = None
        self.running = True

    def finish_episode(self, outcome, success, collided):
        self.publish(0.0, 0.0)
        self.running = False
        ep_return = float(sum(self.cur['rew']))
        n = len(self.cur['rew'])
        self.batch.append(self.cur)
        self.episode += 1
        self.ep_log.append(dict(episode=self.episode, label=self.cur['label'],
                                outcome=outcome, success=success,
                                collided=collided, steps=n, ret=ep_return))
        print(f"  ep {self.episode:4d}  start {self.cur['label']}  "
              f"[{outcome:9s}]  steps {n:3d}  return {ep_return:+7.2f}"
              f"{'  SUCCESS' if success else ''}")

        if len(self.batch) >= self.args.episodes_per_update:
            self.update()
            self.batch = []

        if self.episode % self.args.save_every == 0:
            self.save(self.args.out)

        if self.episode >= self.args.episodes:
            self.save(self.args.out)
            self.write_log()
            raise KeyboardInterrupt
        self.begin_episode()

    def update(self):
        all_obs, all_act, all_ret = [], [], []
        for ep in self.batch:
            returns = compute_returns(ep['rew'], self.args.gamma)
            all_obs += ep['obs']
            all_act += ep['act']
            all_ret += list(returns)

        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32)
        act_t = torch.tensor(np.array(all_act), dtype=torch.float32)
        adv_t = torch.tensor(normalize(all_ret), dtype=torch.float32)

        dist = self.policy.distribution(obs_t)
        log_probs = dist.log_prob(act_t).sum(-1)
        loss = policy_gradient_loss(log_probs, adv_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        mean_ret = np.mean([sum(ep['rew']) for ep in self.batch])
        n_success = sum(1 for e in self.ep_log[-len(self.batch):] if e['success'])
        print(f"    -- update: loss {loss.item():+.4f}  "
              f"mean batch return {mean_ret:+.2f}  "
              f"{n_success}/{len(self.batch)} succeeded  "
              f"std(v,w) = {self.policy.log_std.exp().tolist()}")

    # -- control loop ------------------------------------------------------

    def step(self):
        if not self.running or self.pose is None or self.odom is None:
            return
        if self.odom_origin is None:
            self.odom_origin = self.odom

        x, y, yaw = self.pose
        ox, oy, oyaw = self.odom
        ox0, oy0, _ = self.odom_origin
        obs_raw = np.array([ox - ox0, oy - oy0, math.sin(oyaw), math.cos(oyaw)])
        self.normalizer.update(obs_raw)
        obs_n = self.normalizer.normalize(obs_raw).astype(np.float32)

        with torch.no_grad():
            dist = self.policy.distribution(torch.tensor(obs_n).unsqueeze(0))
            action = dist.sample().squeeze(0).numpy()
        v, w = action
        v_c = float(np.clip(v, -MAX_LIN, MAX_LIN))
        w_c = float(np.clip(w, -MAX_ANG, MAX_ANG))
        self.publish(v_c, w_c)

        # -- reward, from ground truth, never seen by the policy -----------
        dist_to_goal = math.hypot(x - self.goal['x'], y - self.goal['y'])
        progress = 0.0 if self.prev_dist is None else self.prev_dist - dist_to_goal
        self.prev_dist = dist_to_goal
        reward = self.args.progress_scale * progress - self.args.time_penalty

        clear, which = worst_clearance(x, y, self.obstacles, self.room)
        elapsed = time.time() - self.t_start
        moving = abs(v_c) > 0.01 or abs(w_c) > 0.05
        done, outcome, success, collided = False, None, False, False

        if clear < 0.0:
            reward += self.args.collision_penalty
            done, outcome, collided = True, 'collided', True
        elif not moving:
            if self.still_since is None:
                self.still_since = time.time()
            elif time.time() - self.still_since > self.args.settle:
                if in_goal(x, y, self.goal):
                    reward += self.args.success_bonus
                    done, outcome, success = True, 'stopped', True
                else:
                    done, outcome = True, 'stalled'
        else:
            self.still_since = None

        if not done and elapsed > self.args.timeout:
            done, outcome = True, 'timeout'

        self.cur['obs'].append(obs_n)
        self.cur['act'].append(action.astype(np.float32))
        self.cur['rew'].append(reward)

        if done:
            self.finish_episode(outcome, success, collided)

    # -- persistence ------------------------------------------------------

    def save(self, path):
        torch.save(dict(
            state_dict=self.policy.state_dict(),
            obs_dim=4, act_dim=2, hidden=[64, 64],
            normalizer=self.normalizer.state_dict(),
            episode=self.episode,
        ), path)
        print(f"    saved {path} (episode {self.episode})")

    def write_log(self):
        if not self.args.episode_log:
            return
        os.makedirs(os.path.dirname(self.args.episode_log) or '.', exist_ok=True)
        with open(self.args.episode_log, 'w', newline='') as f:
            wri = csv.DictWriter(f, fieldnames=list(self.ep_log[0].keys()))
            wri.writeheader()
            wri.writerows(self.ep_log)
        print(f"wrote {self.args.episode_log}")

        if self.args.curve:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
            except ImportError:
                return
            rets = [e['ret'] for e in self.ep_log]
            window = max(1, len(rets) // 20)
            smoothed = np.convolve(rets, np.ones(window) / window, mode='valid')
            os.makedirs(os.path.dirname(self.args.curve) or '.', exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.plot(rets, alpha=0.3, label='per-episode return')
            plt.plot(range(window - 1, len(rets)), smoothed,
                     label=f'moving average ({window})')
            plt.xlabel('episode')
            plt.ylabel('return')
            plt.title('REINFORCE learning curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.args.curve, dpi=140)
            print(f"wrote {self.args.curve}")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=None)
    p.add_argument('--starts', default='train')
    p.add_argument('--out', default='policy_pg.pt')
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--episodes-per-update', type=int, default=4)
    p.add_argument('--save-every', type=int, default=20)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--rate', type=float, default=10.0)
    p.add_argument('--timeout', type=float, default=20.0)
    p.add_argument('--settle', type=float, default=1.0)
    p.add_argument('--progress-scale', type=float, default=10.0)
    p.add_argument('--time-penalty', type=float, default=0.01)
    p.add_argument('--collision-penalty', type=float, default=-5.0)
    p.add_argument('--success-bonus', type=float, default=10.0)
    p.add_argument('--jitter-xy', type=float, default=0.1)
    p.add_argument('--jitter-yaw-deg', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--robot', default='burger')
    p.add_argument('--cmd-out', default='/cmd_vel')
    p.add_argument('--odom-topic', default='/odom')
    p.add_argument('--episode-log', default='logs/pg_episodes.csv')
    p.add_argument('--curve', default='figures/pg_training_curve.png')
    return p


def main():
    args = build_parser().parse_args()
    if args.config is None:
        args.config = default_config_path()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rclpy.init()
    node = PGTrainer(args, cfg['obstacles'], cfg['goal'], cfg['room'])
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
