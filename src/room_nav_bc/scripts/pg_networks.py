#!/usr/bin/env python3
"""
Policy network for REINFORCE (vanilla policy gradient) - Part 3 of the
assignment.

Observation: the SAME 4-dim odometry-only features used elsewhere in this
project (dx, dy, sin(yaw), cos(yaw), relative to the episode's own start -
see bc_dataset_room.py's docstring for why that's the only reference
available). No camera here, deliberately - see pg_train_room.py's
docstring for why a vision policy isn't tractable to train online with
plain REINFORCE in a reasonable amount of wall-clock time.

Action: (v, w), modelled as a diagonal Gaussian. The mean is a function of
the observation (a small MLP); the standard deviation is NOT - it's a
single learned vector, shared across all states, which is the standard
simplification for a first policy-gradient implementation (state-
dependent std is a reasonable later extension, not needed here).
"""

import math

import torch
import torch.nn as nn


class GaussianPolicy(nn.Module):

    def __init__(self, obs_dim=4, act_dim=2, hidden=(64, 64),
                init_std=(0.05, 0.5)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.mean_net = nn.Sequential(*layers)
        # One log_std per action dimension - v and w have very different
        # ranges (+-0.22 m/s vs +-2.84 rad/s), so they need different
        # starting exploration noise, not a single shared scalar.
        self.log_std = nn.Parameter(
            torch.tensor([math.log(s) for s in init_std], dtype=torch.float32))

    def mean(self, obs):
        """obs: (B, obs_dim) -> (B, act_dim) mean action (pre-clip)."""
        return self.mean_net(obs)

    def distribution(self, obs):
        """-> torch.distributions.Normal over actions at these observations."""
        mean = self.mean(obs)
        std = self.log_std.exp().clamp(min=1e-3)
        return torch.distributions.Normal(mean, std)
