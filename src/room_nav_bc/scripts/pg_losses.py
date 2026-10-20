#!/usr/bin/env python3
"""
REINFORCE math - Part 3 of the assignment. Pure functions, no ROS, so
they're unit-testable without the simulator (see the bottom of this file
for a self-check: `python3 pg_losses.py`).
"""

import numpy as np


def compute_returns(rewards, gamma):
    """[r_0, r_1, ..., r_{T-1}] -> [G_0, G_1, ..., G_{T-1}], where
    G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...

    Computed back-to-front: G_t = r_t + gamma * G_{t+1}, with G_T = 0.
    """
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def normalize(x, eps=1e-8):
    """Zero mean, unit std - the simple REINFORCE baseline. Subtracting
    the batch mean return is what keeps the gradient estimate from being
    dominated by the overall scale of the reward (e.g. whether most
    episodes score around +5 or around -5 doesn't matter, only which
    episodes did better or worse than the batch average)."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


def policy_gradient_loss(log_probs, advantages):
    """log_probs, advantages: torch tensors, shape (N,).
    -> scalar loss to MINIMISE (i.e. -1 x the REINFORCE objective).

    REINFORCE: maximise E[log pi(a|s) * A], where A is the (normalised)
    return. An action that did better than average (A > 0) gets its log-
    probability pushed up; worse than average (A < 0) gets pushed down.
    """
    return -(log_probs * advantages).mean()


if __name__ == '__main__':
    # Offline self-check - no ROS/Gazebo needed.
    rewards = [1.0, 1.0, 1.0]
    returns = compute_returns(rewards, gamma=0.5)
    expected = np.array([1 + 0.5 * 1 + 0.25 * 1, 1 + 0.5 * 1, 1.0], np.float32)
    assert np.allclose(returns, expected), (returns, expected)
    print('compute_returns: OK', returns)

    import torch
    lp = torch.tensor([0.1, -0.2, 0.3])
    adv = torch.tensor([1.0, -1.0, 0.5])
    loss = policy_gradient_loss(lp, adv)
    expected_loss = -((0.1 * 1.0 + -0.2 * -1.0 + 0.3 * 0.5) / 3)
    assert abs(loss.item() - expected_loss) < 1e-6
    print('policy_gradient_loss: OK', loss.item())

    n = normalize([1.0, 2.0, 3.0])
    assert abs(n.mean()) < 1e-6 and abs(n.std() - 1.0) < 1e-5
    print('normalize: OK', n)
    print('\nall checks passed')
