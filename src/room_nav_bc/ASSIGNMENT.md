# room_nav_bc — assignment map

One room, one TurtleBot3 (camera + odometry), one task: get from a start
pose to the goal, choosing either side of the central island. Four parts,
each a different way of getting a policy that does that. Every part
grades the same way — success/collision judged from ground truth, which
the policy itself never sees — so results are comparable across parts.

```
worlds/, config/, launch/        the environment - shared by every part
scripts/record_demo_room.py      teleop recorder (Part 1, human track)
scripts/expert_demo_room.py      scripted oracle (Part 1 expert track, Part 2's relabeller)
scripts/verify_expert_room.py    offline collision check for the oracle - run this after
                                  touching expert_demo_room.py or the room layout, before
                                  trusting any episode it records
scripts/show_starts_room.py      visualise start poses in Gazebo
scripts/analyse_rollouts_room.py averaging-vs-OOD diagnosis on a logged rollout
scripts/plot_policy_room.py      plot a rollout log's turn-rate distribution
```

## Part 1 — Behavioural Cloning (offline, supervised)

```
scripts/bc_dataset_room.py       episodes (camera+odometry+action) -> training set
scripts/bc_train_room.py         CNN+odometry policy, regression (MSE/L1)
scripts/bc_policy_node_room.py   run it closed-loop, grade it
```
Get demonstrations first, either by driving (`record_demo_room.py`) or
from the scripted oracle (`expert_demo_room.py`), then build → train → test.

## Part 2 — DAgger (online relabelling)

```
scripts/dagger_room.py           current policy drives; you take over (SPACE) when it's
                                  wrong; every step is saved with the command actually
                                  executed and a human/policy flag
scripts/bc_dataset_room.py --human-only   aggregate your corrections into a dataset
scripts/bc_train_room.py         retrain on demos + corrections
```
Repeat: run a round, retrain, run the next round starting from the new
policy. The intervention rate (how much of each round you had to drive)
should fall if it's working.

## Part 3 — Policy Gradient (online, REINFORCE)

```
scripts/pg_networks.py           GaussianPolicy - the action distribution
scripts/pg_losses.py             compute_returns, normalize, policy_gradient_loss
scripts/pg_train_room.py         online training loop: roll out, compute returns,
                                  one gradient step, repeat
scripts/pg_policy_node_room.py   run the trained policy deterministically, grade it
```
No demonstrations at all here — the policy starts random and learns from
a reward signal computed from ground truth (progress toward the goal,
a collision penalty, a success bonus). Observation is odometry only
(`dx, dy, sin(yaw), cos(yaw)`), not the camera — see `pg_train_room.py`'s
docstring for why a vision policy isn't tractable to train online with
plain REINFORCE in reasonable wall-clock time. `pg_losses.py` is unit-
testable without Gazebo: `python3 pg_losses.py`.

## Part 4 — Flow Matching (offline, generative)

Not yet added. Will reuse Part 1's dataset/CNN encoder, replacing the
regression head with a flow-matching one (`FlowMatchingSchedule`,
`flow_matching_loss` — same shape as the CS224R reference this project
mirrors), to show the multimodal-averaging failure some BC runs exhibit
at the island resolved without DAgger.

## Shared grading convention

Every test/eval node (`bc_policy_node_room.py`, `dagger_room.py`,
`pg_policy_node_room.py`) uses the same success definition — stopped,
inside the goal region — and the same collision check,
`worst_clearance()`, against every obstacle and wall, not just the
island. All three also jitter the start pose a little each trial by
default, so nothing is ever tested from the exact recorded/config pose.
