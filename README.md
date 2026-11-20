# SuperMimic Gym

**English** | [한국어](README_ko.md)

<p align="center"><img src="docs/teaser.gif" width="720" alt="Reference motion, SuperMimic and SuperMimic-ODE tracking dance, jumps and sprint clips"></p>
<p align="center"><sub>Three LAFAN1 clips in MuJoCo. Blue: reference motion. Light blue: SuperMimic. Default G1: SuperMimic-ODE.</sub></p>

Whole-body motion tracking for the Unitree G1 with [SuperMimic](https://arxiv.org/abs/2610.28480): a differentiable world model fit on replay, and a goal-conditioned policy trained by back-propagating the tracking reward through imagined rollouts. The structure follows [legged-gym-mujoco](https://github.com/JihoonPark20/legged-gym-mujoco). Physics runs on multithreaded CPU MuJoCo; the world model, policy and critic train on CUDA, MPS or CPU, so the same code runs on Linux and macOS.

## Install

```bash
python -m pip install -e .                      # Python 3.10–3.13; install a CUDA build of PyTorch first on NVIDIA machines
python -m legged_gym.scripts.prepare_motions    # 40 LAFAN1 clips, SHA256-checked -> data/motions/lafan1
```

## Train

```bash
python -m legged_gym.scripts.train                                   # 1024 environments, 21000 steps
python -m legged_gym.scripts.train --resume --max_iterations 10000   # continue from the latest checkpoint
```

Logs and checkpoints (`model_*.pt`, ~216 MB, every 500 steps) go to `logs/g1_mimic/`. One iteration is one vectorised environment step with 0.4 learner updates. The learner device is chosen automatically (CUDA, then MPS, then CPU) or with `--rl_device`. Settings live in [g1_mimic_config.py](legged_gym/envs/g1_mimic/g1_mimic_config.py).

### SuperMimic and SuperMimic-ODE

Two task presets share the existing G1 environment, world model, critic, replay and runner. Only the actor's action parameterization differs; environment settings, network sizes and training budgets are identical.

```bash
python -m legged_gym.scripts.train --task supermimic --seed 0
python -m legged_gym.scripts.train --task supermimic-ode --seed 0
python -m legged_gym.scripts.train --task supermimic-ode --resume --max_iterations 10000
```

| Task | Actor | Logs |
|---|---|---|
| `g1_mimic` (unchanged default) | Direct action, MLP | `logs/g1_mimic/` |
| `supermimic` | Direct action, MLP | `logs/supermimic/` |
| `supermimic-ode` | Previous action + bounded action increment | `logs/supermimic-ode/` |

The ODE actor follows [ODE Policy's RL implementation](https://github.com/NTUMARS/ODE-POLICY/tree/9c505796ad0ecb811a12ab2c1ba3b8975af52a01/rl/supermimic_ode): `a_t = clip_amax(a_prev + clip_dmax(decoder(z, history)))`. It reads the previous **clean** action from the observation history, starts from zero after each environment reset, and takes one Euler step per 50 Hz control tick. The decoder predicts an increment per tick, so there is no additional `dt` multiplier or solver dependency. The gradient through previous actions is retained in imagined rollouts.

`G1MimicCfgSuperMimicODE.policy` sets `actor="ode"`, `ode_dmax=1.62` and the 29 upstream per-joint `ode_amax` bounds. These are empirical bounds in normalized action units, not physical joint limits. Their joint order and action scaling match this G1 task. Both presets use the local defaults (1024 environments, imagination batch 512), not the upstream 4096-environment GPU experiment. Compare matched seeds and budgets; see Results below.

TensorBoard adds `policy/imag_action_rate`, `policy/dJda_first`, `policy/dJda_last`, and `policy/dJda_log_first_over_last`. ODE also reports `policy/ode_step_bound_fraction` and `policy/ode_action_bound_fraction`: fractions of final increments/actions at their configured bounds, not raw decoder clipping rates.

## Play and evaluate

```bash
mjpython -m legged_gym.scripts.play --checkpoint_path checkpoints/g1_mimic_lafan1.pt --motion jumps1_subject1   # macOS viewer
python -m legged_gym.scripts.play --checkpoint_path checkpoints/g1_mimic_lafan1.pt --evaluate                   # 40-clip benchmark
python -m legged_gym.scripts.export_policy logs/g1_mimic/<run>/model_21000.pt checkpoints/my_policy.pt          # policy only
python -m pytest tests
```

On Linux the viewer runs with `python`. `--evaluate` plays every clip once from its first frame without noise or randomization and writes a JSON report beside the checkpoint. `--resume` instead of `--checkpoint_path` picks the latest training checkpoint.

Use the same task when training, resuming, playing or evaluating an ODE policy:

```bash
python -m legged_gym.scripts.play --task supermimic-ode --resume --evaluate
python -m legged_gym.scripts.export_policy logs/supermimic-ode/<run>/model_21000.pt checkpoints/my_ode_policy.pt
mjpython -m legged_gym.scripts.play --task supermimic-ode --checkpoint_path checkpoints/supermimic_ode_lafan1.pt --motion jumps1_subject1
```

Checkpoints and policy exports preserve the actor and its bounds. Loading an MLP checkpoint as ODE, or vice versa, raises an error even though their weight shapes match. Older checkpoints without an actor field remain MLP and work with `g1_mimic` or `supermimic`. Policy-only exports support playback/evaluation; training resumes require a full checkpoint. Keep `run_config.json` beside checkpoints when using nondefault environment or network settings.

Evaluation JSON includes the task, policy configuration, completion rate, full-length return and smoothness. Full-length return divides total reward by `max_step_reward * max(reference_frames, executed_steps)`, assigning zero reward to the unexecuted part after a fall while accommodating terminal bookkeeping steps. Existing normalized returns retain their original definition. Smoothness reports mean squared action differences (`action_rate`) and second differences (`action_acc`), mean absolute joint acceleration in rad/s² and jerk in rad/s³. They use consecutive samples within a clip, including pre-reset terminal velocities; summary values are weighted by the number of valid differences. No differences are taken across resets.

## Results

Both checkpoints were trained with their task defaults and seed 0 for 21000 iterations (8384 updates) on an Apple M1 Max: 3 h 27 min for SuperMimic, 3 h 32 min for SuperMimic-ODE. On the 40-clip evaluation:

| Policy | Mean normalized return | Step-weighted | Full-length return | Completed clips | Action rate | Joint jerk (rad/s³) |
|---|---:|---:|---:|---:|---:|---:|
| SuperMimic, `g1_mimic_lafan1.pt` | 0.758 | 0.780 | 0.637 | 29 / 40 | 0.0201 | 667 |
| SuperMimic-ODE, `supermimic_ode_lafan1.pt` | 0.802 | 0.821 | 0.688 | 27 / 40 | 0.0209 | 664 |

SuperMimic-ODE tracks more closely while it stays up and has the higher full-length return, but finishes two fewer clips; smoothness is about the same. Six of the early terminations of each policy are fall-and-get-up clips. These are single-seed results.

## Physics on the CPU, learning on the accelerator

Measured on an Apple M1 Max with 1024 environments and batch 512; one iteration is one control step plus 0.4 updates.

| Setup | Physics | Update | Per iteration |
|---|---:|---:|---:|
| **This repository: MuJoCo CPU, 10 threads + MPS** | 38 ms | 1.22 s | **0.53 s** |
| MuJoCo Warp on the CPU device + MPS (mjlab on macOS) | 1101 ms | 1.22 s | 1.59 s |
| MuJoCo CPU + CPU learner | 38 ms | 7.94 s | 3.21 s |

Without CUDA, MuJoCo Warp runs on its CPU device and used a single core here; native `mujoco.batch` was 29× faster with 10 threads on the same model and solver settings. The learner is dominated by wide MLPs and 32-step backpropagation, where MPS with `torch.compile` was 6.5× faster than the CPU. On NVIDIA GPUs, MuJoCo Warp would simulate on the GPU instead; this repository keeps physics on the CPU so one code path serves every platform.

## Layout

```
legged_gym/envs/g1_mimic/   task configuration and environment
legged_gym/supermimic/      world model, imagination, policy/critic, replay, runner, benchmark
legged_gym/scripts/         train, play, prepare_motions, export_policy
legged_gym/sim/             MuJoCo model construction and batched simulation
resources/                  G1 model (Unitree), LAFAN1 manifest
```

## Licenses

Code: [BSD-3](LICENSE). The SuperMimic implementation is ported from [supermimic-m5](https://github.com/WXyuany/supermimic-m5) ([MIT](licenses/supermimic_m5_LICENSE)); robot model and configuration sources are in [licenses](licenses). LAFAN1 is CC BY-NC-ND 4.0: it is downloaded rather than redistributed, and policies trained on it are for non-commercial research.

The ODE integration adapts NTUMARS/ODE-POLICY (Apache-2.0) and its patch to SuperMimic-Reproduced (MIT). Pinned sources, adaptations and license files are listed in [ode_policy_NOTICE](licenses/ode_policy_NOTICE). Please cite [ODE Policy](https://arxiv.org/abs/2611.01729) and SuperMimic when using this method.
