# safe_rl

A modular PyTorch pipeline for training reinforcement learning agents with **pluggable runtime safety shielding**.

`safe_rl` provides the complete training loop end to end — environment construction, data collection, replay buffering, agent optimization, evaluation, plotting, checkpointing, and experiment tracking — with a plugin-style architecture at every layer: swap the RL algorithm, the dynamics model, the planner, the environment, or the safety shield independently of the rest of the stack.

**What's inside:**

- **Model-free RL** — DDPG, TD3, SAC (and an experimental MADDPG), with Ornstein–Uhlenbeck / Gaussian exploration and noise-annealing schedules.
- **Model-based RL** — dynamics learning (deterministic NN, probabilistic Gaussian NN, bootstrap ensembles, exact GPs via GPyTorch, structured control-affine models) with an analytic nominal prior plus learned residual, and planning by random shooting or CEM-MPC over the learned model.
- **Safety shielding** — a shield layer that filters any agent's actions at runtime: a learned-CBF QP filter, an analytic backup-policy shield, and an RL-trained backup shield that *learns* to enlarge the certified safe region.
- **Composite agents** — factories assemble the above recursively, so a task policy + learned dynamics + shield behave as one agent to the trainer.
- **Environment plugin system** — drop in a folder of convention-named files (`<name>_configs.py`, `<name>_safety.py`, `<name>_dynamics.py`, …) to add an environment with its own physics, safe sets, observation processing, reward, and plots. Ships with a customized Gym pendulum, a Safety-Gym Point robot, and two custom mass-spring-damper systems.
- **Experiment infrastructure** — layered configuration with per-env/per-agent overrides, safe environment resets, safety-violation accounting, Weights & Biases + CSV + matplotlib logging, video recording, and checkpoint save/resume with partial model loading.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Pipeline architecture](#pipeline-architecture)
- [Agents](#agents)
- [Safety shielding](#safety-shielding)
- [Repository structure](#repository-structure)
- [Configuration system](#configuration-system)
- [Environments](#environments)
- [Logging and results](#logging-and-results)
- [Adding a new environment](#adding-a-new-environment)
- [Known caveats](#known-caveats)
- [Research using this framework](#research-using-this-framework)

---

## Installation

```bash
git clone <this-repo>
cd safe_rl
pip install -r requirements.txt
```

The pinned stack is from the 2021 era: `torch==1.8.1`, `gym==0.15.7`, `mujoco_py==2.0.2.7`, `safety-gym`, `gpytorch==1.4.0`, `cvxopt`. The `point` environment additionally requires a working [MuJoCo 2.0](https://www.roboti.us/) + [safety-gym](https://github.com/openai/safety-gym) installation; the `pendulum` and `misc_env` environments run without MuJoCo.

Some packages used by the code are **not** listed in `requirements.txt` and must be installed manually:

```bash
pip install wandb cvxpy torchdiffeq
```

> **Note:** `requirements.txt` pins numpy twice (`1.25.1` and `1.17.5`); with `torch==1.8.1` / `gym==0.15.7` you want the older pin (Python 3.7/3.8 era).

## Quick start

The environment and agent are selected at the bottom of [main.py](main.py):

```python
setup = make_setup(env_nickname='pendulum',   # pendulum | point | cbf_test | multi_dashpot
                   agent='rlbus')             # see Agents table below
```

Then:

```bash
python main.py
```

Behavior is controlled by [config.py](config.py). Two flags matter most on a first run:

- `debugging_mode = True` (default) — runs locally with **no** wandb logging and **nothing saved to disk**. Set to `False` for a real experiment.
- `evaluation_mode = False` — set to `True` (with `load_models = True` and a `load_run_name`) to evaluate a saved run instead of training.

Random seeding is centralized in `utils/seed.py` (`set_seed()` seeds `random`, NumPy's `default_rng`, and torch CPU/CUDA). The whole framework runs in `torch.float64`.

## Pipeline architecture

One `agent` string drives two factories: [agents/agent_factory.py](agents/agent_factory.py) builds the agent (recursively, for composite agents) and [trainers/trainer_factory.py](trainers/trainer_factory.py) picks the matching trainer. `BaseTrainer` then wires the full experiment: config merge → logger/wandb → train & eval envs (with action-scaling and optional video wrappers) → observation processor → agent → safe sets → `Sampler`. The loop:

```
for itr in train_iter:
    trainer._train(itr)        # agent-specific: collect data + optimize
    checkpoint if due          # torch.save(agent.get_params()) (+ replay buffers)
    evaluate if due            # n_episodes_evaluation rollouts, no exploration
```

The pieces each layer plugs into:

- **Sampler** ([samplers/sampler.py](samplers/sampler.py)) — synchronous data collection with initial-data and random-warmup phases, safe resets, per-episode return and **safety-violation** logging, optional noise injection when buffering, and dict-observation support.
- **Buffers** ([buffers/](buffers/)) — a numpy circular `ReplayBuffer` plus `BufferQueue`, a multi-key buffer for heterogeneous data (e.g. safe/unsafe/derivative samples) with coupled-index sampling. Agents can hold several buffers at once.
- **Trainers** — `MFTrainer` (collect + N gradient steps), `MBTrainer` (periodic dynamics fits, controller switch from random to CEM partway through training), `SFTrainer` (two buffers, filter pretraining, interleaved model-free / dynamics / filter updates on separate schedules), `BUSTrainer` / `RLBUSTrainer` (shield-specific loops with contour plotting and backup-policy training).
- **Networks** ([networks/](networks/)) — configurable `MLPNetwork`, `GaussianMLP`, SAC-style `SquashedGaussianMLP`, `MultiInputMLP` (late action concatenation for critics), and a GPyTorch `GP`.
- **Checkpointing** — full model/optimizer state dicts every `train_iter / num_save_sessions` iterations, optional buffer snapshots, resume/benchmark/evaluation modes, and partial restore via `custom_load_list`.

## Agents

| Nickname | Agent class | Trainer | What it is |
|---|---|---|---|
| `ddpg` | `DDPGAgent` | `MFTrainer` | DDPG with OU-noise exploration |
| `td3` | `TD3Agent` | `MFTrainer` | TD3 (twin critics, delayed policy updates, target smoothing) |
| `sac` | `SACAgent` | `MFTrainer` | SAC (squashed-Gaussian policy, twin critics, entropy bonus) |
| `maddpg` | `MADDPGAgent` | `MFTrainer` | Multi-agent DDPG with a centralized critic (untested) |
| `mb` | `MBAgent` | `MBTrainer` | Model-based RL: learned dynamics + planning (random controller → CEM-MPC) |
| `sf` | `SFAgent` | `SFTrainer` | Shielded RL: DDPG + learned dynamics + *learned* CBF filter |
| `bus` | `BUS` | `BUSTrainer` | Shielded analytic policy: backup-policy shield, no learning |
| `rlbus` | `RLBUS` | `RLBUSTrainer` | Shielded RL with a *learned* backup policy (SAC task policy + SAC backup) |

Composite agents are assembled recursively by the factory: `sf` = `ddpg` + `mb` + `cbf_filter`; `rlbus` = `rl_backup_shield` (which itself wraps a `sac` backup agent) + a `sac` task policy. Which sub-agents compose them is itself configuration (`sf_params.mf`, `rlbus_params.shield_agent`, `rlbus_params.desired_policy_agent`, …), so shields can wrap any compatible algorithm.

Planning for `mb` lives in [controller/](controller/): `RandomController`, `RandomShootController` (random-shooting MPC), and `CEMController` (cross-entropy-method MPC with warm starts) — all plan by "dreaming" particle rollouts through the learned dynamics (`BaseController.dream`), with ensemble models handled transparently.

## Safety shielding

Shields implement one interface — `shield(obs, ac, ...)`: take the agent's desired action, return a safe one ([shields/base_shield.py](shields/base_shield.py)). The trainer and sampler don't know or care whether a shield is present; it is just part of the agent's `step`. Three shields are included:

- **`CBFSheild`** ([shields/cbf_shield.py](shields/cbf_shield.py)) — a *learned* barrier: an MLP `h(x)` trained from sampled safe/unsafe states and derivative data, executed as a minimum-intervention QP whose constraint is robustified against the learned dynamics' uncertainty (`k_delta`-scaled worst-case margin from the affine-in-action Gaussian model). Used by `sf`.
- **`BackupShield`** ([shields/backup_shield.py](shields/backup_shield.py)) — an *analytic* shield: forward-integrates a set of hand-designed backup policies over a finite horizon (`torchdiffeq.odeint`), composes the safe-set and backup-set barrier values along those rollouts through soft-min/soft-max into a single implicit barrier, and solves a min-intervention CBF-QP under input box constraints, smoothly blending to the best backup policy near the boundary. Used by `bus`.
- **`RLBackupShield` / `RLBackupShieldExplorer`** ([shields/rl_backup_shield.py](shields/rl_backup_shield.py), [rl_backup_shield_explorer.py](shields/rl_backup_shield_explorer.py)) — extends `BackupShield` with one extra backup policy that is *trained with RL* (SAC): its reward is derived from the barrier values of its own rollouts, so improving return directly enlarges the certified safe region; a raised-cosine "melt law" hands control back to the analytic backups near their backup sets, preserving the guarantees throughout training. The explorer variant seeds rollouts with the task policy for richer training data. Used by `rlbus`.

Backing this up is a **safe-set abstraction** ([utils/safe_set.py](utils/safe_set.py)): `SafeSet` → `SafeSetFromCriteria` → `SafeSetFromBarrierFunction` provide the desired safe set (e.g. a p-norm geofence), the actual certified set, batch sampling by criteria (safe / unsafe / near-boundary), and safe environment resets — each environment supplies its own barriers. QP/LP machinery (CBF-QPs, Lie derivatives, box constraints, cvxpy/cvxopt solvers) lives in [utils/cbf_utils.py](utils/cbf_utils.py) and [utils/optim.py](utils/optim.py); the conservative soft-min/soft-max operators are in [utils/torch_utils.py](utils/torch_utils.py).

## Repository structure

```
safe_rl/
├── main.py                  # entry point: pick env + agent, train/evaluate
├── config.py                # all defaults & per-agent hyperparameters (Config class)
├── agents/
│   ├── agent_factory.py     # nickname → agent, builds composite agents recursively
│   ├── base_agent.py        # act/step/optimize API, buffer & model plumbing
│   ├── model_free/          # ddpg, td3, sac, maddpg
│   └── model_based/         # mb (dyn + planner), bus, rlbus, cbf_test (LQR baseline)
├── trainers/
│   ├── trainer_factory.py   # nickname → trainer (ddpg/td3/sac share MFTrainer)
│   ├── base_trainer.py      # env/agent/sampler setup, train loop, checkpoints, eval
│   └── {mf,mb,sf,bus,rlbus}_trainer.py
├── samplers/sampler.py      # data collection, episodes, safety-violation accounting
├── buffers/                 # ReplayBuffer, BufferQueue (multi-key buffer)
├── controller/              # random / random-shooting / CEM planners for mb
├── dynamics/                # nominal + learned (NN/Gaussian/ensemble/GP), affine-in-action
├── networks/                # MLP, GaussianMLP, SquashedGaussianMLP, MultiInputMLP, GP
├── shields/                 # CBFSheild, BackupShield, RLBackupShield(+Explorer)
├── explorations/            # OU noise, random noise
├── distributions/           # torch distribution extensions
├── envs_utils/
│   ├── get_env_info.py      # nickname registry
│   ├── gym/pendulum/        # configs, RK45 wrapper, dynamics, safe sets, backup
│   │                        #   sets/policies, obs-proc, plotters, reward
│   ├── safety_gym/point/    # Point robot: engine config, dynamics, safe sets, backup shield
│   └── misc_env/            # cbf_test (mass-spring-damper), multi_dashpot (custom gym.Envs)
├── utils/
│   ├── make_env.py          # env construction + wrappers (action scaling, video Monitor)
│   ├── safe_set.py          # safe-set class hierarchy + factory
│   ├── cbf_utils.py         # CBF-QPs, min-intervention QP, Lie derivatives
│   ├── torch_utils.py       # soft-min/soft-max, jacobians, polyak, freeze/copy
│   ├── optim.py             # QP/LP solvers (cvxpy + cvxopt)
│   ├── mpc_utils.py         # LQR / condensed finite-horizon MPC matrices
│   └── ...
└── logger/logger.py         # console/CSV/matplotlib/wandb logging (see logger/logger_readme.md)
```

## Configuration system

Configuration is layered, all merged at trainer startup:

1. **Global defaults** — the `Config` class in [config.py](config.py): run modes, trainer/sampler settings, and per-agent hyperparameter blocks (`_get_<agent>_params`, e.g. `_get_sac_params`, `_get_rlbus_params`).
2. **Per-environment overrides** — each env ships `envs_utils/<collection>/<nickname>/<nickname>_configs.py` exposing:
   - `config`: a dict of overrides (an `init` block for top-level `Config` attributes plus `<agent>_params` blocks) applied via deep-merge;
   - `env_config`: physics and env-specific parameters (masses, torque limits, timestep, safe-set bounds, integrator choice; for safety-gym also the full `Engine` config).
3. **Checkpoint configs** — runs can snapshot and later reload their config (`load_config_path`, `envs_utils/save_config.py`).

Observation processors and plotters are also swappable per env *and per agent* via `obs_proc_index_dict` / `plotter_dict` in the env's `_obs_proc.py` / `_plotter.py`.

## Environments

Registered in [envs_utils/get_env_info.py](envs_utils/get_env_info.py):

| Nickname | Env | Collection | Notes |
|---|---|---|---|
| `pendulum` | `Pendulum-v0` (customized) | `gym` | Custom mass/length/torque limits, LSODA/RK45 integration wrapper, analytic backup sets & policies at multiple equilibria — the richest example env |
| `point` | Safety-Gym `Point` robot | `safety_gym` | Goal task with hazards, custom engine config, lidar/quaternion obs processing, safe sets from data or criteria |
| `cbf_test` | mass-spring-damper | `misc_env` | Custom `gym.Env`; LQR + integral-action baseline agent; safe sets from criteria/propagation |
| `multi_dashpot` | multi-mass dashpot | `misc_env` | Vectorized extension of `cbf_test` |

## Logging and results

The module-level logger ([logger/logger.py](logger/logger.py), documented in [logger/logger_readme.md](logger/logger_readme.md)) provides colorized console tables, per-category CSVs (`episode`, `iteration`, `evaluation_episode`, …), matplotlib figure queues (including barrier contour plots for the shielded agents), and [Weights & Biases](https://wandb.ai) integration. With `debugging_mode = False`, runs land in `results/<wandb_project_name>/wandb/<run>/` containing `tabular/*.csv` and `plots/`. Video is recorded by the vendored `Monitor` wrapper during evaluation. Checkpoints save the full model/optimizer state dict and, optionally, replay buffers, and support partial loading via `custom_load_list`.

## Adding a new environment

Everything is resolved by naming convention — create `envs_utils/<collection>/<name>/` with the files your agent needs, register the nickname in `get_env_info.py`, and the factories pick them up:

| File | Provides | Needed for |
|---|---|---|
| `<name>_configs.py` | `config` overrides + `env_config` | always |
| `<name>_env.py` | `<Name>Env(gym.Env)` | `misc_env` collection only |
| `<name>_safety.py` | `get_safe_set()` → safe-set class | shields, safety metrics |
| `<name>_backup_shield.py` | backup sets, backup policies, desired policy | `bus` / `rlbus` |
| `<name>_dynamics.py` | `<Name>TorchDyn` (analytic f, g) and/or `<Name>NominalDynamics` | shields / `mb` / `sf` |
| `<name>_obs_proc.py` | `<Name>ObsProc` (+ per-agent `obs_proc_index_dict`) | optional |
| `<name>_plotter.py`, `<name>_reward_gen.py` | custom plots, reward function | optional |

## Known caveats

Research code, shared as-is:

- `requirements.txt` has a conflicting duplicate numpy pin, and `wandb`, `cvxpy`, `torchdiffeq` are used but unlisted.
- `MADDPGAgent` is marked untested; `MPCController` is a stub; `utils/wrappers/safety_gym_wrapper.py` (custom safety-gym dynamics injection) is unfinished.
- A couple of base classes carry a historical typo (`BaseSheild`, `CBFSheild`) — grep accordingly.
- No license file is currently included; contact the author about reuse.

## Research using this framework

This pipeline was used to develop and evaluate work on safe exploration with backup control barrier functions:

- P. Rabiee and J. B. Hoagg, *"Safe Exploration in Reinforcement Learning: Training Backup Control Barrier Functions with Zero Training-Time Safety Violations"*, L4DC 2025 ([arXiv:2312.07828](https://arxiv.org/abs/2312.07828), [PMLR v283](https://proceedings.mlr.press/v283/rabiee25a.html)) — the `rlbus` agent.
- P. Rabiee and J. B. Hoagg, *"Soft-minimum and soft-maximum barrier functions for safety with actuation constraints"*, Automatica, vol. 171, 2024 ([arXiv:2305.10620](https://arxiv.org/abs/2305.10620)) — the `bus` agent's shielding method.

If you use this code in your research, please cite:

```bibtex
@inproceedings{rabiee2025rlbus,
  title     = {Safe Exploration in Reinforcement Learning: Training Backup Control Barrier Functions with Zero Training-Time Safety Violations},
  author    = {Rabiee, Pedram and Hoagg, Jesse B.},
  booktitle = {Proceedings of the 7th Annual Learning for Dynamics \& Control Conference (L4DC)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {283},
  year      = {2025},
}

@article{rabiee2024softmin,
  title   = {Soft-minimum and soft-maximum barrier functions for safety with actuation constraints},
  author  = {Rabiee, Pedram and Hoagg, Jesse B.},
  journal = {Automatica},
  volume  = {171},
  pages   = {111921},
  year    = {2024},
}
```
