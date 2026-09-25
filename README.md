# UAV Visual Gap

A simulated quadrotor learns to traverse a tilted narrow aperture and stabilize
at a post-gap goal. The learning pipeline is **PPO privileged state teacher →
causal visual behavior cloning → DAgger**. Physics and the fixed low-level rate
controller run at 200 Hz; policy and depth observations run at 50 Hz.

## Current evidence

- The seed-11 reset-v2 PPO teacher scored **100/100 fixed** and **99/100
  held-out** validation episodes at its exact final checkpoint.
- The seed-11 visual DAgger-GRU pilot used **196,608 teacher labels** and
  **12,000 optimizer updates**. Its final checkpoint scored **98/100 fixed**
  and **88/100 held-out** validation episodes.
- The current learning film is V10: **21.68 s**, 542 frames at 25 fps. It shows
  four training milestones and a held-out flight. The files are local generated
  artifacts and are excluded from Git.

These are single-seed validation results. The formal multi-seed comparison,
final test set, and real-flight/VIO validation are incomplete. See
[results and evidence boundaries](docs/RESULTS.md) before quoting the metrics.

## System boundary

The visual student receives depth, simulated noisy/delayed own-state, a relative
goal vector, and its previous command. It outputs collective thrust and body
rates. A fixed rate controller maps these to motor-level actuation. The student
does not receive aperture geometry or privileged teacher state. This is a
simulation study; it does not claim learned motor RPM, onboard state estimation,
VIO, or real-drone transfer. See [project boundaries](docs/PROJECT_BOUNDARIES.md).

## Reproduce the code environment

The supported reference is Linux x86_64, Python 3.8, PyTorch 2.1/CUDA 12.1,
NVIDIA GPU, and the licensed Isaac Gym Preview 4 distribution. Isaac Gym cannot
be redistributed. Obtain it under its license and install it under
`third_party/isaacgym/isaacgym/python`. On a fresh checkout without a local
Aerial Gym copy, fetch the pinned source with the helper. It refuses to replace
an existing ignored copy; that copy has no Git metadata and its revision must be
independently verified or replaced from a fresh checkout before claiming exact
source reproduction. Then:

```bash
./scripts/prepare_aerial_gym.sh
conda create --name uav_gap --file configs/conda-linux-64.explicit.lock
conda activate uav_gap
./scripts/install_runtime.sh
scripts/run.sh scripts/check_runtime.py --write-inventory
scripts/run.sh -m pytest -q
```

The explicit Conda lock fixes the tested Linux package layer. The pip layer is
pinned in `requirements-runtime.txt`; local editable installs are performed by
the installer. `environment.yml` is a readable baseline for other Conda
resolvers, not a bitwise lock. Full setup and evaluation steps are in
[REPRODUCING.md](docs/REPRODUCING.md).

A simulator smoke test creates a CUDA context. Run it only after confirming the
GPU is idle and the required free-memory threshold is available:

```bash
scripts/run.sh scripts/smoke_aerialgym.py --num-envs 2 --steps 100
```

## Review map

- `src/uav_gap/task.py`: task transitions, collisions, termination and reward.
- `src/uav_gap/`: task, observations, controller adapters and policy interfaces.
- `scripts/train_teacher.py`, `scripts/train_student.py`: PPO and recurrent
  student training entrypoints.
- `scripts/run_visual_rounds.py`: data collection, DAgger rounds, training and
  validation orchestration.
- `configs/`: frozen task, training, scene and protocol configurations.
- `tests/`: CPU unit and boundary checks.
- `docs/FORMAL_PROTOCOL.md`, `docs/ALGORITHM_NOTES.md`,
  `docs/VISUAL_TRAINING.md`: method and study protocol.
- `docs/REPLAY.md`: faithful offline rendering and the current film manifest.
- `docs/EXECUTION_PLAN.md`: accepted study scope, completed work and remaining
  acceptance gates.

Generated results, checkpoints, datasets, licensed dependencies, and the demo
video are intentionally ignored by Git. The current Git history has not been
sanitized for public release; review `docs/REPRODUCING.md` before publishing or
transferring repository history.
