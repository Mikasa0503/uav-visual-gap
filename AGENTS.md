# Project instructions

- Keep changes inside this repository and use its dedicated Python environment.
  `scripts/run.sh` discovers `UAV_GAP_ENV` or the active `CONDA_PREFIX`.
- Keep the accepted study scope and remaining gates in `docs/EXECUTION_PLAN.md`.
- Keep `docs/STATUS.md` suitable for review: no private machine paths, process
  IDs, credentials, or user-specific operational history. Put live job handles
  and detailed server operations in ignored `local/` files.
- Treat `third_party/`, generated assets, checkpoints, datasets, logs, runs and
  media as local data. Do not publish licensed files, trained weights, or raw
  experiment data without an explicit review.
- Never stop another user's GPU process. GPU entrypoints must check availability
  before creating a CUDA context. CPU tests do not authorize a GPU job.
- Preserve the distinction between the privileged PPO teacher and the visual
  student. Do not present a teacher, scripted trajectory, renderer, or runtime
  smoke as visual-policy evidence.
- The student has no aperture pose, aperture geometry, crossing flag, or teacher
  state at inference. The simulated noisy/delayed own-state input is not VIO.
- Save genuine learning checkpoints from the first training run. Record fixed
  evaluation scenes, held-out conditions, seeds, commands and hashes.
- Keep upstream third-party code unmodified; put adapters in `src/uav_gap/` and
  record dependency provenance in `docs/THIRD_PARTY.md`.
- Before claiming reproducibility, state whether the result is code-test,
  infrastructure-smoke, validation, formal-test, or real-flight evidence.
