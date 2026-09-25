# Public project status

**Last reviewed:** 2026-09-25

## Current state

- Project source contains the PPO teacher, visual BC/DAgger-GRU pipeline,
  evaluation tools and offline replay workflow.
- Seed-11 teacher and visual-student pilot results are recorded in
  [RESULTS.md](RESULTS.md); they are validation evidence, not formal multi-seed
  or final-test results.
- The current presentation edit is the 21.68-second V10 film. Its video and
  source experiment artifacts are excluded from Git.
- Formal multi-seed comparisons, final-test evaluation and real-flight/VIO
  validation are incomplete. Scope and next acceptance gates are in
  [EXECUTION_PLAN.md](EXECUTION_PLAN.md).

## Verification

Post-cleanup CPU test and runtime-check outcomes are pending and will be recorded
here after they run. No GPU experiment was started as part of this documentation
and environment cleanup. A runtime smoke test is infrastructure evidence, not a
learned-flight result.

## Evidence storage

Checkpoints, raw traces, datasets, generated assets and videos remain local and
are ignored by Git. Review `docs/RESULTS.md` and `docs/REPRODUCING.md` for exact
boundaries and the evidence needed to independently recalculate a score. Live
job handles and detailed host operations belong in ignored `local/` files, not
in this public status page.
