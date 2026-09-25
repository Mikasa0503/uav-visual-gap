# Visual imitation protocol

The visual policy consumes only a 64×64 rendered depth image, its simulated
noisy own-state estimate, a goal vector, and the last known applied command.
No aperture pose, dimensions, crossing flags, or teacher state enters its input.
The baseline sensor transport delay is one policy frame (20ms). Only measured
depth and the first 12 proprioception components are delayed; the known previous
command is current. At reset, history is initialized from the first observation.
This is simulated state estimation, **not VIO**.

## Collection and supervision

- `scripts/collect_demonstrations.py --teacher ... --name ...` collects genuine
  teacher-controlled physics rollouts for BC. A `--student` checkpoint switches
  execution to pure student control (DAgger beta=0), while the teacher labels the
  visited states. All completed outcomes are retained, including failures.
- Each NPZ contains pre-action depth, proprioception, teacher action and actual
  applied action, aligned at 50Hz. Its JSON records initial geometry, start pose,
  outcome and length. Collection uses training randomization, never evaluation
  manifests. The manifest records checkpoint/source hashes and unique labels.
- `scripts/train_student.py --datasets ... --kind gru|stack4 --name ...` trains
  with action MSE, Adam 3e-4 and norm clipping 1. Target windows contain at most 64
  frames. The whole preceding episode is run without gradient to recover causal
  memory; only the target window receives gradients. Padding is masked, and
  reset occurs at the true episode start. No window crosses an episode boundary.
- DAgger rounds pass all prior data directories plus the new student rollout
  directory and continue the previous student checkpoint. Initial weights and
  learning snapshots are saved before/through training, with actual updates and
  label presentations. A new deterministic sampler starts at each round; this
  is not a claim of bitwise mid-minibatch resume.
- `scripts/evaluate_student.py` loads only the student checkpoint, uses fixed
  validation/test scenarios, and records complete closed-loop traces on request.
  Latency is reported separately as GPU batch latency and a warmed batch-one
  benchmark on a real sensor sample (100 iterations, excluding rendering and
  transfers). Benchmarking preserves the evaluation sensor-noise RNG stream.
  Success and recovery durations are success-conditioned; failures and
  timeouts are retained in the denominator.

## Evidence boundary and remaining protocol work

The visual pipeline has passed a small real-physics integration run. A tiny
integration checkpoint is not a trained final visual policy.
Collect final task demonstrations only after the teacher meets its validation
gate. Formal BC/DAgger/stack comparisons must enforce identical teacher-label
and optimizer-update budgets across seeds. `collect_demonstrations.py
--label-budget N` now queries exactly N live retained states, ends at that budget,
and stores unfinished tails with `budget_truncated=true`. These tails are valid
supervision, but are not counted as successful or failed complete episodes.
An odd 17-label / 2-environment integration check verified this boundary. All
prior datasets are supplied to each DAgger round. `run_visual_rounds.py` now
orchestrates equal-budget rounds for bc_gru, dagger_gru and dagger_stack4, sharing
identical round-zero teacher data. Later BC rounds remain teacher-controlled;
later DAgger rounds execute the preceding student. The runner verifies teacher,
dataset, checkpoint and evaluation provenance at each stage. Completed stages can
be resumed with the identical request; incomplete output directories are preserved
for diagnosis and are never silently overwritten or counted as completed work.

`configs/visual_pilot.json` specifies four rounds of 16,384 new labels and 1,000
updates each (65,536 labels and 4,000 updates total per group). This is exploratory
budget selection, not the final preregistered experiment. Each round validates
100 show and 100 held-out cases; the test set stays untouched. For example, after
an eligible reset-v2 teacher checkpoint and exact matching validation exist:

```bash
scripts/run.sh scripts/run_visual_rounds.py --name visual_pilot_v1 \
  --group dagger_gru --seed 11 \
  --teacher checkpoints/TEACHER_RUN/final.pth \
  --teacher-validation runs/evaluation/TEACHER_VALIDATION/summary.json
```

Use `--dry-run` to inspect commands, or `--resume` to reuse only validated completed
stages. The three-seed formal experiment and frozen final budgets remain pending.

Before any stack-policy training, its feed-forward head was widened to 384→128→64
to match the GRU model's parameter count within 1%. The GRU architecture is
unchanged. Stack architecture revision is `stack4_capacitymatched_v2`; no trained
old-stack checkpoints are used. Evaluation metadata records parameter counts.

Stress execution is specified in `configs/stress_v1.yaml` and implemented in the
evaluation entrypoints. Conditions remain independent. The 20% mass condition
changes the URDF's physical mass/inertia while keeping policy thrust conversion,
motor initialization and rate-control calibration nominal. Real-physics checks
verified unchanged nominal mass, the expected loss of hover, and a one-shot
world impulse. Three missed depth acquisitions hold the last received depth;
their eventual visibility follows the normal observation delay. Formal success
measurements on all conditions still require mature visual checkpoints.

## Pilot extensions and explicit learning-rate schedule

The original four-round pilot is immutable. `visual_pilot_extension.json` adds
two rounds while retaining its checkpoint and complete dataset aggregate.
`visual_pilot_lowrate.json` adds two more from round 5, preserving all six earlier
datasets, with 1e-4 learning rate per round. Its total budget is 131,072 unique
labels / 8,000 updates. These are exploratory runs, not formal comparison data.
`parent_experiment` links the plotting history; the plot includes regressions.

`train_student.py --learning-rate` changes the optimizer rate AFTER restoring
Adam state, records the effective value, and retains optimizer moments. If this
argument is absent, a resumed rate is preserved (fresh default 3e-4). The runner
supports one explicit rate per round and checks the recorded effective rate.
CPU continuation integration verified the override and cumulative update count.

Current first-six-round validation history (show/held-out, out of 100) is
0/0, 8/3, 0/0, 78/54, 88/75, 75/67. Success targets remain unchanged. The model
uses depth in offline diagnostics; strong fit alone is not closed-loop success.

The low-rate extension completed with 94/77 then 92/73 show/held-out successes.
`visual_pilot_moredata.json` continues from round 7 with all eight datasets,
two more rounds of 32,768 labels / 2,000 updates at 1e-4 each, for expected totals
196,608 labels / 12,000 updates. Formal budgets remain pending; no target changed.

## Frozen formal recipe

The expanded-data pilot finished at 98/100 show and 88/100 held-out. Formal
configuration `visual_formal_v1.json` now reproduces its ten-round label/update
and learning-rate schedule from scratch for all three groups and seeds. It
freezes relevant sources/assets before every subprocess. Parameter matching is
0.26% (820,548 GRU vs 822,692 stack); the stack has four depth frames and current
proprioception. See FORMAL_PROTOCOL.md for final-test isolation and reporting.

## Recurrent-call optimization before formal runs

For training chunks with resets only at their first frame, the GRU zeros the
appropriate initial hidden states and processes the entire chunk in one call.
Arbitrary mid-chunk resets still use the general stepwise implementation.
This preserves sequence/reset semantics and changes neither budgets nor loss.
CPU tests compare predictions, hidden state, and all parameter/input gradients.
CUDA comparison gave zero action difference and maximum gradient difference
2.49e-8. A synthetic 8×64 training-window benchmark (both GRU buffers explicitly
flattened) measured 56.95ms → 7.11ms for forward/backward. This is not an end-to-end
training-speed claim: loading, burn-in and Adam are excluded. No-reset evaluation
is unchanged, so existing pilot checkpoints/recordings remain valid.
