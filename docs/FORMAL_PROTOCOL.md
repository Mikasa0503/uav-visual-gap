# Formal experiment protocol

## V3 stopped; gradual-geometry diagnostic passed

V3 seed11 L2 epochs1430/1440/1450 each returned0/100 success,0/100 crossing,
100 failures and0 timeouts. The main matrix stopped at the failed gate. No V3 students or final tests ran.
The bounded development experiment is documented in
`plans/2026-09-20-curriculum-bridge-design.md`. It changes only opt-in training
geometry; final L2 evaluation and success criteria remain unchanged. Do not
resume frozen V1/V2/V3 under the modified source, or report this diagnostic as
formal evidence. Its full-L2 epochs1830/1840/1850 each passed100/100, with no
failures or timeouts. This supports the transfer hypothesis for seed11 only; no
replacement three-seed formal revision has been declared.

## Historical V3 declaration: original reward and equal longer l0 curriculum

The original-reward seed22 continuation passed100/100 at all580/590/600
validation snapshots. Its development result is retained separately and is NOT
a formal teacher. V3 starts seeds11/22/33 independently from scratch, each with
300epochs of l0 warmup followed by another300epochs of l0 before the promotion
gate. Remaining phases retain their V2 budgets:150 l1warmup,300 l1,400 l2,
600 l3nominal and500 l3perturbed. Total2550epochs /41,779,200transitions per
teacher. Original reward, geometry, termination, promotion gates and exact-final
show threshold remain unchanged. The dense-cost variant is not adopted.

Frozen historical declarations: configs/teacher_formal_v3.json, visual_formal_v3.json,
teacher_sensor_test_v3.json and formal_video_v3.json. Student labels, updates,
learning rates, parameter matching and all test cases remain identical to V2.
Versioned evaluation names prevent mixing evidence from failed and active runs.

Historical command chain (not to be resumed after its failed gate):
`scripts/run.sh scripts/run_formal_matrix.py --version v3`; after completion
run `scripts/run.sh scripts/run_teacher_sensor_supplement.py --version v3`.
Then aggregate with `scripts/run.sh scripts/aggregate_formal_results.py --version v3`
and render with `scripts/run.sh scripts/render_formal_evaluations.py --config
configs/formal_video_v3.json --name formal_uncut_v3`. All72episode0 selections
were redeclared before any V3 final test results. All9students finish training
before any final test. Failed V1/V2 sources and development evidence stay intact.


## Historical V2 outcome and recovery diagnostics

V2 seed11 passed every curriculum gate and final validation (100/100 show,
99/100 held-out). Seed22 l0 epochs280/290/300 each crossed100/100 but recovered
0/100; all failures were timeouts. The matrix exited1 before any student training
or final testing. V2 must not be resumed under edited source code. Its requests,
checkpoints, traces and failed gate reports remain preserved.

A separate development diagnostic uses seeds22/33/11, each from scratch for
300epochs on l0 with the same PPO/reset settings. Optional dense_v2 reward adds
bounded independent post-crossing distance/speed/tilt costs to the original
reward. Physics, observations, action mapping and success/failure criteria are
unchanged. Every seed receives100-case validation at epochs280/290/300, even if
an earlier seed fails. Results live in
`runs/experiments/teacher_recovery_dense_v2_diagnostic`. No final-test cases are
read, and these runs must never be relabeled as formal seeds. The diagnostic finished: seed11 scored100/100/100, seed22 scored0/0/0, and
seed33 scored0/0/100 at the last three snapshots. It fails the all-seed stability
requirement, so dense_v2 is not adopted. An independent original-reward seed22
continuation to600epochs now tests a larger optimization budget. Any later
formal revision must apply equally to all seeds and start them from scratch.

The sections below preserve the frozen V2 protocol for audit; the current formal
state remains the failed V3 attempt described above, not a claim of completion.

The pilot runs are exploratory and stay separate. Their non-monotonic validation
history informed budget selection; none is silently relabeled as a formal seed.
Validation and test manifests remain unchanged. Final test data must not be used
to choose training budgets, model architecture or checkpoints.

## Teachers

`configs/teacher_formal_v2.json` specifies three independent from-scratch seeds,
11/22/33, each with 512 environments and identical PPO settings. It reproduces
six successful pilot phase boundaries: l0 300, l1 warmup 150, l1 300, l2 400,
l3 nominal 600, l3 reset-v2 500 epochs. First 1,750 epochs use nominal upright
resets (position, velocity and gate randomization still apply). Last 500 add
RPY +/- (8,8,10) degrees and body rates +/-0.25rad/s. Total per seed is 2,250
epochs / 36,864,000 transitions. Twenty percent older-level resets remain.
The l1 warmup is not a promotion boundary; its subsequent 300-epoch phase is.

Revision history: formal v1 used reset-v2 disturbances from the first epoch.
Its seed-11 l0 validation snapshots 280/290/300 all scored 0/100. At epoch 300
all cases crossed but timed out before stable recovery. The gate stopped v1,
exit 1, before any student training or final tests. Failed evidence is retained.
V2 changes the reset curriculum and phase boundaries, not final geometry,
success definition, promotion thresholds, or total per-seed training budget.

Each promotion phase is checked on the same 100 validation starts at the last three
ten-epoch snapshots. All three must score at least 85% before advancing. The
final stage additionally evaluates the exact final checkpoint on show and
held-out cases; show must be at least 95%. These are repeated checks on 100
starts, not three independent 100-case samples. A failed gate stops the run.
Any protocol revision must be explicit and retain the failed run and its budget.

`scripts/run_teacher_curriculum.py` saves the request, source/config/manifest
hashes, commands, per-stage wall times, all gate reports and exact checkpoint
hashes. It checks source consistency before each subprocess. GPU-stage wall
times include initialization and are a conservative accounting measure, not a
measurement of kernel utilization. Completed stages may be resumed only with
identical protocol and verified provenance. Partial output directories are never
overwritten. `scripts/curriculum_job.sh` captures job PID and exit status.

Teacher evaluation records success-conditioned completion/recovery time and
warmed batch-one policy latency. New traces align poses and actions before the
physics step and record this convention explicitly; older pilot teacher traces
retain their documented post-action convention. The teacher has no depth input. Depth-loss cases must therefore be labeled as
input-unaffected controls, not evidence of visual robustness. Extra observation
delay is also meaningful for privileged state and is covered by the supplement below.

## Visual students — frozen after pilot validation

The formal per-seed matrix is BC-GRU, DAgger-GRU and DAgger-stack4. Each seed uses
its matching teacher; groups share round-zero teacher demonstrations, exact
unique-label totals, optimizer-update totals and learning-rate schedule. The
stack model is parameter-matched within 1% of the 820,548-parameter GRU.
Report actual label presentations as well as unique labels; masked tail windows
mean equal update counts do not guarantee equal numbers of valid presentations.

`configs/visual_formal_v2.json` freezes the exact successful pilot recipe:
10 rounds, the first eight collecting 16,384 labels and taking 1,000 updates each,
then two rounds of 32,768 labels and 2,000 updates. Total per student: 196,608
unique labels / 12,000 updates. Learning rate is 3e-4 for rounds 0–5 and 1e-4
for rounds 6–9. The final pilot scored 98/100 show and 88/100 held-out; test
manifests were untouched when this recipe was frozen. Pilot round 8 scored
99/100 and 88/100; it is not substituted for the final fixed-budget checkpoint.

Evaluate the final fixed-budget checkpoint of every
group/seed rather than substituting whichever seed/checkpoint has the best test
score. Report all seeds, their mean/spread, failures and timeouts. Use 100 fixed
test cases per condition. Success targets remain 90% show and 80% held-out.

The film uses seed 11 and the predeclared nominal showcase scene. Actual training
budgets and the associated batch results must appear with each stage, including
regressions. One successful nominal clip is not an aggregate test result.

## Serial execution and final-test isolation

`scripts/run_formal_matrix.py` has 76 explicit stages: one three-seed teacher
curriculum, nine complete student-training runs, 54 student test evaluations,
and 12 teacher test evaluations. All student budgets complete before the first
final-test episode. Each test condition contains 100 fixed scenarios per seed.
The recurrent student has 820,548 parameters; the stack model has 822,692,
a 0.26% difference. Parameter/label/update matching is not a FLOP-matching claim.

The matrix and formal visual runner freeze relevant source, asset, configuration
and scene hashes, preserve stage logs/timings, and verify completed-stage resume.
Physics traces are required for every formal test. Teacher depth-loss is not a visual stress test; the declared control and
privileged-state-delay evaluations are covered separately below. Summary aggregation and
failure interpretation remain subsequent work; a completed job is not itself a
claim that the main method meets every performance target.

## Remaining teacher sensor-condition coverage

The original six-condition scope is preserved. In addition to the active matrix's
four teacher conditions, `configs/teacher_sensor_test_v2.json` declares six
supplemental runs (two conditions × three seeds), before any final test results
are seen. Use the same predeclared test cases and exact final teacher checkpoints.
Extra 40ms means two delayed 50Hz frames of measured privileged state; the latest
known command remains current. The depth-loss condition has no policy-input
effect because the teacher has no camera, so report it as N/A for visual robustness
with an explicitly labeled input-unaffected rollout control. Do not simulate a
fictional teacher visual input. Implemented entrypoints are `evaluate_teacher_sensor_stress.py` and
`run_teacher_sensor_supplement.py --version v2`. The latter requires the main
matrix's completion and verifies exact final teacher provenance. Run
`audit_teacher_sensor_trace.py --evaluation <directory>` on all six outputs to
verify actual delayed inputs and current-command retention, or exact identity
for depth-loss controls. Two-episode pilot integration checks passed; their
success rates are not formal evidence. Main plus supplement totals 7,200 test
episodes. The active matrix's frozen files remain unchanged.

## Verified reporting and uncut case selection

`scripts/aggregate_formal_results.py --version v2 --inventory` reports missing
artifacts without inspecting final-test metrics. The full command, without
`--inventory`, requires both main and supplemental completion markers and all
72 evaluation directories (7,200 episodes). It checks fixed case order and exact
checkpoint/manifest identities, recomputes summary metrics from episode rows,
requires teacher sensor execution audits, and preserves trace hashes. Outputs
are `artifacts/results/formal_v2/{results.json,per_seed.csv,report.md}`. Existing
reports are never overwritten; use a new `--output` directory for a new report.

Success and crossing rates use all cases. Completion/recovery times use successful
episodes only; a seed with no success has undefined timing, never zero. Aggregates
are equal-weight means and sample SD across training seeds, with all per-seed
values retained. The same 100 scenarios repeat across seeds; do not describe them
as 300 unique independent scenes. Latency is mean/SD of per-seed warm batch-one
GPU policy medians/p95s, excluding sensing and transfers. Paired algorithm
comparisons use the same seed and scenario; they are not automatically causal
claims about an isolated architectural mechanism. Stress conditions have their
own frozen randomized starts, so stress-minus-show also contains finite-case
variation and is not a paired clean-versus-stress estimate.

`configs/formal_video_v2.json` was declared while no formal test summaries existed.
It selects episode 0 for ALL four groups, three seeds and six conditions: 72 uncut
replays, independent of outcomes. `scripts/render_formal_evaluations.py --dry-run`
prints that selection; running without it requires main and supplemental
completion. CPU rendering uses four workers with at most sixteen threads each,
full 25fps sampled episodes, original speed, fixed camera and actual depth input.
No hold, excerpt, best-case selection or trajectory smoothing is permitted in
these uncut recordings. Teacher depth-loss control is visibly marked as no-camera
and N/A for visual robustness. The caption changes apply only to final tests;
the completed pilot film retains its existing renderer hashes and source version.
Final-test caption visual QA awaits real final-test data.
