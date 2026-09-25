# Bounded aperture-transfer diagnostic

V3 stopped with exit1 after seed11 L2 promotion snapshots1430/1440/1450 all
scored0/100, no crossings and100 failures. Saved50Hz pre-action poses show
nearest-gate roll medians -28.2/-29.8/-24.5degrees and gate-y offset about-.33m;
V2's successful seed11 L2 snapshots roll near35degrees and are near-centered.
These sampled poses do not establish actual physical contact timing or prove a
causal mechanism. See runs/diagnostics/formal_v3_aperture/report.json.

Hypothesis: the added L0 training altered the preceding policy and the abrupt
L1-to-L2 geometry change now traps transfer in a failing approach. Test smoother
geometry transfer before freezing another formal protocol. Alternatives are a
longer unchanged L2 continuation (does not address transfer directly) and reward
shaping (adds another mechanism and the previous recovery-shaping test failed).

Add an opt-in curriculum_fraction to training. For current-level samples only,
interpolate aperture width/height/roll from the preceding level toward the current
level. Preserve20% previous-level replay, randomization, original reward, policy,
physics, success criteria, and exact final evaluation geometry. Default1.0 must
preserve the original float32 geometry exactly; explicit evaluation scenes retain
priority. This is a temporary training curriculum, not an easier final task.

Diagnostic seed11 starts from the successful V3 L1 final checkpoint, not the
failed L2 policy: fraction1/3 for200epochs, fraction2/3 for200epochs, then full L2
for400epochs. Cumulative budgets1250/1450/1850. No intermediate gate claims;
last full-L2 snapshots1830/1840/1850 evaluated on the same100 validation cases.
All three must meet85% before this hypothesis is considered supported. Failure
is retained and stops the diagnostic; no automatic extension or test inspection.

Freeze the source/checkpoint hashes in the diagnostic request, run one GPU job
at a time, record per-stage wall time, and retain all outcomes. Unit tests cover
unchanged default, only-current-level interpolation, invalid fractions, and level0.
Do not relabel this diagnostic formal or resume frozen V1/V2/V3 with changed code.
A future formal revision still requires identical declared recipe for all3seeds,
all9students,7200test episodes,72uncut clips and the final report/handoff.
