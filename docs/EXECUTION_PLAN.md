# Accepted study scope and remaining gates

## Goal

Build a reproducible simulation study of a quadrotor learning to cross a tilted
narrow aperture from depth input, and present the learning progression with a
faithful, concise demo. This file is the scope of record; generated results and
private operational history are kept elsewhere.

## Fixed task and interfaces

- One quadrotor and a static rectangular aperture tilted 45 degrees, followed by
  a goal that remains fixed as aperture conditions vary.
- Conservative rotor-envelope collision checks at every physical substep,
  including swept motion. Contact, bypass, bounds and timeout are distinct
  outcomes.
- 200 Hz physics/controller, 50 Hz depth/policy and an 8-second episode.
  Success requires the complete body to cross and then hold within 0.3 m of the
  goal, below 0.3 m/s, under 10 degrees tilt, for 0.5 seconds.
- Teacher: PPO MLP with privileged aperture pose and simulator state. Student:
  causal visual CNN-GRU trained with BC and DAgger. Student input is depth,
  simulated noisy/delayed own-state, relative goal and previous action; output is
  collective thrust plus body rates. A fixed rate controller handles lower-level
  motor allocation.
- The own-state estimate is simulated, not VIO. No VLA, dynamic obstacles,
  visual-PPO fine-tuning, learned motor RPM or real-flight claim is in scope.

## Acceptance plan and current state

| Gate | Acceptance | Current state |
| --- | --- | --- |
| Environment and task | Isolated dependencies; physical model, depth, rate adapter, collision and trajectory tests | Implemented; runtime checks and CPU suite must pass on the reviewed tree. |
| PPO teacher | Fixed and held-out evaluation with recovery; exact checkpoint and scene provenance | A reset-v2 seed-11 teacher scored 100/100 fixed and 99/100 held-out validation. Later formal v3 failed its seed-11 L2 gate; that attempt remains failed. |
| Visual student | BC then DAgger; no geometric privilege; recurrent memory reset at episode boundaries | Seed-11 r9 pilot used 196,608 labels / 12,000 updates and scored 98/100 fixed, 88/100 held-out validation. |
| Comparative study | Three seeds each for PPO teacher, BC-GRU, DAgger-GRU and DAgger frame-stack, with matched data/optimization budgets | Incomplete. The single-seed pilot is not a substitute. |
| Generalization/stress | 100 episodes per seed for fixed, held-out and separate delay, depth-loss, mass and lateral-disturbance conditions; report seed spread, failures, recovery and latency | Incomplete; no final-test score is claimed. |
| Demo and handoff | Faithful clips, raw replay provenance, final report and complete package | V10 short demo exists. Full formal handoff and interview report remain incomplete. |

## Remaining work

1. Freeze a new formal protocol revision before continuing after the v3 L2
   failure. Document reward/curriculum changes and promotion criteria; do not
   resume a failed run under changed sources.
2. Validate the frozen protocol on three declared seeds, preserving exact
   checkpoints, configurations, labels, datasets, scene manifests and episode
   outcomes from each run.
3. Complete matched BC-GRU, DAgger-GRU and DAgger-frame-stack groups at equal
   budgets. Keep training, validation and final-test scenes/seed sets distinct.
4. Run the predeclared fixed, held-out and individual stress conditions. Report
   per-seed scores and spread, failures, recovery time and inference latency.
5. Rebuild the final evidence report and package only after the formal gates pass.
   Preserve negative results and keep validation, development diagnostics and
   final-test evidence in separate categories.

## Demo boundary

The current V10 film is a pilot presentation artifact: four policy milestones,
a predeclared held-out flight, original-speed clips, and two seconds of measured
post-failure physics for failed flights. It has no opening/ending card. It does
not claim monotonic improvement or replace quantitative validation. Exact film
identity and limitations are recorded in [REPLAY.md](REPLAY.md) and
[RESULTS.md](RESULTS.md).
