# Project and evidence boundaries

## Task

A simulated quadrotor must pass its complete conservative rotor envelope through
a static rectangular aperture tilted 45 degrees, then hold near a fixed goal.
The task uses 200 Hz physics/controller updates, 50 Hz policy/depth updates, and
an 8-second episode. Contact, bypass, floor/flight-bound violations and timeout
are tracked separately from success. Success requires a complete crossing and a
stable goal hold.

## Policy hierarchy

| Layer | Training or control | Information and output |
| --- | --- | --- |
| PPO teacher | On-policy state-based training | Privileged aperture pose and simulator state; outputs collective thrust and body rates. |
| Visual student | BC followed by DAgger; causal CNN-GRU | Depth, simulated noisy/delayed own-state, relative goal and previous command; outputs collective thrust and body rates. |
| Low-level adapter | Fixed, not learned in this project | Converts collective thrust/body-rate targets to motor-level actuation through the rate controller. |

The student inference interface does not include aperture pose/geometry, a
crossing-phase flag, teacher observations, or future trajectory data. The
own-state input is a simulator-derived noisy/delayed estimate. It is not a visual
odometry implementation or a real sensor pipeline. Individual motor RPM is not
learned or logged as a policy action. The rotor spin in the offline film is a
nominal visual animation, not measured motor telemetry.

## Evidence categories

- **Code test:** unit or interface behavior checked without claiming flight.
- **Infrastructure smoke:** simulator, depth or rendering path works; no learned
  task-performance claim.
- **Validation:** model selection and pilot evidence on frozen validation
  scenes. It is not final test evidence.
- **Formal test:** predeclared held-out results from the completed multi-seed
  protocol. This project has not completed that gate.
- **Real flight:** no real-aircraft or VIO evidence is claimed.

The current quantitative teacher and student scores are one-seed validation
results. Diagnostic experiments are development evidence and are not pooled with
the formal protocol. See [RESULTS.md](RESULTS.md) for checkpoint identities and
artifact availability.

## Data and licensing

Checkpoints, traces, datasets, generated assets, simulator binaries and videos
are local artifacts excluded from the source repository. Isaac Gym is licensed
software and must be acquired by each reviewer under its distribution terms.
See [THIRD_PARTY.md](THIRD_PARTY.md). A hash recorded in project docs identifies
an artifact; it does not make an omitted artifact available or independently
reproduce its result.
