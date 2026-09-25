# Results and evidence ledger

All scores below are **single-seed validation results**, unless the row is
explicitly marked as a failed formal attempt or development diagnostic. The
checkpoint and manifest hashes identify the evidence used; the underlying
checkpoints, episode traces and reports live in ignored project-local `runs/`
and `checkpoints/` directories and are not part of a public source checkout.

| Result | Condition | Score | Artifact identity | Interpretation |
| --- | --- | ---: | --- | --- |
| PPO state teacher, seed 11, reset protocol v2 | Fixed validation | 100/100 | Checkpoint SHA-256 `757fff5eaed6204928310c1bfb84e2490b1325231db06effc5e30e2727d9b8ba`; scene manifest SHA-256 `e9c8e7a3fab137e96423084e0d19b9294ec1b996bf39bc364ae72a5d9797efa0`; evidence `runs/evaluation/teacher_l3_resetv2_final_show` | Privileged state teacher only. |
| Same teacher checkpoint | Held-out validation | 99/100 | `runs/evaluation/teacher_l3_resetv2_final_heldout` | Validation, not final test or multi-seed aggregate. |
| Visual DAgger-GRU, seed 11, final pilot r9 | Fixed validation | 98/100 | Checkpoint SHA-256 `825f7e55699d07cb05eee500cf93446f9bfc5959af458424c4c04834693bfee8`; scene manifest SHA-256 `e9c8e7a3fab137e96423084e0d19b9294ec1b996bf39bc364ae72a5d9797efa0`; evidence `runs/evaluation/visual_pilot_moredata_seed11_dagger_gru_r9_show` | 196,608 unique teacher labels and 12,000 optimizer updates. |
| Same visual checkpoint | Held-out validation | 88/100 | `runs/evaluation/visual_pilot_moredata_seed11_dagger_gru_r9_heldout` | This is the mature visual student result used by the demo. |
| Formal teacher v3, seed 11 | L2 promotion snapshots 1430/1440/1450 | 0/100 each | `runs/experiments/teacher_formal_v3` and matching evaluation records | Failed; do not describe as a completed formal comparison. |
| Gradual-geometry teacher diagnostic, seed 11 | Full-L2 snapshots 1830/1840/1850 | 100/100 each | `runs/experiments/teacher_aperture_bridge_diagnostic_seed11` | Development diagnostic only; it does not replace the failed formal run. |

The pilot film's evaluated milestones are updates 250, 3,000, 4,000 and 12,000;
the 12,000-update visual checkpoint is the final r9 student. A successful
individual flight in the film does not establish its batch score. Film metadata
and per-clip hashes are in
`artifacts/film/concise_learning_film_v10_spinning_rotors/manifest.json`.
The encoded video SHA-256 is
`6e20f4eb58580311a65497a4f3803216178dee9e384948f70eeea6e46e13ed52`.

## Not established

- No completed three-seed formal comparison across PPO/BC/DAgger variants.
- No final test-set score; validation scenes are not relabeled as test scenes.
- No visual-PPO result, learned motor-RPM policy, VIO, sim-to-real transfer, or
  real-flight test.
- The generated film is a presentation artifact, not a substitute for raw
  traces, exact checkpoints, configurations and evaluation summaries.

To independently recalculate a score, a reviewer needs the corresponding
checkpoint plus the exact frozen config, scene manifest, raw episode outcomes
and summary. These local evidence files must be transferred through an approved
private channel; this document and its hashes alone are insufficient. See
[REPRODUCING.md](REPRODUCING.md) and [PROJECT_BOUNDARIES.md](PROJECT_BOUNDARIES.md).
