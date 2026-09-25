# Handoff readiness

The final research handoff is **not complete**. The delivered demo and current
policy numbers are seed-11 pilot validation evidence. Formal v3 failed its
seed-11 L2 teacher gate; the separate gradual-geometry result is a development
diagnostic. No final-test score, complete multi-seed comparison, or reviewed
interview report is available. See [RESULTS.md](RESULTS.md) and
[FORMAL_PROTOCOL.md](FORMAL_PROTOCOL.md).

`scripts/package_handoff.py` is a CPU-side gate for a future completed protocol
version. It rejects missing or mixed-version evidence and a dirty source tree.
Its existence or a successful `--check-only` refusal does not imply that the
package can be produced today.

A complete package requires:

- Main and teacher-sensor completion markers bound to frozen requests.
- An aggregate report for 4 groups × 3 seeds × 6 conditions, all 7,200 episode
  outcomes, and hashes of traces, scenes and summaries.
- All 72 predeclared episode-0 full replays and their encoding/source hashes.
- Exactly 12 final policies, their training configurations and dataset
  manifests.
- A reviewed one-page interview report and a demo whose bytes match its
  manifest.
- Reviewed source, runtime/dependency provenance and an inventory for every
  packaged file.

The packager inventories relative paths, byte counts and SHA-256 values, then
checks archived payloads against that inventory. This verifies package
integrity; it does not establish bitwise reproducibility on another host.
Licensed simulator binaries and full training datasets are not redistributed.

The report builder verifies source evidence hashes and Unicode extraction. Its
layout-proof mode contains placeholders and is never a result. A generated PDF
remains a draft until its rendered page is visually reviewed and its provenance
is marked final. The positive full-archive integration check must be rerun only
after the formal evidence exists.
