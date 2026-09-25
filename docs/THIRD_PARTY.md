# Third-party dependencies and assets

## Simulator and learning stack

- **Aerial Gym 2.0.1**: upstream project
  [ntnu-arl/aerial_gym_simulator](https://github.com/ntnu-arl/aerial_gym_simulator),
  expected commit `f0d0f05283f7897bab5a1bcc7b19b91cebbab218`. The clean-checkout
  helper fetches that exact commit. The existing local copy is ignored by Git;
  where it lacks Git metadata, its commit cannot be independently checked from
  that copy alone.
- **Isaac Gym Preview 4**: proprietary licensed distribution. Obtain it from
  its official distribution under the applicable license and install its
  `isaacgym/python` package under `third_party/isaacgym/`. It is not included in
  this repository and must not be redistributed by this project.
- **Conda and pip packages**: see `configs/conda-linux-64.explicit.lock`,
  `requirements-runtime.txt`, `configs/runtime-constraints.txt` and the captured
  `requirements-installed.txt` inventory. The official Warp 1.0.0 wheel replaces
  the CPU-only Conda build in the tested environment.

Upstream Aerial Gym source is kept unmodified. Project adapters and changes live
under `src/uav_gap/`.

## Offline-render assets

- Blender 4.2.21 Linux x64 is installed under ignored `third_party/` by
  `scripts/install_blender.sh`; official archive SHA-256:
  `b9ee313018de52697eeabcb76fc2cd6d404dbb670be9b0d3a5847a09ca325981`.
- The offline renderer uses the platform-free Iris visual from HCSP commit
  `009961b8f5702dd0c1c943cef0e01e09dfcd138d`. The selected `iris.usd` SHA-256
  is `e96833fe3d768c4bd449b02e473dcfe692e46198eaa9f3b91b10a8963a2d1b77`; the
  striking-platform mesh is excluded. A separate HCSP model is only a color
  reference. Neither asset changes simulator collision, inertia, policy,
  observations or trajectories.
- The demo's rotor animation uses direction and nominal hover-speed parameters
  from the visual asset configuration. Individual RPM is not captured by the
  learned policy or simulation trace; blade motion is illustrative only.

Generated models, licensed source/binaries, font files and demo media stay in
ignored project-local directories. This repository contains provenance and
setup instructions, not those binary payloads.
