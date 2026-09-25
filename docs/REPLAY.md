# Faithful offline replay and current demo

## Replay pipeline

`scripts/render_trace.py` reads an evaluation's saved trace and scene metadata.
It copies simulated poses without planning, smoothing, interpolation or edited
trajectories. Recorded 50 Hz states are sampled every second state and encoded at
25 fps, preserving simulation speed. A visual-student inset displays the
recorded delayed depth input. Labels distinguish the privileged state teacher
from the visual student.

The walls are translucent for visibility. The six rendering boxes have the same
union as the four physical wall panels; the rendering split removes coincident
surfaces without changing the aperture. The trailing path uses past states only.

Failed demo flights continue the same policy/controller/physics for two seconds
after the scored failure. The failure outcome remains fixed. No artificial
recoil, altered restitution, scripted trajectory or reset is used. Contact and
velocity reversal are checked before a collision/rebound caption is applied.

## Current primary film: V10

The current edit is
`artifacts/film/concise_learning_film_v10_spinning_rotors/uav_visual_learning_concise.mp4`:
21.68 seconds, 542 frames at 25 fps, SHA-256
`6e20f4eb58580311a65497a4f3803216178dee9e384948f70eeea6e46e13ed52`. It shows
four full original-speed student flights at updates 250, 3,000, 4,000 and 12,000,
followed by one predeclared held-out flight. There is no opening card, ending
card, frame hold or retiming. It shows only gap-flight footage with a minimal
lower-left data line.

Failed flights retain two seconds of post-terminal physical continuation, while
the original scored failure remains unchanged. Chinese captions are standard
UTF-8 text rendered with the explicitly selected Noto Sans CJK SC font face; the
font identity/hash, captions, per-part hashes, encoding commands and QA are in
the adjacent `manifest.json`; editable captions and automated decode/typography
checks are in `captions.json` and `qa.json`. The QA record keeps human visual
review separate from machine checks. The V10 film is one-seed pilot evidence,
not a formal result. The rotor animation is nominal visual motion, not measured
RPM.

The output directory and its sources are local generated artifacts excluded
from Git. Once the source evaluation traces are available locally, rebuild to a
new directory so the recorded version remains immutable:

```bash
scripts/run.sh scripts/assemble_concise_film.py --name concise_learning_film_reproduction
```

The assembler verifies source hashes and checkpoint/result bindings before
encoding. Do not treat an isolated showcase flight as its associated batch
success rate. For quantitative results and experiment boundaries, see
[RESULTS.md](RESULTS.md).

The earlier replay workflow below remains useful for individual uncut episodes.

## Individual trace replay

```bash
third_party/blender-4.2.21-linux-x64/blender --background \
  --python scripts/render_trace.py -- \
  --evaluation runs/evaluation/teacher_l2_epoch1150 \
  --output artifacts/render/teacher_l2_replay --all --width 960 --samples 8
scripts/run.sh scripts/encode_replay.py \
  --frames artifacts/render/teacher_l2_replay \
  --output artifacts/video/teacher_35deg_replay.mp4
```

The encoder verifies consecutive source-frame sampling, frame count and
`ffprobe` duration, then saves source/video hashes and command provenance. A
state-teacher rendering demonstrates teacher behavior only; it is not evidence
of visual control. Formal replay export remains incomplete until the formal
experiment gates pass.
