"""Encode regularly sampled physical replay frames with timing/provenance checks."""
import argparse
import hashlib
import json
import os
import subprocess
from uav_gap.runtime import ROOT, project_output

parser = argparse.ArgumentParser()
parser.add_argument('--frames', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
frames = (ROOT/args.frames).resolve()
frames.relative_to(ROOT)
manifest = json.loads((frames/'render_manifest.json').read_text())
indices, fps = manifest['source_indices'], manifest['output_fps']
if len(indices) < 2 or any(b-a != 2 for a,b in zip(indices,indices[1:])):
    raise ValueError('Video requires consecutive regularly sampled replay frames')
if abs(fps*2*manifest['source_dt']-1) > 1e-6:
    raise ValueError('Frame timing does not preserve original simulation speed')
for i in range(len(indices)):
    if not (frames/('frame_%05d.png' % i)).is_file():
        raise ValueError('Missing rendered frame')
output = project_output(args.output)
if output.exists():
    raise FileExistsError(output)
output.parent.mkdir(parents=True, exist_ok=True)
encoding_env = dict(os.environ)
encoding_env.pop('LD_LIBRARY_PATH',None)
command = ['/usr/bin/ffmpeg','-y','-hide_banner','-loglevel','error','-framerate',str(fps),
           '-i',str(frames/'frame_%05d.png'),'-frames:v',str(len(indices)),
           '-c:v','libx264','-crf','18','-preset','slow','-pix_fmt','yuv420p',
           '-threads','8','-movflags','+faststart',str(output)]
subprocess.run(command, check=True, env=encoding_env)
probe = json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-show_streams','-show_format',
                                           '-of','json',str(output)], text=True, env=encoding_env))
stream = probe['streams'][0]
if int(stream['nb_frames']) != len(indices) or abs(float(stream['duration'])-len(indices)/fps) > .001:
    raise RuntimeError('Encoded frame count/duration does not match physics replay')
record = dict(render=manifest, encoding=command, probe=probe,
              video_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
output.with_suffix('.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(video=str(output),frames=len(indices),fps=fps,duration_seconds=float(stream['duration'])),indent=2))
