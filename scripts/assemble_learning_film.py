"""Assemble all 30 genuine pilot snapshots; preserve speed and disclose every edit.

CPU only. Original replays and their metadata are never modified. This pilot
validation film is deliberately separate from the formal three-seed study.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from uav_gap.runtime import ROOT, project_output

FPS = 25
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ENV = dict(os.environ)
ENV.pop('LD_LIBRARY_PATH', None)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    value = json.loads(subprocess.check_output(['/usr/bin/ffprobe', '-v', 'error',
        '-show_streams', '-show_format', '-of', 'json', str(path)], env=ENV, text=True))
    stream = value['streams'][0]
    if (stream['width'], stream['height'], stream['r_frame_rate']) != (1280, 720, '25/1'):
        raise ValueError('Unexpected replay dimensions/frame rate: '+str(path))
    return int(stream['nb_frames']), float(stream['duration'])


def outcome(episode):
    for key in ('success', 'failure', 'timeout'):
        if episode[key]:
            return key.upper()
    raise ValueError('Episode has no terminal outcome')


def verify_video(path):
    metadata = read(path.with_suffix('.json'))
    if sha(path) != metadata['video_sha256']:
        raise ValueError('Raw replay hash mismatch')
    render = metadata['render']
    evaluation = ROOT/render['source_evaluation']
    if sha(evaluation/'summary.json') != render['evaluation_summary_sha256']:
        raise ValueError('Evaluation summary changed after rendering')
    frames, seconds = probe(path)
    if frames != len(render['source_indices']) or abs(seconds-frames/FPS) > 1e-6:
        raise ValueError('Raw replay timing mismatch')
    if render['pose_timing'] != 'pre_action' or render['output_fps'] != FPS:
        raise ValueError('Unsupported replay timing')
    episode = [json.loads(s) for s in (evaluation/'episodes.jsonl').read_text().splitlines()][render['episode']]
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), frames=frames,
                duration=seconds, episode=episode, outcome=outcome(episode),
                summary=read(evaluation/'summary.json'), render=render)


class Editor:
    def __init__(self, directory):
        self.directory = directory
        self.parts = []
        self.commands = []
        self.text_number = 0

    def text(self, text, x, y, size=24, color='white', enable=None):
        # Text files prevent filter parser/shell interpretation of arbitrary captions.
        self.text_number += 1
        path = self.directory/('caption_%03d.txt' % self.text_number)
        path.write_text(text)
        result = 'drawtext=fontfile=%s:textfile=%s:expansion=none:x=%s:y=%s:fontsize=%d:fontcolor=%s:box=1:boxcolor=0x102330@0.88:boxborderw=8' % (FONT, path, x, y, size, color)
        if enable:
            result += ":enable='%s'" % enable
        return result

    def encode(self, name, input_args, filters, frames, complex_filter=False):
        target = self.directory/(name+'.mp4')
        command = ['/usr/bin/ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                   '-filter_threads', '2', '-filter_complex_threads', '2', *input_args]
        command += (['-filter_complex', filters, '-map', '[out]'] if complex_filter else ['-vf', filters])
        command += ['-an', '-frames:v', str(frames), '-c:v', 'libx264', '-preset', 'medium',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '8', '-movflags', '+faststart', str(target)]
        subprocess.run(command, check=True, env=ENV)
        count, duration = probe(target)
        if count != frames or abs(duration-frames/FPS)>1e-6:
            raise ValueError('Edited segment frame count mismatch')
        self.commands.append(command)
        return target

    def add(self, path, description, **extra):
        frames, seconds = probe(path)
        start = sum(part['frames'] for part in self.parts)/FPS
        self.parts.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                               start_seconds=start, duration_seconds=seconds, frames=frames,
                               description=description, **extra))

    def replay(self, name, raw, frames, label, index=None):
        duration = raw['duration']
        filters = ['setpts=PTS-STARTPTS', 'tpad=stop_mode=clone:stop_duration=10',
                   'trim=end_frame=%d' % frames, self.text(label, 770, 28, 21, '0x8ee7e6')]
        if index is not None:
            filters += ['drawbox=x=40:y=151:w=1200:h=5:color=0x425869:t=fill',
                        'drawbox=x=40:y=151:w=%d:h=5:color=0x8ee7e6:t=fill' % round(1200*index/30)]
        if raw['frames'] > frames:
            caption = 'EXCERPT: first %.2fs of %.2fs replay | full episode: %s' % (frames/FPS,duration,raw['outcome'])
            filters.append(self.text(caption, 42, 174, 21))
        else:
            filters.append(self.text('FULL REPLAY | original speed',42,174,21))
            if raw['frames'] < frames:
                filters.append(self.text('EPISODE ENDED - %s | end-frame hold' % raw['outcome'],
                                         42,211,22,'0xffd28a','gte(n,%d)' % raw['frames']))
        path = self.encode(name, ['-i',str(ROOT/raw['path'])], ','.join(filters), frames)
        self.add(path,label,source_video=raw['path'],source_sha256=raw['sha256'],
                 source_frames=raw['frames'],excerpt_frames=min(frames,raw['frames']),
                 end_hold_frames=max(0,frames-raw['frames']),outcome=raw['outcome'])

    def card(self, name, title, lines, seconds):
        filters = ['drawbox=x=64:y=110:w=1152:h=4:color=0x8ee7e6:t=fill',
                   self.text(title,64,155,40,'0x8ee7e6')]
        filters += [self.text(line,64,250+i*54,25) for i,line in enumerate(lines)]
        path = self.encode(name,['-f','lavfi','-i','color=c=0x102330:s=1280x720:r=25'],','.join(filters),round(seconds*FPS))
        self.add(path,title)

    def comparison(self, snapshots):
        chains = []
        inputs = []
        for i, raw in enumerate(snapshots):
            inputs += ['-i',str(ROOT/raw['path'])]
            filters = ['setpts=PTS-STARTPTS','tpad=stop_mode=clone:stop_duration=8',
                       'trim=end_frame=175','scale=426:240','pad=426:420:0:80:color=0x102330',
                       self.text('%s updates' % format(raw['summary']['student']['updates'],','),18,25,26,'0x8ee7e6'),
                       self.text('ENDED: '+raw['outcome'],18,352,20,'0xffd28a','gte(n,%d)' % raw['frames'])]
            chains.append('[%d:v]%s[v%d]' % (i,','.join(filters),i))
        tail = '[v0][v1][v2]hstack=inputs=3,pad=1280:720:1:150:color=0x102330,'
        tail += self.text('SAME SCENE. SAME CAMERA. ORIGINAL SPEED.',40,55,32,'0x8ee7e6')+','
        tail += self.text('Initial / midpoint / final training budgets. Ended episodes hold the last frame.',40,605,23)+','
        tail += self.text('Pilot seed 11. These individual flights are not aggregate success rates.',40,651,23)+'[out]'
        path = self.encode('comparison',inputs,';'.join(chains+[tail]),175,True)
        self.add(path,'Initial / midpoint / final comparison',sources=[dict(path=r['path'],sha256=r['sha256'],end_hold_frames=175-r['frames']) for r in snapshots])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='pilot_learning_film_v1')
    args=parser.parse_args()
    if not args.name.replace('_','').isalnum():
        raise ValueError('Invalid output name')
    output=project_output('artifacts/film/'+args.name)
    if output.exists():
        raise FileExistsError('Use a new assembly name; inputs and previous edits are preserved')
    batch=read(ROOT/'artifacts/render/pilot_learning_v1/complete.json')
    recording=read(ROOT/'runs/film/pilot_learning_v1/request.json')
    if len(batch['clips'])!=30 or [c['updates'] for c in batch['clips']] != [c['updates'] for c in recording['checkpoints']]:
        raise ValueError('Must retain all 30 preselected checkpoints in chronological order')
    raws=[]
    for row in batch['clips']:
        raw=verify_video(ROOT/row['video'])
        record=read(ROOT/('runs/film/pilot_learning_v1/update_%06d.json' % row['updates']))
        if (raw['sha256']!=row['video_sha256'] or raw['frames']!=row['frames']
            or raw['summary']['checkpoint_sha256']!=record['checkpoint_sha256']
            or raw['summary']['student']['updates']!=row['updates']
            or raw['summary']['teacher_loaded'] or raw['episode']['scene_id']!='showcase_nominal'):
            raise ValueError('Snapshot provenance mismatch')
        raws.append(raw)
    heldout=verify_video(ROOT/'artifacts/video/pilot_final_heldout_000.mp4')
    if (heldout['render']['episode']!=0 or heldout['summary']['split']!='validation'
        or heldout['summary']['condition']!='heldout'
        or heldout['summary']['checkpoint_sha256']!=raws[-1]['summary']['checkpoint_sha256']):
        raise ValueError('Held-out sample must be predeclared episode 0 of the exact final pilot')
    output.mkdir(parents=True)
    editor=Editor(output)
    editor.replay('opening',raws[-1],125,'FINAL VISUAL POLICY | seed 11')
    editor.card('task','LEARNING TO BANK THROUGH A NARROW GAP',[
        '46 x 46 x 12 cm rotor envelope. 66 x 24 cm aperture, tilted 45 degrees.',
        'Cross, recover, then hold for 0.5s near the goal.',
        'PPO state teacher -> visual BC -> closed-loop DAgger.',
        'Student: depth + simulated own-state estimate. No aperture pose.',
        '30 real checkpoints follow. Validation pilot; formal study is separate.'
    ],6)
    for index,raw in enumerate(raws,1):
        editor.replay('learning_%02d'%index,raw,65,'LEARNING %02d/30 | pilot seed 11'%index,index)
    editor.comparison([raws[0],next(r for r in raws if r['summary']['student']['updates']==6000),raws[-1]])
    editor.replay('heldout',heldout,100,'HELD-OUT | predeclared case 000')
    finalshow=read(ROOT/'runs/evaluation/visual_pilot_moredata_seed11_dagger_gru_r9_show/summary.json')
    editor.card('evidence','FROM FAILURE TO CLOSED-LOOP VISUAL FLIGHT',[
        'Final pilot validation: %d/100 fixed aperture; %d/100 held-out geometry.' % (round(100*finalshow['success_rate']),round(100*heldout['summary']['success_rate'])),
        '196,608 unique teacher labels | 12,000 optimizer updates | seed 11.',
        '100 fixed cases per condition. Learning regressions are retained.',
        'Physics replay: exact saved poses, 1x speed; depth inset is policy input.',
        'Simulation only. Own-state estimate is not VIO. Formal results pending.'
    ],8)
    listing=output/'concat.txt'
    listing.write_text(''.join("file '%s'\n" % (ROOT/part['path']) for part in editor.parts))
    film=output/'uav_visual_learning_pilot.mp4'
    command=['/usr/bin/ffmpeg','-y','-hide_banner','-loglevel','error','-f','concat','-safe','0',
             '-i',str(listing),'-c','copy','-movflags','+faststart',str(film)]
    subprocess.run(command,check=True,env=ENV)
    frames,duration=probe(film)
    if frames!=sum(p['frames'] for p in editor.parts) or not 100<=duration<=120:
        raise ValueError('Final duration/frame count outside the declared film specification')
    manifest=dict(schema='truthful_pilot_film_v1',fps=FPS,frames=frames,duration_seconds=duration,
                  video_sha256=sha(film),assembler_sha256=sha(Path(__file__)),
                  raw_batch_manifest_sha256=sha(ROOT/'artifacts/render/pilot_learning_v1/complete.json'),
                  recording_request_sha256=sha(ROOT/'runs/film/pilot_learning_v1/request.json'),
                  seed=11,evidence='pilot validation, not formal three-seed final tests',
                  all_30_snapshots_retained=True,retiming=False,motion_interpolation=False,
                  parts=editor.parts,commands=editor.commands+[command])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(video=str(film.relative_to(ROOT)),duration_seconds=duration,frames=frames),indent=2))


if __name__=='__main__':
    main()
