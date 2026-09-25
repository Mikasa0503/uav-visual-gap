"""Decode the final export and sample every caption block at native resolution."""
import json, subprocess
from pathlib import Path
from PIL import Image
import assemble_learning_film as video
from uav_gap.runtime import ROOT
out=ROOT/'artifacts/film/concise_learning_film_v10_spinning_rotors'
manifest=video.read(out/'manifest.json')
movie=out/'uav_visual_learning_concise.mp4'
subprocess.run(['/usr/bin/ffmpeg','-hide_banner','-v','error','-i',str(movie),'-f','null','-'],env=video.ENV,check=True)
headers=[]
samples=[]
for i,part in enumerate(manifest['parts']):
    # Late in each failure clip, all conditional captions are visible.
    t=part['start_seconds']+max(0,part['duration_seconds']-.32)
    target=out/('qa_%02d.png'%i)
    subprocess.run(['/usr/bin/ffmpeg','-y','-hide_banner','-loglevel','error','-ss',str(t),'-i',str(movie),'-frames:v','1',str(target)],env=video.ENV,check=True)
    samples.append(dict(path=str(target.relative_to(ROOT)),time=t))
    if i<5:
        headers.append(Image.open(target).crop((0,610,900,720)).convert('RGB'))
sheet=Image.new('RGB',(900,110*len(headers)))
for i,header in enumerate(headers):sheet.paste(header,(0,i*110))
sheet.save(out/'typography_sheet.png')
assert len(manifest['parts'])==5
assert manifest['frames']==542 and manifest['duration_seconds']==21.68
assert all(p.get('end_hold_frames',0)==0 for p in manifest['parts'])
(out/'qa.json').write_text(json.dumps(dict(full_decode=True,frames=542,duration_seconds=21.68,samples=samples,font_family='Noto Sans CJK SC',caption_region='lower-left',opening_card=False,closing_explanation_card=False,visual_review='pending'),indent=2))
print(json.dumps(samples,indent=2))
