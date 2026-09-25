"""Verify current V10 media metadata and write editable caption text.

This CPU-only helper does not certify a human visual review or formal results.
"""
import hashlib
import json
from pathlib import Path

from uav_gap.runtime import ROOT


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / 'artifacts/film/concise_learning_film_v10_spinning_rotors'
    film = output / 'uav_visual_learning_concise.mp4'
    manifest_path = output / 'manifest.json'
    qa_path = output / 'qa.json'
    captions_path = output / 'captions.json'
    manifest = read(manifest_path)
    qa = read(qa_path)
    captions = read(captions_path)
    if sha(film) != manifest['video_sha256']:
        raise ValueError('Encoded film does not match its manifest hash')
    if manifest['frames'] != 542 or manifest['duration_seconds'] != 21.68:
        raise ValueError('Unexpected V10 duration or frame count')
    if not qa.get('full_decode') or qa.get('frames') != 542 or qa.get('duration_seconds') != 21.68:
        raise ValueError('Run the V10 decode/typography QA before finalizing metadata')
    if len(manifest['parts']) != 5 or len(captions) != 5:
        raise ValueError('Expected four learning clips and one held-out clip')
    text = ('视频文字为 UTF-8 标准文本；字体：Noto Sans CJK SC。\n'
            '影片时长：21.68 秒；542 帧；25 fps。\n'
            '此文件记录可编辑字幕，不代表正式多种子实验或人工视觉复核。\n\n'
            + '\n\n'.join(row['text'] for row in captions) + '\n')
    (output / 'demo_captions.txt').write_text(text, encoding='utf-8')
    print(json.dumps({
        'video': film.relative_to(ROOT).as_posix(),
        'video_sha256': manifest['video_sha256'],
        'caption_file': (output / 'demo_captions.txt').relative_to(ROOT).as_posix(),
        'human_visual_review': qa.get('visual_review', 'not_recorded'),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
