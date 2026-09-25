"""Select a specific Simplified Chinese face, never the TTC's default JP face."""
from fontTools.ttLib import TTFont
from uav_gap.runtime import ROOT

SOURCE = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def simplified_chinese_font():
    target = ROOT/'artifacts/fonts/NotoSansCJKsc-Regular.otf'
    if not target.exists():
        font = TTFont(SOURCE, fontNumber=2)
        if font['name'].getDebugName(1) != 'Noto Sans CJK SC':
            raise ValueError('Expected Simplified Chinese face at index 2')
        target.parent.mkdir(parents=True, exist_ok=True)
        font.save(target)
        font.close()
    font = TTFont(target)
    if font['name'].getDebugName(1) != 'Noto Sans CJK SC':
        raise ValueError('Wrong extracted font family')
    font.close()
    return target


def validate_text(text, path):
    font = TTFont(path)
    cmap = font.getBestCmap()
    missing = sorted({character for character in text if not character.isspace() and ord(character) not in cmap})
    font.close()
    if missing:
        raise ValueError('Missing font glyphs: '+repr(missing))
