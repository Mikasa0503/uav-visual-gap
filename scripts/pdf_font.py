"""Embed the chosen SC glyphs as a real TrueType subset for PDF compatibility."""
from fontTools.ttLib import TTFont
from fontTools import subset
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.pens.cu2quPen import Cu2QuPen


def build_pdf_font(source,text,target):
    font=TTFont(source)
    options=subset.Options();subsetter=subset.Subsetter(options=options)
    subsetter.populate(text=text);subsetter.subset(font)
    order=font.getGlyphOrder();glyph_set=font.getGlyphSet()
    glyphs={}
    for name in order:
        pen=TTGlyphPen(glyph_set)
        curve_pen=Cu2QuPen(pen,max_err=1.0,reverse_direction=True)
        glyph_set[name].draw(curve_pen)
        glyphs[name]=pen.glyph()
    builder=FontBuilder(font['head'].unitsPerEm,isTTF=True)
    builder.setupGlyphOrder(order);builder.setupCharacterMap(font.getBestCmap())
    builder.setupGlyf(glyphs);builder.setupHorizontalMetrics(font['hmtx'].metrics)
    builder.setupHorizontalHeader(ascent=font['hhea'].ascent,descent=font['hhea'].descent)
    builder.setupNameTable(dict(familyName='UAV Brief Simplified Chinese',styleName='Regular',
        uniqueFontIdentifier='UAVBriefSC-Regular',fullName='UAV Brief SC Regular',psName='UAVBriefSC-Regular'))
    os2=font['OS/2']
    builder.setupOS2(sTypoAscender=os2.sTypoAscender,sTypoDescender=os2.sTypoDescender,
        usWinAscent=os2.usWinAscent,usWinDescent=os2.usWinDescent)
    builder.setupPost();builder.setupMaxp();builder.save(target)
    font.close()
