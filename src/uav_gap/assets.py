"""Generate project-owned assets without editing upstream URDFs."""

import xml.etree.ElementTree as ET

VEHICLE_HALF_SIZE = (.23, .23, .06)
ROTOR_RADIUS = .10


def quadrotor_with_envelope(source_xml):
    root = ET.fromstring(source_xml)
    base = root.find("./link[@name='base_link']")
    if base is None:
        raise ValueError('Expected base_link in upstream quadrotor')
    for collision in list(base.findall('collision')):
        base.remove(collision)
    collision = ET.SubElement(base, 'collision')
    ET.SubElement(collision, 'origin', xyz='0 0 0')
    geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(geometry, 'box', size=' '.join(str(2 * x) for x in VEHICLE_HALF_SIZE))
    for i in range(4):
        motor = root.find("./link[@name='motor_%d']" % i)
        if motor is None:
            raise ValueError('Expected motor_%d' % i)
        visual = ET.SubElement(motor, 'visual')
        ET.SubElement(visual, 'origin', xyz='0 0 .03')
        geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(geometry, 'cylinder', radius=str(ROTOR_RADIUS), length='.004')
        material = ET.SubElement(visual, 'material', name='rotor_%d' % i)
        ET.SubElement(material, 'color', rgba='0.1 0.6 1 0.75' if i < 2 else '1 .4 .05 .75')
    return ET.tostring(root, encoding='unicode')


def gap_wall(width=.66, height=.24, thickness=.05, span=8.):
    if not (0 < width < span and 0 < height < span and thickness > 0):
        raise ValueError('Invalid wall/aperture dimensions')
    root = ET.Element('robot', name='gap_wall')
    link = ET.SubElement(root, 'link', name='wall')
    # Local x normal, y long aperture dimension, z short dimension.
    pieces = [
        ((0, 0, (span + height) / 4), (thickness, span, (span - height) / 2)),
        ((0, 0, -(span + height) / 4), (thickness, span, (span - height) / 2)),
        ((0, (span + width) / 4, 0), (thickness, (span - width) / 2, height)),
        ((0, -(span + width) / 4, 0), (thickness, (span - width) / 2, height)),
    ]
    for origin, size in pieces:
        for kind in ('visual', 'collision'):
            element = ET.SubElement(link, kind)
            ET.SubElement(element, 'origin', xyz=' '.join(map(str, origin)))
            geometry = ET.SubElement(element, 'geometry')
            ET.SubElement(geometry, 'box', size=' '.join(map(str, size)))
            if kind == 'visual':
                material = ET.SubElement(element, 'material', name='wall_blue')
                ET.SubElement(material, 'color', rgba='.10 .18 .27 1')
    return ET.tostring(root, encoding='unicode')


def wall_panel(span=8., thickness=.05):
    root = ET.Element('robot', name='gap_panel')
    link = ET.SubElement(root, 'link', name='panel')
    for kind in ('visual', 'collision'):
        element = ET.SubElement(link, kind)
        geometry = ET.SubElement(element, 'geometry')
        ET.SubElement(geometry, 'box', size='%s %s %s' % (thickness, span, span))
        if kind == 'visual':
            mat = ET.SubElement(element, 'material', name='panel_blue')
            ET.SubElement(mat, 'color', rgba='.10 .18 .27 1')
    return ET.tostring(root, encoding='unicode')


def scale_inertial_properties(source_xml, scale):
    """Scale physical mass/inertia only, preserving collision and visuals."""
    if scale <= 0:
        raise ValueError('Mass scale must be positive')
    root = ET.fromstring(source_xml)
    for inertial in root.findall('./link/inertial'):
        mass = inertial.find('mass')
        mass.set('value', str(float(mass.get('value')) * scale))
        inertia = inertial.find('inertia')
        for key, value in list(inertia.attrib.items()):
            inertia.set(key, str(float(value) * scale))
    return ET.tostring(root, encoding='unicode')


def nonoverlapping_wall_boxes(width, height):
    """Exact union of the four physical panels, partitioned to avoid ray z-fighting."""
    pieces = []
    for sign in (-1,1):
        pieces.append(((0,0,sign*(height/2+4)), (.05,8,8)))
        pieces.append(((0,sign*(4+width/2)/2,0), (.05,4-width/2,height)))
        pieces.append(((0,sign*(6+width/4),0), (.05,4+width/2,8)))
    return pieces
