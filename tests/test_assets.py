import xml.etree.ElementTree as ET
import pytest
from uav_gap.assets import gap_wall, quadrotor_with_envelope, VEHICLE_HALF_SIZE


def test_gap_visual_and_collision_boxes_are_identical():
    root = ET.fromstring(gap_wall())
    visuals = root.findall('./link/visual')
    collisions = root.findall('./link/collision')
    assert len(visuals) == len(collisions) == 4
    for visual, collision in zip(visuals, collisions):
        assert visual.find('geometry/box').attrib == collision.find('geometry/box').attrib
        assert visual.find('origin').attrib == collision.find('origin').attrib


def test_vehicle_envelope_replaces_sphere_and_preserves_motors():
    xml = '<robot><link name="base_link"><collision><geometry><sphere radius=".18"/></geometry></collision></link>'
    xml += ''.join('<link name="motor_%d"/>' % i for i in range(4)) + '</robot>'
    root = ET.fromstring(quadrotor_with_envelope(xml))
    box = root.find('./link/collision/geometry/box')
    assert tuple(map(float, box.attrib['size'].split())) == tuple(2*x for x in VEHICLE_HALF_SIZE)
    assert not root.findall('./link/collision/geometry/sphere')
    assert len(root.findall('./link/visual/geometry/cylinder')) == 4


def test_invalid_aperture_rejected():
    with pytest.raises(ValueError):
        gap_wall(width=-1.)


def test_render_partition_matches_full_physical_wall_union():
    import numpy as np
    from uav_gap.assets import nonoverlapping_wall_boxes
    width, height = .66, .24
    rng = np.random.default_rng(5)
    points = rng.uniform([-0.03,-9,-9],[.03,9,9],(10000,3))
    physical = [((0,y,z),(.05,8,8)) for y,z in
                [(width/2+4,0),(-width/2-4,0),(0,height/2+4),(0,-height/2-4)]]
    def membership(boxes):
        return np.stack([(abs(points-np.array(center)) < np.array(size)/2).all(-1)
                         for center,size in boxes])
    rendered = membership(nonoverlapping_wall_boxes(width,height))
    assert np.array_equal(rendered.any(0),membership(physical).any(0))
    assert rendered.sum(0).max() == 1
