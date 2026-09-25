"""Create visual/collision-identical local geometry for the task prototype."""
import hashlib
import json
from uav_gap.assets import quadrotor_with_envelope, gap_wall, wall_panel, VEHICLE_HALF_SIZE, scale_inertial_properties
from uav_gap.runtime import ROOT, project_output, assert_environment

assert_environment()
source = ROOT / 'third_party/aerial_gym_simulator/resources/robots/quad/quad.urdf'
directory = project_output('assets/gap')
directory.mkdir(parents=True, exist_ok=True)
(directory / 'quad.urdf').write_text(quadrotor_with_envelope(source.read_text()))
(directory / 'quad_mass120.urdf').write_text(scale_inertial_properties((directory/'quad.urdf').read_text(), 1.2))
(directory / 'wall.urdf').write_text(gap_wall())
(directory / 'panel.urdf').write_text(wall_panel())
manifest = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                vehicle_half_size=VEHICLE_HALF_SIZE, aperture_width=.66, aperture_height=.24,
                wall_thickness=.05, note='Frozen after feasibility_03; reference control is not a learned policy')
(directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
