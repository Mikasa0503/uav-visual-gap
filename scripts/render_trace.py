"""Blender CPU rendering of saved simulation poses, without trajectory editing.

Run with project-local Blender: --background --python scripts/render_trace.py
-- --evaluation runs/evaluation/NAME --output artifacts/render/NAME [--all].
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Quaternion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from uav_gap.assets import nonoverlapping_wall_boxes
HCSP_IRIS = ROOT/'assets/hcsp_iris/iris.usd'
HCSP_LICENSE = ROOT/'assets/hcsp_iris/LICENSE'
HCSP_IRIS_SHA256 = 'e96833fe3d768c4bd449b02e473dcfe692e46198eaa9f3b91b10a8963a2d1b77'
HCSP_COLOR_REFERENCE = 'hcsp/robots/assets/usd/iris_batVisualOnly.usd'
HCSP_COLOR_REFERENCE_SHA256 = '43bd07b25de3dc8c63babf05543fd3db05694fbabfae12f7f0f2001ffe4d0abe'
HCSP_IRIS_SCALE = .68
HCSP_IRIS_BODY_YAW_DEG = 45
HCSP_BODY_COLOR = (.8, .8, .8)
# Exact linear diffuse color of the plastic_red material referenced by iris.usd.
HCSP_ROTOR_COLOR = (.547994, .027755, .027755)
# HCSP iris.yaml lists mass=1.52 kg, KF=8.54858e-6 and directions below.
# The replay trace has no motor RPM, so this is visual-only nominal hover spin.
HCSP_ROTOR_DIRECTIONS = (1, 1, -1, -1)
HCSP_NOMINAL_ROTOR_RATE_RAD_S = math.sqrt(1.52 * 9.81 / (4 * 8.54858e-6))
parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--episode', type=int, default=0)
parser.add_argument('--all', action='store_true')
parser.add_argument('--indices', help='Comma-separated source-frame indices for diagnostic stills')
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--samples', type=int, default=16)
parser.add_argument('--threads', type=int, default=16)
parser.add_argument('--validation-summary',help='Exact-checkpoint 100-case validation summary for film captions')
parser.add_argument('--phase',choices=['BC','DAgger'])
parser.add_argument('--flight-only',action='store_true',
                    help='Hide renderer annotations and depth inset for a clean flight demo')
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
evaluation, output = (ROOT/args.evaluation).resolve(), (ROOT/args.output).resolve()
evaluation.relative_to(ROOT)
output.relative_to(ROOT)
output.mkdir(parents=True, exist_ok=False)
summary = json.loads((evaluation/'summary.json').read_text())
validation = None
if args.validation_summary:
    validation_path = (ROOT/args.validation_summary).resolve()
    validation_path.relative_to(ROOT)
    validation = json.loads(validation_path.read_text())
    if not (validation['kind']=='visual_student_evaluation' and validation['split']=='validation'
            and validation['condition']=='show' and validation['num_episodes']==100
            and validation['level']==summary['level']==3
            and validation['checkpoint_sha256']==summary['checkpoint_sha256']):
        raise ValueError('Film batch caption requires matching-checkpoint validation')
scenes = json.loads((evaluation/'scenarios.json').read_text())
episodes = [json.loads(line) for line in (evaluation/'episodes.jsonl').read_text().splitlines()]
episode, scenario = episodes[args.episode], scenes[args.episode]
trace_path = evaluation/'trace.npz'
with np.load(trace_path, allow_pickle=False) as data:
    positions = data['position'][:,args.episode].copy()
    rotations = data['quaternion'][:,args.episode].copy()
    dt = float(data['dt'])
    pose_timing = str(data['pose_timing']) if 'pose_timing' in data else 'post_action'
    depths = data['depth'][:,args.episode,0].copy() if 'depth' in data else None
recorded_steps = episode.get('recorded_steps', episode.get('steps', int(math.ceil(episode['seconds']/dt))))
if recorded_steps > episode.get('steps',recorded_steps) and 'continuation' not in summary:
    raise ValueError('Post-terminal rendering requires an explicit continuation declaration')
length = min(len(positions), recorded_steps)
if args.indices:
    indices = [int(x) for x in args.indices.split(',')]
elif args.all:
    indices = list(range(0, length, 2))
else:
    indices = [int(np.abs(positions[:length,0]).argmin())]
if any(i < 0 or i >= length for i in indices):
    raise ValueError('Requested frame outside recorded live episode')

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = args.samples
scene.cycles.transparent_max_bounces = 64
scene.cycles.use_denoising = True
scene.render.threads_mode = 'FIXED'
scene.render.threads = args.threads
scene.render.resolution_x, scene.render.resolution_y = args.width, int(args.width*9/16)
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.fps = round(1/(2*dt))
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.34,.40,.5,1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .5
scene.view_settings.view_transform = 'AgX'


def material(name, color, alpha=1., emission=0., flat=False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, alpha)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color,1)
    bsdf.inputs['Roughness'].default_value = .38
    bsdf.inputs['Specular IOR Level'].default_value = 0.
    bsdf.inputs['Emission Color'].default_value = (*color,1)
    bsdf.inputs['Emission Strength'].default_value = emission
    if alpha < 1:
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs[0].default_value = alpha
        links.new(transparent.outputs[0],mix.inputs[1])
        if flat:
            tint = nodes.new('ShaderNodeEmission')
            tint.inputs['Color'].default_value = (*color,1)
            links.new(tint.outputs[0],mix.inputs[2])
        else:
            links.new(bsdf.outputs[0],mix.inputs[2])
        links.new(mix.outputs[0],nodes.get('Material Output').inputs['Surface'])
    return mat


floor_mat = material('floor', (.018,.028,.055))
grid_mat = material('grid', (.06,.09,.15))
wall_mat = material('wall translucent for visibility only', (.10,.30,.42), .06, flat=True)
edge_mat = material('aperture boundary', (.05,.85,.85), emission=1.)
goal_mat = material('goal', (.08,.85,.44), alpha=.5, emission=.7)
trail_mat = material('actual past trajectory', (.12,.55,1.), emission=.4)
text_mat = material('labels', (.75,.88,1.), emission=1.)


def hcsp_plastic(name, color):
    mat = material(name, color)
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = .24
    bsdf.inputs['Specular IOR Level'].default_value = .45
    return mat


def box(name, location, dimensions, mat, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name, obj.dimensions = name, dimensions
    obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
    return obj


def rod(name, start, end, radius, mat, parent=None):
    start, end = Vector(start), Vector(end)
    direction = end-start
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=direction.length,
                                       location=(start+end)/2)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = direction.to_track_quat('Z','Y')
    obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
    return obj


box('ground', (0,0,-.025), (22,22,.05), floor_mat)
for coord in range(-6,7):
    box('floor grid x', (coord,0,.001), (.008,12,.002), grid_mat)
    box('floor grid y', (0,coord,.001), (12,.008,.002), grid_mat)
gate = bpy.data.objects.new('physical gate transform', None)
scene.collection.objects.link(gate)
gate.location = scenario['gate_center']
width, height, angle = scenario['geometry']
gate.rotation_euler[0] = math.radians(angle)
for location, dimensions in nonoverlapping_wall_boxes(width,height):
    box('physical wall union without overlapping surfaces', location, dimensions, wall_mat, gate)
for sign in (-1,1):
    box('opening horizontal edge', (0,0,sign*(height/2+.007)), (.053,width,.014), edge_mat,gate)
    box('opening vertical edge', (0,sign*(width/2+.007),0), (.053,.014,height+.028), edge_mat,gate)

drone = bpy.data.objects.new('recorded quadrotor pose', None)
scene.collection.objects.link(drone)
drone.rotation_mode = 'QUATERNION'
if not HCSP_IRIS.exists() or not HCSP_LICENSE.exists():
    raise FileNotFoundError('Copy the MIT-licensed HCSP Iris visual into assets/hcsp_iris')
if hashlib.sha256(HCSP_IRIS.read_bytes()).hexdigest() != HCSP_IRIS_SHA256:
    raise ValueError('Unexpected HCSP Iris visual asset')
before_import = set(scene.objects)
bpy.ops.wm.usd_import(filepath=str(HCSP_IRIS))
imported = [obj for obj in scene.objects if obj not in before_import]
imported_meshes = [obj for obj in imported if obj.type == 'MESH']
# iris.usd has five mesh objects. The sixth object in iris_batVisualOnly.usd is
# the volleyball striking platform, which the user explicitly excluded.
if len(imported_meshes) != 5:
    raise ValueError('Expected platform-free HCSP Iris with five mesh objects')
# Blender imports the HCSP MDL material names, but not their MDL shader nodes.
# Rebuild the two referenced plastics in Blender so the render retains HCSP's
# light ABS body and exact plastic_red rotor color instead of default gray.
hcsp_body_mat = hcsp_plastic('HCSP Plastic_ABS', HCSP_BODY_COLOR)
hcsp_rotor_mat = hcsp_plastic('HCSP plastic_red', HCSP_ROTOR_COLOR)
hcsp_material_counts = {'Plastic_ABS': 0, 'plastic_red': 0}
for obj in imported_meshes:
    for slot in obj.material_slots:
        source_name = slot.material.name if slot.material is not None else ''
        if source_name.startswith('Plastic_ABS'):
            slot.material = hcsp_body_mat
            hcsp_material_counts['Plastic_ABS'] += 1
        elif source_name.startswith('plastic_red'):
            slot.material = hcsp_rotor_mat
            hcsp_material_counts['plastic_red'] += 1
        else:
            raise ValueError('Unexpected HCSP Iris material: %s' % source_name)
if hcsp_material_counts != {'Plastic_ABS': 1, 'plastic_red': 4}:
    raise ValueError('Unexpected HCSP Iris material layout: %r' % hcsp_material_counts)
rotors = [next((obj for obj in imported if obj.name == 'rotor_%d' % number), None)
          for number in range(4)]
if any(rotor is None or rotor.type != 'EMPTY' or
       len([child for child in rotor.children if child.type == 'MESH']) != 1
       for rotor in rotors):
    raise ValueError('Expected four HCSP rotor pivots, each with one blade mesh')
visual_frame = bpy.data.objects.new('HCSP Iris visual frame (no striking platform)', None)
scene.collection.objects.link(visual_frame)
visual_frame.parent = drone
visual_frame.scale = (HCSP_IRIS_SCALE,)*3
# HCSP's long/short mesh axes are diagonal to this project's square rotor
# envelope. This fixed body-frame alignment makes the visual footprint about
# 0.45 x 0.45 m inside the unchanged 0.46 x 0.46 m collision envelope.
visual_frame.rotation_euler.z = math.radians(HCSP_IRIS_BODY_YAW_DEG)
for obj in imported:
    if obj.parent not in imported:
        obj.parent = visual_frame
bpy.ops.mesh.primitive_torus_add(major_radius=.3, minor_radius=.007, location=(2,0,1.5))
bpy.context.object.name = 'success goal radius visualization'
bpy.context.object.data.materials.append(goal_mat)

bpy.ops.object.camera_add(location=(-5.5,-7.8,4.1))
camera = bpy.context.object
camera.rotation_euler = (Vector((-.35,0,1.3))-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.type, camera.data.ortho_scale = 'ORTHO', 5.8
scene.camera = camera
for loc, power, size in [((-3,-4,7),1700,6),((3,3,5),1100,5)]:
    bpy.ops.object.light_add(type='AREA', location=loc)
    light = bpy.context.object
    light.data.energy, light.data.shape, light.data.size = power,'DISK',size
    light.rotation_euler = (Vector((0,0,1))-light.location).to_track_quat('-Z','Y').to_euler()


def label(body, location, scale=.09):
    bpy.ops.object.text_add()
    obj = bpy.context.object
    overlay_scale = camera.data.ortho_scale/7.7
    obj.data.body, obj.data.size = body, scale*overlay_scale
    obj.data.materials.append(text_mat)
    obj.parent, obj.location = camera, (location[0]*overlay_scale,location[1]*overlay_scale,location[2])
    obj.hide_render = args.flight_only
    return obj


kind = 'VISUAL STUDENT' if summary['kind']=='visual_student_evaluation' else 'STATE TEACHER'
label(kind+'  |  RECORDED PHYSICS', (-3.60,1.89,-3), .115)
label('%.0f deg   |   %.0f x %.0f cm   |   %s' % (angle,width*100,height*100,summary['condition']),
      (-3.60,1.70,-3), .08)
if 'student' in summary:
    budget_path = (ROOT/summary['checkpoint']).with_suffix('.json')
    budget = json.loads(budget_path.read_text())
    label('%s   |   %s updates   |   %s teacher labels' % (
        summary['student']['kind'].upper(), format(summary['student']['updates'],','),
        format(budget['unique_labels'],',')), (-3.60,1.54,-3), .075)
if validation is not None:
    label('%s   |   Fixed-scene validation: %d/100' % (args.phase or 'Visual policy',
        round(100*validation['success_rate'])),(-3.60,1.38,-3),.075)
if summary['split']=='test':
    label('FINAL TEST | %d fixed cases' % summary['num_episodes'], (.85,1.89,-3), .09)
    result = next(key.upper() for key in ('success','failure','timeout') if episode[key])
    label('Episode %03d: %s' % (args.episode,result), (.85,1.70,-3), .08)
    if summary.get('stress_protocol',{}).get('applicability')=='not_applicable_no_depth_input':
        label('NO DEPTH INPUT: control, N/A visual', (.85,1.51,-3), .065)
label('Wall transparency is for visibility. Motion is the saved simulator trace.', (-3.60,-1.99,-3), .067)
status = label('', (-3.60,-1.82,-3), .09)
depth_image = None
if depths is not None and not args.flight_only:
    depth_image = bpy.data.images.new('actual policy depth input', width=64, height=64, alpha=True)
    depth_image.colorspace_settings.name = 'Non-Color'
    depth_mat = bpy.data.materials.new('policy depth image')
    depth_mat.use_nodes = True
    nodes, links = depth_mat.node_tree.nodes, depth_mat.node_tree.links
    texture = nodes.new('ShaderNodeTexImage')
    texture.image, texture.interpolation = depth_image, 'Closest'
    emission = nodes.new('ShaderNodeEmission')
    links.new(texture.outputs['Color'],emission.inputs['Color'])
    links.new(emission.outputs[0],nodes.get('Material Output').inputs['Surface'])
    bpy.ops.mesh.primitive_plane_add(size=1.)
    screen = bpy.context.object
    overlay_scale = camera.data.ortho_scale/7.7
    screen.parent, screen.location = camera, (2.93*overlay_scale,-1.10*overlay_scale,-2.9)
    screen.scale = (overlay_scale,overlay_scale,overlay_scale)
    screen.data.materials.append(depth_mat)
    label('POLICY DEPTH INPUT', (2.42,-.51,-2.9), .072)
continuation_label = label('', (-3.60,1.10,-3), .08) if 'continuation' in summary else None
trail = None
for frame, index in enumerate(indices):
    drone.location = positions[index]
    x,y,z,w = rotations[index]
    drone.rotation_quaternion = Quaternion((float(w),float(x),float(y),float(z)))
    timestamp = (index+(pose_timing=='post_action'))*dt
    for rotor, direction in zip(rotors, HCSP_ROTOR_DIRECTIONS):
        rotor.rotation_euler.z = (direction * HCSP_NOMINAL_ROTOR_RATE_RAD_S * timestamp) % (2*math.pi)
    if depth_image is not None:
        gray = np.clip(np.flipud(depths[index]),0,1)
        rgba = np.concatenate((np.repeat(gray[:,:,None],3,axis=-1),np.ones((64,64,1))),axis=-1)
        depth_image.pixels.foreach_set(rgba.astype(np.float32).ravel())
        depth_image.update()
    if trail is not None:
        bpy.data.objects.remove(trail, do_unlink=True)
    curve = bpy.data.curves.new('past trajectory', 'CURVE')
    curve.dimensions, curve.bevel_depth, curve.bevel_resolution = '3D', .004, 2
    points = positions[max(0,index-60):index+1]
    spline = curve.splines.new('POLY')
    spline.points.add(len(points)-1)
    for point, position in zip(spline.points,points):
        point.co = (*position,1)
    trail = bpy.data.objects.new('actual past trajectory',curve)
    scene.collection.objects.link(trail)
    curve.materials.append(trail_mat)
    if continuation_label is not None:
        ended = timestamp >= episode['seconds'] and not episode['success']
        continuation_label.data.body = 'FAILED EPISODE | continuing original physics and policy' if ended else ''
    context = 'SHOWCASE' if summary['split']=='showcase' else 'batch success %d/%d' % (
        round(summary['success_rate']*summary['num_episodes']),summary['num_episodes'])
    status.data.body = '%s   |   t = %.2f s   |   %s' % (scenario['id'],timestamp,context)
    scene.render.filepath = str(output/('frame_%05d.png' % frame))
    bpy.ops.render.render(write_still=True)
manifest = dict(source_evaluation=str(evaluation.relative_to(ROOT)), episode=args.episode,
    trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(), source_indices=indices,
    source_dt=dt, pose_timing=pose_timing, output_fps=scene.render.fps, kind=summary['kind'],
    cpu_threads=args.threads, blender_version=bpy.app.version_string,
    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    camera=dict(location=list(camera.location),rotation_euler=list(camera.rotation_euler),
                orthographic_scale=camera.data.ortho_scale),
    resolution=[scene.render.resolution_x,scene.render.resolution_y],cycles_samples=args.samples,
    flight_only=args.flight_only,
    drone_visual=dict(kind='HCSP Iris USD',source='thu-uav/HCSP',asset=str(HCSP_IRIS.relative_to(ROOT)),
        asset_sha256=HCSP_IRIS_SHA256,license='MIT',license_sha256=hashlib.sha256(HCSP_LICENSE.read_bytes()).hexdigest(),
        uniform_scale=HCSP_IRIS_SCALE,body_frame_yaw_degrees=HCSP_IRIS_BODY_YAW_DEG,
        mesh_objects=len(imported_meshes),striking_platform=False,
        materials=dict(Plastic_ABS=list(HCSP_BODY_COLOR),plastic_red=list(HCSP_ROTOR_COLOR),
                       material_slots=hcsp_material_counts,colorspace='scene-linear',
                       color_reference=HCSP_COLOR_REFERENCE,
                       color_reference_sha256=HCSP_COLOR_REFERENCE_SHA256),
        rotor_animation=dict(kind='visual_only_nominal_hover',
            pivot_names=[rotor.name for rotor in rotors],
            directions=list(HCSP_ROTOR_DIRECTIONS),
            nominal_angular_rate_rad_s=HCSP_NOMINAL_ROTOR_RATE_RAD_S,
            parameter_source='HCSP hcsp/robots/assets/usd/iris.yaml mass, force_constants, directions',
            source_time='saved trace index and dt',
            measured_motor_rpm_available=False),
        physics_collision='unchanged conservative 0.46 x 0.46 x 0.12 m envelope'),
    validation_summary_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest() if validation is not None else None,
    continuation=summary.get('continuation'),phase=args.phase,evaluation_summary_sha256=hashlib.sha256((evaluation/'summary.json').read_bytes()).hexdigest(),
    note='Exact saved poses, no smoothing, no retiming; walls translucent for visibility')
(output/'render_manifest.json').write_text(json.dumps(manifest,indent=2))
