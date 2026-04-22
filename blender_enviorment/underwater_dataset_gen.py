"""
underwater_dataset_gen.py
---
Open Blender -> Scripting tab -> Paste -> Run Script (Made on 5.1.0 Blender)

Output:
    OUTPUT_DIR/
        images/     <label>_####.png
        labels/  <label>_####.txt   (YOLO-Instance Segmentations: class x1 y1 x2 y2 ...)
"""
# Imports
import bpy
import math
import random
import os
import sys
from pathlib import Path
from mathutils import Vector

# CONFIG
## Data Stream
OUTPUT_DIR        = os.path.expanduser("~/Mechatronics_Vision_2026/data")
HDRI_DIR          = os.path.expanduser("~/Mechatronics_Vision_2026/blender_enviorment/HDRIs") # every .exr/.hdr here is loaded once and picked at random per image. Empty/missing folder keeps the scene's current HDRI.
RES_X             = 1280
RES_Y             = 720
GEN_PER_DOI_RANGE = (1, 10) #inclusive (start, end). e.g. (1, 250) generates images 1 through 250. To continue after existing images 1-3, use (4, N) with a new seed.
RANDOM_SEED       = random.randint(-sys.maxsize, sys.maxsize)

## DOIs (Data Of Interest)
'''
collection = exact collection name in the Blender outliner
image_obj  = name of the image-plane object inside that collection
is_obj     = name of the _IS annotation mesh inside that collection (Note: _IS just stands for Instance Segmentation)
class_id   = class (intger, used for Computer Vision training)
label      = used for filenames e.g. firetruck_0001.png
'''
DOIS = [
    {"collection": "Firetruck", "image_obj": "firetruck", "is_obj": "firetruck_IS", "class_id": 0, "label": "firetruck"},
    {"collection": "Fire",      "image_obj": "fire",      "is_obj": "fire_IS",      "class_id": 1, "label": "fire"},
    {"collection": "Ambulance", "image_obj": "ambulance", "is_obj": "ambulance_IS", "class_id": 2, "label": "ambulance"},
    {"collection": "Blood",     "image_obj": "blood",     "is_obj": "blood_IS",     "class_id": 3, "label": "blood"},
]

CAM_NAME        = "camera"
SUN_NAME        = "sun"
VOLUMETRIC_NAME = "volumetric"
OCEAN_NAME      = "ocean"
ANTIHDRI_NAME   = "anti-hdri"

## RANGES - For all of these tuples you're controlling the (min, max) values, a random point in this range will be choosen while generating the data
## Also, the tuples are inclusive, so (0, 250) includes 0 and 250 as possible values to be choosen

### Camera
'''
The camera orbits the DOI on a sphere and always points at it, so the DOI stays in frame. 
augmentation pipeline adds larger shifts on top of these renders.
'''
CAM_DISTANCE_RANGE      = (3.0, 6.0)    # metres from DOI (closer / further)
CAM_AZIMUTH_DEG_RANGE   = (-20.0, 20.0) # left / right tilt around the DOI
CAM_ELEVATION_DEG_RANGE = (15.0, 45.0)  # forward / back tilt around the DOI
FOCAL_MM_RANGE          = (35.0, 85.0)  # Wider visibility for the camera

### Sun
'''
Sun lamps are directional, only rotation affects shading, so we place the sun
at a random offset above the DOI and aim it at the DOI to pick a consistent
light direction per image. Z stays positive so light comes from above.
'''
SUN_X_OFFSET_RANGE      = (-4.0, 4.0)
SUN_Y_OFFSET_RANGE      = (-4.0, 4.0)
SUN_Z_OFFSET_RANGE      = ( 5.0, 10.0)

### DOI rotation (yaw / pitch / roll, degrees)
'''
Applied to image_obj AND is_obj each image so light catches the DOI at
different angles and the _IS annotation tracks with it. Requires image_obj
and is_obj to share the same origin in the .blend.
'''
DOI_YAW_DEG_RANGE       = (-30.0, 30.0) # around world Z
DOI_PITCH_DEG_RANGE     = (-20.0, 20.0) # around local X
DOI_ROLL_DEG_RANGE      = (-15.0, 15.0) # around local Y

### Environment
SUN_STRENGTH_RANGE      = (10.0, 35.0)
VOL_DENSITY_RANGE       = (0.01, 0.35)  # 0.05=clear water, 0.50=murky
VOL_ANISOTROPY_RANGE    = (0.60, 0.90)  # real water ~0.7-0.9
OCEAN_IOR_RANGE         = (1.30, 1.40)  # pure water = 1.333
OCEAN_ROUGHNESS_RANGE   = (0.00, 0.30)
OCEAN_FRAME_RANGE       = (0, 250)      # inclusive ints, matches baked sim
ANTIHDRI_HIDE_PROB      = 0.7           # per-image chance the anti-HDRI shield is hidden so the HDRI peeks through

VOL_COLOR_CHOICES = [
    (0.012, 0.145, 0.306),   # deep ocean blue (~ #03254D)
    (0.020, 0.220, 0.180),   # tropical teal
]
ANTIHDRI_COLOR_CHOICES = [
    (0.005, 0.008, 0.015),   # near-black cold blue
    (0.002, 0.004, 0.008),   # darker, more neutral
]

# Helpers

def get_obj(name: str):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(
            f"Object '{name}' not found. Available: {[o.name for o in bpy.data.objects]}"
        )
    return obj

def get_collection(name: str):
    col = bpy.data.collections.get(name)
    if col is None:
        raise ValueError(
            f"Collection '{name}' not found. Available: {[c.name for c in bpy.data.collections]}"
        )
    return col

def _find_layer_collection(layer_col, name: str):
    if layer_col.name == name:
        return layer_col
    for child in layer_col.children:
        result = _find_layer_collection(child, name)
        if result:
            return result
    return None

def set_collection_render_visibility(collection_name: str, visible: bool):
    # hide_viewport is deliberately untouched in Blender 5.1.0 it also suppresses rendering even with hide_render=False.
    col = get_collection(collection_name)
    layer_col = _find_layer_collection(
        bpy.context.view_layer.layer_collection, collection_name
    )
    if layer_col:
        layer_col.exclude = not visible
    for obj in col.objects:
        obj.hide_render = not visible


def force_hide_is_objects():
    for doi in DOIS:
        obj = bpy.data.objects.get(doi["is_obj"])
        if obj:
            obj.hide_render = True


def _set_node_input(obj_name: str, node_type: str, input_name: str, value):
    obj = get_obj(obj_name)
    for mat in obj.data.materials:
        if mat is None or mat.node_tree is None:
            continue
        for node in mat.node_tree.nodes:
            if node.type == node_type and input_name in node.inputs:
                node.inputs[input_name].default_value = value
                return
    print(f"  WARNING: {node_type}/{input_name} not found on '{obj_name}'")

# Parameter Sampling

def sample_params() -> dict:
    return {
        "focal_mm":        random.uniform(*FOCAL_MM_RANGE),
        "sun_strength":    random.uniform(*SUN_STRENGTH_RANGE),
        "vol_density":     random.uniform(*VOL_DENSITY_RANGE),
        "vol_anisotropy":  random.uniform(*VOL_ANISOTROPY_RANGE),
        "ocean_ior":       random.uniform(*OCEAN_IOR_RANGE),
        "ocean_roughness": random.uniform(*OCEAN_ROUGHNESS_RANGE),
        "ocean_frame":     random.randint(*OCEAN_FRAME_RANGE),
        "vol_color":       random.choice(VOL_COLOR_CHOICES),
        "antihdri_color":  random.choice(ANTIHDRI_COLOR_CHOICES),
    }


def apply_shading(params: dict):
    """Apply all non-camera parameters. Camera is handled by place_camera."""
    sun = get_obj(SUN_NAME)
    if sun.data and hasattr(sun.data, "energy"):
        sun.data.energy = params["sun_strength"]

    _set_node_input(VOLUMETRIC_NAME, "PRINCIPLED_VOLUME", "Density",    params["vol_density"])
    _set_node_input(VOLUMETRIC_NAME, "PRINCIPLED_VOLUME", "Anisotropy", params["vol_anisotropy"])
    r, g, b = params["vol_color"]
    _set_node_input(VOLUMETRIC_NAME, "PRINCIPLED_VOLUME", "Color", (r, g, b, 1.0))

    r, g, b = params["antihdri_color"]
    _set_node_input(ANTIHDRI_NAME, "BSDF_PRINCIPLED", "Base Color", (r, g, b, 1.0))

    _set_node_input(OCEAN_NAME, "BSDF_REFRACTION", "IOR",       params["ocean_ior"])
    _set_node_input(OCEAN_NAME, "BSDF_REFRACTION", "Roughness", params["ocean_roughness"])
    bpy.context.scene.frame_set(int(params["ocean_frame"]))

## Camera

def place_camera(cam_obj, target_obj, focal_mm: float):
    """
    Random spherical position around target_obj, aimed at the target.
    Applies focal length so the projection used for annotations matches
    the render exactly.
    """
    r    = random.uniform(*CAM_DISTANCE_RANGE)
    azim = random.uniform(*CAM_AZIMUTH_DEG_RANGE)
    elev = random.uniform(*CAM_ELEVATION_DEG_RANGE)

    elev_r = math.radians(elev)
    azim_r = math.radians(azim)

    dx = r * math.cos(elev_r) * math.cos(azim_r)
    dy = r * math.cos(elev_r) * math.sin(azim_r)
    dz = r * math.sin(elev_r)

    origin = target_obj.location.copy()
    cam_obj.location  = Vector((origin.x + dx, origin.y + dy, origin.z + dz))
    cam_obj.data.lens = focal_mm

    direction = origin - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

## Sun / DOI / Anti-HDRI / HDRI

def place_sun(sun_obj, target_obj):
    """
    Sun is a directional lamp (only its rotation affects shading), but we
    move the position too so the viewport matches the render intent. Z stays
    positive so the light comes from above.
    """
    dx = random.uniform(*SUN_X_OFFSET_RANGE)
    dy = random.uniform(*SUN_Y_OFFSET_RANGE)
    dz = random.uniform(*SUN_Z_OFFSET_RANGE)

    origin = target_obj.location.copy()
    sun_obj.location = Vector((origin.x + dx, origin.y + dy, origin.z + dz))

    direction = origin - sun_obj.location
    sun_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def randomise_doi_rotation(image_obj, is_obj):
    """
    Same yaw/pitch/roll Euler on image_obj and is_obj so both orient the
    same way in world space. extract_yolo_polygon reads _IS.matrix_world at
    projection time, so the label tracks the rotation automatically.
    """
    pitch = math.radians(random.uniform(*DOI_PITCH_DEG_RANGE))
    roll  = math.radians(random.uniform(*DOI_ROLL_DEG_RANGE))
    yaw   = math.radians(random.uniform(*DOI_YAW_DEG_RANGE))

    image_obj.rotation_euler = (pitch, roll, yaw)
    if is_obj is not None and is_obj is not image_obj:
        is_obj.rotation_euler = (pitch, roll, yaw)


def apply_antihdri_visibility():
    """Random chance the anti-HDRI shield is hidden so the HDRI peeks through."""
    ah = bpy.data.objects.get(ANTIHDRI_NAME)
    if ah is not None:
        ah.hide_render = (random.random() < ANTIHDRI_HIDE_PROB)


def load_hdris() -> list:
    """
    Load every .exr/.hdr under HDRI_DIR into Blender as Image datablocks
    (cached -- loading the same file twice is a no-op). Returns [] if the
    folder is missing or empty, which leaves the scene's current HDRI in place.
    """
    hdri_dir = Path(HDRI_DIR)
    if not hdri_dir.is_dir():
        print(f"  HDRI dir not found ({hdri_dir}); keeping scene's HDRI.")
        return []
    paths = sorted(
        p for p in hdri_dir.iterdir()
        if p.suffix.lower() in (".exr", ".hdr")
    )
    images = [bpy.data.images.load(str(p), check_existing=True) for p in paths]
    print(f"  Loaded {len(images)} HDRI(s) from {hdri_dir}")
    return images


def set_hdri(image):
    """Swap the World's Environment Texture node image."""
    world = bpy.context.scene.world
    if world is None or world.node_tree is None:
        return
    for node in world.node_tree.nodes:
        if node.type == "TEX_ENVIRONMENT":
            node.image = image
            return
    print("  WARNING: no Environment Texture node found in World shader")

## Annotations
'''
Uses Blender's own bpy_extras.object_utils.world_to_camera_view() so the projection exactly matches the current camera state
- covers: focal length, sensor fit, aspect, position and rotation
- keeps annotations aligned when the camera tilts, moves, or zooms the _IS mesh is evaluated in world space using its live matrix_world, then projected through the live camera matrix.
'''

def world_to_image_uv(world_pos: Vector, cam_obj, scene):
    from bpy_extras.object_utils import world_to_camera_view

    co = world_to_camera_view(scene, cam_obj, world_pos)
    if co.z <= 0:
        return None  # behind the camera

    # Blender: (0,0) bottom-left. Image/YOLO: (0,0) top-left.
    return co.x, 1.0 - co.y

def get_ordered_vertex_indices(mesh) -> list:
    """
    Walk the edge graph so vertices come out in perimeter order.
    _IS meshes are closed outlines (every vertex has exactly 2 edges);
    Blender stores vertices in creation order, which may jump around the
    perimeter if they were added out of sequence. If the mesh isn't a
    clean closed loop we warn and fall back to index order.
    """
    n = len(mesh.vertices)
    if n == 0:
        return []

    adj = {i: [] for i in range(n)}
    for edge in mesh.edges:
        v1, v2 = edge.vertices[0], edge.vertices[1]
        adj[v1].append(v2)
        adj[v2].append(v1)

    if not all(len(adj[i]) == 2 for i in range(n)):
        bad = [i for i in range(n) if len(adj[i]) != 2]
        print(f"  WARNING: _IS mesh is not a clean closed loop "
              f"({len(bad)} vertex(es) with != 2 edges). Using index order.")
        return list(range(n))

    ordered = [0]
    prev, current = None, 0
    while True:
        nxt = [v for v in adj[current] if v != prev]
        if not nxt:
            break
        nxt = nxt[0]
        if nxt == 0:
            break
        ordered.append(nxt)
        prev, current = current, nxt
        if len(ordered) > n:
            break
    return ordered


def extract_yolo_polygon(is_obj_name: str, cam_obj, scene):
    """Return flat [x1,y1, x2,y2, ...] normalised coords, or None."""
    is_obj = get_obj(is_obj_name)
    wm     = is_obj.matrix_world
    mesh   = is_obj.data

    order = get_ordered_vertex_indices(mesh)
    flat  = []
    for idx in order:
        uv = world_to_image_uv(wm @ mesh.vertices[idx].co, cam_obj, scene)
        if uv is None:
            continue
        u = max(0.0, min(1.0, uv[0]))
        v = max(0.0, min(1.0, uv[1]))
        flat.extend([round(u, 6), round(v, 6)])
    return flat or None


def write_yolo_label(filepath: str, class_id: int, flat_coords: list):
    with open(filepath, "w") as f:
        f.write(f"{class_id} {' '.join(str(c) for c in flat_coords)}\n")

# Renders

def configure_render(res_x: int, res_y: int):
    s = bpy.context.scene
    s.render.resolution_x          = res_x
    s.render.resolution_y          = res_y
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = "PNG"


def render_to(filepath: str):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

# Main

def main():
    random.seed(RANDOM_SEED)
    scene   = bpy.context.scene
    cam_obj = get_obj(CAM_NAME)
    sun_obj = get_obj(SUN_NAME)
    configure_render(RES_X, RES_Y)

    images_dir   = Path(OUTPUT_DIR) / "images"
    label_dir = Path(OUTPUT_DIR) / "labels"
    for d in (images_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    hdris = load_hdris()

    all_doi_collections = [d["collection"] for d in DOIS]
    start, end = GEN_PER_DOI_RANGE
    n_per_doi = end - start + 1
    total = n_per_doi * len(DOIS)
    print(f"\n  Generating {n_per_doi} x {len(DOIS)} = {total} images (indices {start}-{end})\n")

    for doi in DOIS:
        label, class_id, image_name, is_name, col_name = (
            doi["label"], doi["class_id"], doi["image_obj"],
            doi["is_obj"], doi["collection"],
        )
        print(f"  --- {label} (class {class_id}) ---")

        for other in all_doi_collections:
            set_collection_render_visibility(other, visible=False)
        set_collection_render_visibility(col_name, visible=True)
        force_hide_is_objects()

        target_obj = get_obj(image_name)
        is_obj     = get_obj(is_name)

        for i in range(start, end + 1):
            stem = f"{label}_{i:04d}"

            params = sample_params()
            place_camera(cam_obj, target_obj, params["focal_mm"])
            place_sun(sun_obj, target_obj)
            randomise_doi_rotation(target_obj, is_obj)
            apply_antihdri_visibility()
            if hdris:
                set_hdri(random.choice(hdris))
            apply_shading(params)

            render_to(str(images_dir / f"{stem}.png"))

            flat = extract_yolo_polygon(is_name, cam_obj, scene)
            if flat is None:
                print(f"    WARNING: no projected vertices for {stem}, empty label")
                flat = []
            write_yolo_label(str(label_dir / f"{stem}.txt"), class_id, flat)

            if i == start or i % 25 == 0 or i == end:
                print(f"    {stem}  focal={params['focal_mm']:.0f}  "
                      f"sun={params['sun_strength']:.0f}  "
                      f"density={params['vol_density']:.2f}  "
                      f"frame={params['ocean_frame']}")

    # Restore scene visibility for manual inspection afterwards.
    for doi in DOIS:
        set_collection_render_visibility(doi["collection"], visible=True)
    force_hide_is_objects()

    print(f"\n  Done. Images -> {images_dir}\n        Labels -> {label_dir}\n")

main()