"""Build RoboVerse scenes from imported Genie Sim rigid-object specifications."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from copy import deepcopy
from itertools import pairwise
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.spatial.transform import Rotation

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import BaseCameraCfg, FisheyeCameraCfg, PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, DomeLightCfg
from metasim.scenario.objects import ArticulationObjCfg, PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.fisheye import invertible_fisheye_fov
from roboverse_pack.tasks.icl.asset_sources import HISTORICAL_ASSETS, source_revision

_FORMAT = "roboverse.icl.geniesim_benchmark_specs.v1"
_OBJECT_FORMAT = "roboverse.icl.geniesim_rigid_object.v3"
_SCENE_WRAPPER_FORMAT = "roboverse.icl.geniesim_scene_object_wrapper.v4"
_SOURCE_SCENE_FORMAT = "roboverse.icl.geniesim_source_scene.v1"
_ARTICULATED_OBJECT_FORMAT = "roboverse.icl.geniesim_articulated_object.v1"
_ARTICULATED_WRAPPER_FORMAT = "roboverse.icl.geniesim_articulated_instance.v1"
_G1_SOURCE_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
_GENIESIM_ASSET_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
_GENIESIM_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "icl" / "sources" / "geniesim-assets"
_SOURCE_ENVIRONMENT_MAX_FACES = 40_000
_SOURCE_ENVIRONMENT_MAX_TEXTURE_SIZE = 1024
_STUDIO_PANORAMA_MAX_WIDTH = 1024
_BILLIARDS_SPHERE_ASSETS = frozenset({
    "benchmark_billiards_007",
    "benchmark_billiards_008",
    "benchmark_billiards_009",
    "benchmark_billiards_010",
    "benchmark_billiards_011",
    "benchmark_billiards_013",
})
# Use a common simulated time per action across backends. The source records
# at 30 fps; a finer solver step improves stability for the portable models.
_BENCHMARK_ACTION_HZ = 30
_BENCHMARK_PHYSICS_SUBSTEPS = 20
_COMPATIBILITY_ASSET_ALIASES = {
    "waic_beverage_bottle_009": {
        "asset_id": "genie_beverage_bottle_009",
        "source_revision": "a0813aad7c16165daffbe6c2737754e0344809ee",
        "aligned_usd_sha256": "260317e37c02b49254fc9b18392436be22f1c4d4e7088d0fd47e5fa6eb5ced0c",
        "reason": "exact WAIC USD is unavailable; candidate dimensions match, identity is unverified",
    }
}


def load_instance_spec(spec_path: Path, task: str, instance_id: str) -> dict:
    """Load one source task instance from the imported, provenance-linked benchmark pack."""
    pack = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if pack.get("format") != _FORMAT:
        raise ValueError(f"Unsupported Genie Sim spec format: {pack.get('format')!r}")
    key = f"{task}/{instance_id}"
    try:
        return pack["instances"][key]
    except KeyError as exc:
        raise KeyError(f"Unknown Genie Sim scene instance {key!r} in {spec_path}") from exc


def load_run_instance(spec_path: Path, run_id: str, instance_index: int = 0) -> tuple[dict, dict, dict]:
    """Resolve one source run to its scene and evaluation configuration."""
    if instance_index < 0:
        raise ValueError("instance_index must be non-negative")
    pack = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if pack.get("format") != _FORMAT:
        raise ValueError(f"Unsupported Genie Sim spec format: {pack.get('format')!r}")
    run = next((item for item in pack.get("runs", []) if item.get("id") == run_id), None)
    if run is None:
        raise KeyError(f"Unknown Genie Sim run {run_id!r} in {spec_path}")
    try:
        instance_key = run["instances"][instance_index]
        instance = pack["instances"][instance_key]
        evaluation = pack["evaluation_configs"][run["evaluation_config"]]["config"]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Run {run_id!r} has no valid instance/configuration at index {instance_index}") from exc
    resolved_run = dict(run)
    initial_poses = pack.get("initial_pose_source")
    if initial_poses is not None:
        try:
            positions = initial_poses["joint_positions"][run["subtask"]][run["initial_pose_robot"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Source reset pose is missing for run {run_id!r}") from exc
        if not isinstance(positions, dict):
            raise ValueError(f"Source reset pose for run {run_id!r} must be a named joint mapping")
        resolved_run["source_initial_positions"] = positions
        resolved_run["initial_pose_options"] = initial_poses["joint_positions"][run["subtask"]]
        # A few source task definitions only supply one robot's pose. The
        # common tabletop source pose is used solely for cross-robot scene
        # inspection and is explicitly recorded as an unverified adaptation.
        resolved_run["cross_robot_reference_poses"] = initial_poses["joint_positions"].get("pick_block_color", {})
    robot_config_name = evaluation.get("robot", {}).get("robot_cfg")
    if robot_config_name:
        try:
            resolved_run["robot_config"] = pack["robot_configs"][robot_config_name]["config"]
        except KeyError as exc:
            raise ValueError(f"Source robot configuration {robot_config_name!r} was not imported") from exc
    return resolved_run, instance, evaluation


def geniesim_pinhole_camera(
    source: dict, *, camera_name: str, robot_name: str = "agibot_g1", image_size: int = 640
) -> PinholeCameraCfg:
    """Convert one source USD camera into a MetaSim pinhole camera configuration."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    try:
        matrix = np.asarray(source["head_link_from_camera_usd_row_matrix"], dtype=float)
        focal_length = float(source["focalLength"])
        aperture = float(source["horizontalAperture"])
        vertical_aperture = float(source["verticalAperture"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Incomplete source calibration for G1 camera {camera_name}") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid source transform for G1 camera {camera_name}")
    if not all(math.isfinite(value) and value > 0 for value in (focal_length, aperture, vertical_aperture)):
        raise ValueError(f"Invalid source intrinsics for G1 camera {camera_name}")
    quaternion = Rotation.from_matrix(matrix[:3, :3].T).as_quat(scalar_first=True)
    native = source.get("native_image_size")
    calibration = source.get("opencv_pinhole")
    if native is not None:
        if not isinstance(native, (list, tuple)) or len(native) != 2 or any(int(value) <= 0 for value in native):
            raise ValueError(f"Invalid native image size for G1 camera {camera_name}")
        source_width, source_height = map(int, native)
        height = max(1, round(image_size * source_height / source_width))
    else:
        source_width = source_height = None
        height = max(1, round(image_size * vertical_aperture / aperture))
    intrinsic = None
    if calibration is not None:
        if source_width is None or not isinstance(calibration, dict) or not {"fx", "fy", "cx", "cy"}.issubset(calibration):
            raise ValueError(f"Incomplete OpenCV calibration for G1 camera {camera_name}")
        fx, fy, cx, cy = (float(calibration[key]) for key in ("fx", "fy", "cx", "cy"))
        if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
            raise ValueError(f"Invalid OpenCV calibration for G1 camera {camera_name}")
        # Backends without an explicit intrinsic matrix still use the source
        # horizontal focal length; their centered, square-pixel pinhole model
        # cannot reproduce arbitrary principal points or lens distortion.
        focal_length = aperture * fx / source_width
        intrinsic = [
            [fx * image_size / source_width, 0.0, cx * image_size / source_width],
            [0.0, fy * height / source_height, cy * height / source_height],
            [0.0, 0.0, 1.0],
        ]
    distortion_keys = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4")
    distortion = None
    if calibration is not None and any(key in calibration for key in distortion_keys):
        distortion = tuple(float(calibration.get(key, 0.0)) for key in distortion_keys)
    return PinholeCameraCfg(
        name=camera_name,
        width=image_size,
        height=height,
        data_types=["rgb"],
        mount_to=robot_name,
        mount_link="head_link2",
        mount_pos=tuple(float(value) for value in matrix[3, :3]),
        mount_quat=tuple(float(value) for value in quaternion),
        focal_length=focal_length,
        horizontal_aperture=aperture,
        intrinsic=intrinsic,
        opencv_distortion=distortion,
        distortion_source="geniesim_g1_usd" if distortion is not None else None,
    )


def geniesim_g1_cameras(
    robot_root: Path, image_size: int = 640, *, fisheye_projection: str = "calibrated",
    camera_names: tuple[str, ...] | None = None,
) -> list[BaseCameraCfg]:
    """Use G1 camera mounts with G2 optics until G1 is calibrated."""
    if fisheye_projection not in {"calibrated", "pinhole"}:
        raise ValueError("fisheye_projection must be calibrated or pinhole")
    provenance_path = Path(robot_root) / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("source_revision") != _G1_SOURCE_REVISION:
        raise ValueError(f"G1 camera calibration is not from pinned Genie Sim revision: {provenance_path}")
    source_cameras = provenance.get("cameras", {})
    camera_specs = (
        ("Head_Camera", "head_rgb"),
        ("Fisheye_Camera_01", "head_fisheye_left"),
        ("Fisheye_Camera_02", "head_fisheye_right"),
    )
    missing = [source_name for source_name, name in camera_specs if (camera_names is None or name in camera_names) and source_name not in source_cameras]
    if missing:
        raise ValueError(f"Missing pinned Genie Sim G1 camera calibrations: {missing}")
    cameras = [
        geniesim_pinhole_camera(source_cameras[source_name], camera_name=camera_name, image_size=image_size)
        for source_name, camera_name in camera_specs if camera_names is None or camera_name in camera_names
    ]
    # G1's source USD does not provide a usable lens-distortion calibration.
    # Borrow every G2 lens model/intrinsic by matching camera names, while
    # retaining G1's separately calibrated mounting transforms. This is an
    # explicit temporary cross-robot approximation, not G1 hardware calibration.
    g2 = {camera.name: camera for camera in geniesim_g2_cameras(
        image_size=image_size, camera_profile="hardware", fisheye_projection=fisheye_projection
    )}
    cameras = [_copy_camera_optics(camera, g2[camera.name]) for camera in cameras]
    return cameras


def _copy_camera_optics(camera: BaseCameraCfg, optics: BaseCameraCfg) -> BaseCameraCfg:
    """Apply another camera's lens and image geometry while retaining this mount."""
    mount = {
        "name": camera.name,
        "width": optics.width,
        "height": optics.height,
        "data_types": list(camera.data_types),
        "mount_to": camera.mount_to,
        "mount_link": camera.mount_link,
        "mount_pos": camera.mount_pos,
        "mount_quat": camera.mount_quat,
    }
    if isinstance(optics, FisheyeCameraCfg):
        return FisheyeCameraCfg(
            **mount,
            projection_model=optics.projection_model,
            calibration_size=optics.calibration_size,
            optical_center=optics.optical_center,
            focal_length_px=optics.focal_length_px,
            coefficients=optics.coefficients,
            max_fov_rad=optics.max_fov_rad,
            calibration_source="borrowed_g2",
            calibration_source_sha256=optics.calibration_source_sha256,
            max_fov_source=optics.max_fov_source,
        )
    if not isinstance(optics, PinholeCameraCfg):
        raise TypeError(f"Unsupported G2 camera optics: {type(optics).__name__}")
    if not isinstance(camera, PinholeCameraCfg):
        raise TypeError(f"G1 camera {camera.name!r} is incompatible with G2 pinhole optics")
    return PinholeCameraCfg(
        **mount,
        focal_length=optics.focal_length,
        horizontal_aperture=optics.horizontal_aperture,
        intrinsic=deepcopy(optics.intrinsic),
        opencv_distortion=deepcopy(optics.opencv_distortion),
        distortion_source=optics.distortion_source,
    )


def geniesim_g2_cameras(
    image_size: int = 640,
    *,
    camera_profile: str = "hardware",
    robot_name: str = "agibot_g2",
    source_robot_usd: Path | None = None,
    fisheye_projection: str = "calibrated",
) -> list[BaseCameraCfg]:
    """Load the G2 camera transforms and intrinsics authored in Genie Sim USD.

    ``hardware`` exposes the head RGB and its two fisheye cameras, matching
    the user's G1 observation layout. ``benchmark`` exposes the head RGB and
    the two wrist cameras used by the RoboColiseum inference contract. MetaSim
    uses the source OpenCV fisheye lens parameters by default.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if camera_profile not in {"hardware", "benchmark"}:
        raise ValueError("camera_profile must be 'hardware' or 'benchmark'")
    if fisheye_projection not in {"calibrated", "pinhole"}:
        raise ValueError("fisheye_projection must be calibrated or pinhole")
    source_robot_usd = Path(source_robot_usd or (_GENIESIM_SOURCE_ROOT / "robot" / "G2_omnipicker" / "robot.usda"))
    if not source_robot_usd.is_file():
        raise FileNotFoundError(f"Missing Genie Sim G2 camera source USD: {source_robot_usd}")
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("Reading Genie Sim G2 camera calibration requires usd-core (pxr)") from exc
    stage = Usd.Stage.Open(str(source_robot_usd))
    if stage is None:
        raise ValueError(f"Cannot open Genie Sim G2 camera source USD: {source_robot_usd}")
    cache = UsdGeom.XformCache()
    camera_specs = [
        ("/genie/head_link3/head_front_Camera", "head_link3", "head_rgb"),
    ]
    if camera_profile == "hardware":
        camera_specs.extend([
            ("/genie/head_link3/head_left_fisheye", "head_link3", "head_fisheye_left"),
            ("/genie/head_link3/head_right_fisheye", "head_link3", "head_fisheye_right"),
        ])
    else:
        camera_specs.extend([
            ("/genie/gripper_l_base_link/Left_Camera", "gripper_l_base_link", "hand_left"),
            ("/genie/gripper_r_base_link/Right_Camera", "gripper_r_base_link", "hand_right"),
        ])

    cameras = []
    for camera_path, parent_name, camera_name in camera_specs:
        prim = stage.GetPrimAtPath(camera_path)
        parent = stage.GetPrimAtPath(f"/genie/{parent_name}")
        if not prim or not prim.IsA(UsdGeom.Camera) or not parent:
            raise ValueError(f"G2 source USD is missing camera {camera_path} or its mount {parent_name}")
        camera = UsdGeom.Camera(prim)
        transform = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(parent).GetInverse()
        matrix = np.asarray(transform, dtype=np.float64)
        focal_length = float(camera.GetFocalLengthAttr().Get())
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = float(camera.GetVerticalApertureAttr().Get())
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"G2 source camera {camera_path} has an invalid mount transform")
        if min(focal_length, horizontal_aperture, vertical_aperture) <= 0:
            raise ValueError(f"G2 source camera {camera_path} has invalid intrinsics")
        quaternion = Rotation.from_matrix(matrix[:3, :3].T).as_quat(scalar_first=True)
        is_fisheye = camera_name.startswith("head_fisheye_")
        source_size = None
        for projection in (("opencvFisheye", "opencvPinhole") if is_fisheye else ("opencvPinhole",)):
            size_attr = camera.GetPrim().GetAttribute(f"omni:lensdistortion:{projection}:imageSize")
            if size_attr:
                source_size = size_attr.Get()
            if source_size:
                break
        if source_size:
            if len(source_size) != 2 or min(int(source_size[0]), int(source_size[1])) <= 0:
                raise ValueError(f"G2 source camera {camera_path} has invalid native image size")
            source_width, source_height = int(source_size[0]), int(source_size[1])
            height = max(1, round(image_size * source_height / source_width))
        else:
            source_width = source_height = None
            height = max(1, round(image_size * vertical_aperture / horizontal_aperture))
        intrinsic = None
        if not is_fisheye and source_width is not None:
            attributes = [camera.GetPrim().GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}") for key in ("fx", "fy", "cx", "cy")]
            if all(attr and attr.Get() is not None for attr in attributes):
                fx, fy, cx, cy = (float(attr.Get()) for attr in attributes)
                if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
                    raise ValueError(f"G2 source camera {camera_path} has invalid OpenCV calibration")
                focal_length = horizontal_aperture * fx / source_width
                intrinsic = [
                    [fx * image_size / source_width, 0.0, cx * image_size / source_width],
                    [0.0, fy * height / source_height, cy * height / source_height],
                    [0.0, 0.0, 1.0],
                ]
        distortion = None
        if not is_fisheye:
            distortion_keys = (
                "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"
            )
            attributes = [prim.GetAttribute(f"omni:lensdistortion:opencvPinhole:{key}") for key in distortion_keys]
            if any(attribute and attribute.Get() is not None for attribute in attributes):
                distortion = tuple(
                    float(attribute.Get()) if attribute and attribute.Get() is not None else 0.0
                    for attribute in attributes
                )
        if is_fisheye and fisheye_projection == "calibrated":
            if source_width is None:
                raise ValueError(f"G2 source camera {camera_path} has no fisheye image size")
            attrs = [prim.GetAttribute(f"omni:lensdistortion:opencvFisheye:{key}") for key in
                     ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")]
            if not all(attr and attr.Get() is not None for attr in attrs):
                raise ValueError(f"G2 source camera {camera_path} has incomplete fisheye calibration")
            fx, fy, cx, cy, *coefficients = (float(attr.Get()) for attr in attrs)
            cameras.append(FisheyeCameraCfg(
                name=camera_name, width=image_size, height=height, data_types=["rgb"],
                mount_to=robot_name, mount_link=parent_name,
                mount_pos=tuple(float(value) for value in matrix[3, :3]),
                mount_quat=tuple(float(value) for value in quaternion),
                calibration_size=(source_width, source_height), optical_center=(cx, cy),
                focal_length_px=(fx, fy), coefficients=tuple(coefficients),
                max_fov_rad=invertible_fisheye_fov("opencv_fisheye", coefficients, 2 * math.pi),
                calibration_source="geniesim_g2_usd", calibration_source_sha256=_sha256(source_robot_usd),
                max_fov_source="inferred_monotonic",
            ))
            continue
        cameras.append(
            PinholeCameraCfg(
                name=camera_name,
                width=image_size,
                height=height,
                data_types=["rgb"],
                mount_to=robot_name,
                mount_link=parent_name,
                mount_pos=tuple(float(value) for value in matrix[3, :3]),
                mount_quat=tuple(float(value) for value in quaternion),
                focal_length=focal_length,
                horizontal_aperture=horizontal_aperture,
                intrinsic=intrinsic,
                opencv_distortion=distortion,
                distortion_source=(
                    f"geniesim_g2_usd:{_sha256(source_robot_usd)}" if distortion is not None else None
                ),
            )
        )
    return cameras


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_geniesim_environment_hdr(scene_usd: str | None, source_root: Path = _GENIESIM_SOURCE_ROOT) -> Path | None:
    """Resolve the source USD dome-light HDR, rejecting missing or escaping paths."""
    if not scene_usd:
        return None
    source_root = Path(source_root).resolve()
    scene_path = (source_root / scene_usd.lstrip("/\\")).resolve()
    if not scene_path.is_relative_to(source_root) or not scene_path.is_file():
        return None
    try:
        from pxr import Usd, UsdLux

        stage = Usd.Stage.Open(str(scene_path))
    except Exception:
        stage = None
    if stage is not None:
        for prim in stage.Traverse():
            if not prim.IsA(UsdLux.DomeLight):
                continue
            texture = UsdLux.DomeLight(prim).GetTextureFileAttr().Get()
            if not texture:
                continue
            resolved = getattr(texture, "resolvedPath", "")
            image_path = Path(resolved) if resolved else (scene_path.parent / str(texture))
            image_path = image_path.resolve()
            if (
                image_path.suffix.lower() in {".hdr", ".exr"}
                and image_path.is_relative_to(source_root)
                and image_path.is_file()
            ):
                return image_path

    # Keep compatibility with plain-text USD layers when usd-core is absent or
    # the authored light is not represented as a UsdLux.DomeLight prim.
    content = scene_path.read_text(encoding="utf-8", errors="replace")
    for reference in re.findall(r"(?:inputs:)?texture:file\s*=\s*@([^@]+)@", content):
        if Path(reference).suffix.lower() not in {".hdr", ".exr"}:
            continue
        image_path = (scene_path.parent / reference).resolve()
        if image_path.is_relative_to(source_root) and image_path.is_file():
            return image_path
    return None


def _source_diffuse_material(
    binding_prim, source_root: Path, scene_directory: Path
) -> tuple[Path | None, tuple[float, float, float]]:
    """Resolve the diffuse texture/color used by one source USD material binding."""
    from pxr import Usd, UsdShade

    material, _ = UsdShade.MaterialBindingAPI(binding_prim).ComputeBoundMaterial()
    if not material:
        return None, (0.7, 0.7, 0.7)
    shaders = [prim for prim in Usd.PrimRange(material.GetPrim()) if prim.IsA(UsdShade.Shader)]
    color = (0.7, 0.7, 0.7)
    for shader_prim in shaders:
        shader = UsdShade.Shader(shader_prim)
        color_input = shader.GetInput("diffuse_color_constant")
        color_value = color_input.Get() if color_input else None
        color = tuple(float(value) for value in color_value[:3]) if color_value else (0.7, 0.7, 0.7)
        texture_input = shader.GetInput("diffuse_texture")
        texture_value = texture_input.Get() if texture_input else None
        if not texture_value:
            continue
        resolved = getattr(texture_value, "resolvedPath", "")
        texture_path = Path(resolved) if resolved else Path(texture_value.path)
        if not texture_path.is_absolute():
            scene_relative = (scene_directory / texture_path).resolve()
            texture_path = scene_relative if scene_relative.is_file() else source_root / texture_path
        texture_path = texture_path.resolve()
        if not texture_path.is_relative_to(source_root) or not texture_path.is_file():
            raise FileNotFoundError(f"Missing or unsafe Genie Sim environment texture: {texture_value.path}")
        return texture_path, color
    return None, color


def _source_mesh_material_groups(mesh_prim, source_root: Path, scene_directory: Path, face_count: int) -> list[dict]:
    """Group USD mesh faces by material subset, retaining the authored color textures."""
    from pxr import UsdGeom

    groups: list[dict] = []
    assigned = np.zeros(face_count, dtype=bool)

    def add_group(selected: np.ndarray, binding_prim) -> None:
        texture_path, color = _source_diffuse_material(binding_prim, source_root, scene_directory)
        if not len(selected):
            return
        groups.append({"face_indices": selected, "texture_path": texture_path, "color": color})

    for subset_prim in mesh_prim.GetChildren():
        if not subset_prim.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(subset_prim)
        if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
            continue
        selected = np.asarray(subset.GetIndicesAttr().Get() or [], dtype=np.int64)
        if (
            not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= face_count)
            or len(np.unique(selected)) != len(selected)
            or np.any(assigned[selected])
        ):
            raise ValueError(f"Invalid or overlapping Genie Sim USD material subset: {subset_prim.GetPath()}")
        assigned[selected] = True
        add_group(selected, subset_prim)
    unassigned = np.flatnonzero(~assigned)
    if len(unassigned):
        add_group(unassigned, mesh_prim)
    return groups


def _source_mesh_triangles(mesh_prim) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Triangulate a USD mesh and preserve its face-varying or vertex UVs."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
    if len(points) < 3 or not len(counts) or np.any(counts < 3) or int(counts.sum()) != len(indices):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), None
    triangles: list[tuple[int, int, int]] = []
    triangle_faces: list[int] = []
    corner_indices: list[tuple[int, int, int]] = []
    offset = 0
    for face_index, count in enumerate(counts):
        face = indices[offset : offset + count]
        for corner in range(1, count - 1):
            triangles.append((int(face[0]), int(face[corner]), int(face[corner + 1])))
            corner_indices.append((offset, offset + corner, offset + corner + 1))
            triangle_faces.append(face_index)
        offset += int(count)
    uv_primvar = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
    if not uv_primvar:
        return points, np.asarray(triangles, dtype=np.int64), None
    flattened_uvs = uv_primvar.ComputeFlattened()
    uv_values = np.asarray(flattened_uvs if flattened_uvs is not None else [], dtype=np.float64)
    if not len(uv_values):
        return points, np.asarray(triangles, dtype=np.int64), None
    interpolation = uv_primvar.GetInterpolation()
    if interpolation == UsdGeom.Tokens.faceVarying:
        uv_indices = np.asarray(corner_indices, dtype=np.int64)
    elif interpolation == UsdGeom.Tokens.vertex:
        uv_indices = np.asarray(triangles, dtype=np.int64)
    elif interpolation == UsdGeom.Tokens.constant and len(uv_values) == 1:
        uv_indices = np.zeros((len(triangles), 3), dtype=np.int64)
    else:
        return points, np.asarray(triangles, dtype=np.int64), None
    if int(uv_indices.max(initial=-1)) >= len(uv_values):
        raise ValueError(f"Invalid Genie Sim USD UV indexing on {mesh_prim.GetPath()}")
    return points, np.asarray(triangles, dtype=np.int64), (uv_values, uv_indices, np.asarray(triangle_faces))


def _simplify_environment_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uv_values: np.ndarray | None,
    uv_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, bool]:
    """Reduce high-detail static scenery, carrying diffuse UVs to retained vertices."""
    if len(triangles) <= _SOURCE_ENVIRONMENT_MAX_FACES:
        return vertices, triangles, uv_values, uv_indices, False
    try:
        import fast_simplification
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Simplifying Genie Sim background meshes requires fast-simplification; "
            "install the RoboVerse icl-geniesim extra"
        ) from exc

    used_vertices = np.unique(triangles)
    source_vertices = vertices[used_vertices]
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices))
    source_triangles = remap[triangles].astype(np.int32, copy=False)
    simplified_vertices, simplified_triangles = fast_simplification.simplify(
        source_vertices,
        source_triangles,
        target_count=_SOURCE_ENVIRONMENT_MAX_FACES,
        agg=3.0,
    )
    if len(simplified_triangles) >= len(triangles):
        return vertices, triangles, uv_values, uv_indices, False
    simplified_triangles = simplified_triangles.astype(np.int64)
    if uv_values is None or uv_indices is None:
        return simplified_vertices, simplified_triangles, None, None, True

    uv_sum = np.zeros((len(vertices), 2), dtype=np.float64)
    uv_count = np.zeros(len(vertices), dtype=np.int64)
    np.add.at(uv_sum, triangles.reshape(-1), uv_values[uv_indices].reshape(-1, 2))
    np.add.at(uv_count, triangles.reshape(-1), 1)
    valid = uv_count > 0
    uv_sum[valid] /= uv_count[valid, None]
    nearest_vertex = cKDTree(source_vertices).query(simplified_vertices, k=1)[1]
    simplified_uv = uv_sum[used_vertices[nearest_vertex]]
    return simplified_vertices, simplified_triangles, simplified_uv, simplified_triangles, True


def _copy_environment_texture(source: Path, destination: Path) -> None:
    """Convert a diffuse albedo to MuJoCo-compatible PNG and cap its resolution."""
    from PIL import Image

    with Image.open(source) as source_image:
        image = source_image.convert("RGB") if source_image.mode not in {"RGB", "RGBA"} else source_image.copy()
    image.thumbnail(
        (_SOURCE_ENVIRONMENT_MAX_TEXTURE_SIZE, _SOURCE_ENVIRONMENT_MAX_TEXTURE_SIZE),
        Image.Resampling.LANCZOS,
    )
    image.save(destination, format="PNG", optimize=True)


def _source_environment_geometry_assets(
    cache_root: Path,
    scene_usd: str | None,
    source_root: Path = _GENIESIM_SOURCE_ROOT,
    exclude_asset_ids: Sequence[str] = (),
) -> tuple[Path, Path, list[str]] | None:
    """Convert static source USD scenery into shared textured URDF/MJCF visuals."""
    if not scene_usd:
        return None
    source_root = Path(source_root).resolve()
    scene_path = (source_root / scene_usd.lstrip("/\\")).resolve()
    if not scene_path.is_relative_to(source_root) or not scene_path.is_file():
        return None
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        return None

    xform_cache = UsdGeom.XformCache()
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    excluded = set(exclude_asset_ids)
    mesh_groups: list[dict] = []
    texture_paths: set[Path] = set()
    simplified_mesh_groups = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        path_parts = set(prim_path.split("/"))
        if (
            not prim.IsA(UsdGeom.Mesh)
            or "Collision" in prim_path
            or "/World/objects/" in prim_path
            or excluded.intersection(path_parts)
            or UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        ):
            continue
        mesh = UsdGeom.Mesh(prim)
        points, triangles, uv_data = _source_mesh_triangles(prim)
        if not len(triangles):
            continue
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        groups = _source_mesh_material_groups(prim, source_root, scene_path.parent, len(counts))
        world_transform = np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        vertices = (points @ world_transform[:3, :3] + world_transform[3, :3]) * meters_per_unit
        triangle_faces = uv_data[2] if uv_data is not None else None
        for group in groups:
            selected = np.isin(triangle_faces, group["face_indices"]) if triangle_faces is not None else None
            if selected is None:
                # Without UV metadata, a triangulated USD polygon fan is still
                # grouped in face order. Build triangle source-face indices once.
                face_map = np.repeat(np.arange(len(counts)), np.maximum(counts - 2, 0))
                selected = np.isin(face_map, group["face_indices"])
            face_indices = np.flatnonzero(selected)
            if not len(face_indices):
                continue
            if group["texture_path"] is not None and uv_data is not None:
                texture_paths.add(group["texture_path"])
            group_uv_values = uv_data[0] if group["texture_path"] is not None and uv_data is not None else None
            group_uv_indices = (
                uv_data[1][face_indices] if group["texture_path"] is not None and uv_data is not None else None
            )
            group_vertices, group_triangles = vertices, triangles[face_indices]
            group_vertices, group_triangles, group_uv_values, group_uv_indices, was_simplified = (
                _simplify_environment_mesh(group_vertices, group_triangles, group_uv_values, group_uv_indices)
            )
            simplified_mesh_groups += int(was_simplified)
            mesh_groups.append({
                "prim_path": prim_path,
                "vertices": group_vertices,
                "triangles": group_triangles,
                "uv_values": group_uv_values,
                "uv_indices": group_uv_indices,
                "texture_path": group["texture_path"] if uv_data is not None else None,
                "color": group["color"],
            })
    if not mesh_groups:
        return None

    source_hash = _sha256(scene_path)
    texture_hashes = {path: _sha256(path) for path in texture_paths}
    digest = hashlib.sha256(source_hash.encode("ascii"))
    for path, file_hash in sorted(texture_hashes.items(), key=lambda item: str(item[0])):
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    environment_hash = digest.hexdigest()
    directory = Path(cache_root).resolve() / f"geniesim_source_environment_v7_{environment_hash[:12]}"
    urdf_path, mjcf_path = directory / "environment.urdf", directory / "environment.xml"
    obj_paths = [directory / f"mesh_{index:04d}.obj" for index in range(len(mesh_groups))]
    texture_names = {path: f"texture_{texture_hashes[path][:12]}.png" for path in texture_paths}
    required = (urdf_path, mjcf_path, directory / "environment.mtl", *obj_paths)
    required += tuple(directory / name for name in texture_names.values())
    if all(path.is_file() for path in required):
        return urdf_path, mjcf_path, [str(path) for path in required[2:]]
    if directory.exists():
        raise FileExistsError(f"Incomplete cached Genie Sim environment geometry: {directory}")

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geniesim-source-environment-", dir=directory.parent) as temporary:
        staged = Path(temporary) / directory.name
        staged.mkdir()
        for source, name in texture_names.items():
            _copy_environment_texture(source, staged / name)
        mtl_lines: list[str] = []
        urdf_visuals: list[str] = []
        mjcf_assets: list[str] = []
        mjcf_geoms: list[str] = []
        declared_textures: set[str] = set()
        for index, item in enumerate(mesh_groups):
            mesh_name = f"mesh_{index:04d}"
            material_name = f"material_{index:04d}"
            texture_path = item["texture_path"]
            color = tuple(float(value) for value in item["color"])
            material_color = (1.0, 1.0, 1.0) if texture_path is not None else color
            _write_source_obj(
                staged / f"{mesh_name}.obj",
                item["vertices"],
                item["triangles"],
                item["uv_values"],
                item["uv_indices"],
                material_name,
            )
            mtl_lines.extend([
                f"newmtl {material_name}",
                "Kd " + " ".join(f"{value:.7g}" for value in material_color),
                "d 1",
                f"map_Kd {texture_names[texture_path]}" if texture_path is not None else "",
                "",
            ])
            filename = (directory / f"{mesh_name}.obj").resolve().as_posix()
            rgba = " ".join(f"{value:.7g}" for value in (*material_color, 1.0))
            texture_tag = (
                f'<texture filename="{(directory / texture_names[texture_path]).resolve().as_posix()}"/>'
                if texture_path is not None
                else ""
            )
            urdf_visuals.append(
                f'<visual><geometry><mesh filename="{filename}"/></geometry>'
                f'<material name="{material_name}"><color rgba="{rgba}"/>{texture_tag}</material></visual>'
            )
            mjcf_assets.append(f'<mesh name="{mesh_name}" file="{filename}" inertia="shell"/>')
            if texture_path is not None:
                texture_name = f"texture_{texture_hashes[texture_path][:12]}"
                texture_file = (directory / texture_names[texture_path]).resolve().as_posix()
                if texture_name not in declared_textures:
                    mjcf_assets.append(f'<texture name="{texture_name}" type="2d" file="{texture_file}"/>')
                    declared_textures.add(texture_name)
                mjcf_assets.append(f'<material name="{material_name}" texture="{texture_name}" rgba="{rgba}"/>')
                material_ref = f' material="{material_name}"'
            else:
                material_ref = f' rgba="{rgba}"'
            mjcf_geoms.append(
                f'<geom name="{mesh_name}" type="mesh" mesh="{mesh_name}"{material_ref} '
                'contype="0" conaffinity="0" group="1"/>'
            )
        (staged / "environment.mtl").write_text("\n".join(mtl_lines), encoding="utf-8")
        (staged / "environment.urdf").write_text(
            '<?xml version="1.0"?><robot name="geniesim_source_environment"><link name="geniesim_source_environment">'
            '<inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" '
            f'izz="0.1"/></inertial>{"".join(urdf_visuals)}</link></robot>\n',
            encoding="utf-8",
        )
        (staged / "environment.xml").write_text(
            '<mujoco model="geniesim_source_environment"><compiler inertiafromgeom="false"/><asset>'
            f"{''.join(mjcf_assets)}</asset><worldbody>{''.join(mjcf_geoms)}</worldbody></mujoco>\n",
            encoding="utf-8",
        )
        (staged / "provenance.json").write_text(
            json.dumps(
                {
                    "format": "roboverse.icl.geniesim_source_environment_geometry.v7",
                    "source_scene_usd": scene_usd,
                    "source_scene_sha256": source_hash,
                    "source_texture_sha256": {
                        path.relative_to(source_root).as_posix(): digest for path, digest in texture_hashes.items()
                    },
                    "included_mesh_material_groups": len(mesh_groups),
                    "included_source_meshes": len({item["prim_path"] for item in mesh_groups}),
                    "simplified_mesh_material_groups": simplified_mesh_groups,
                    "maximum_simplified_face_count": _SOURCE_ENVIRONMENT_MAX_FACES,
                    "maximum_texture_dimension": _SOURCE_ENVIRONMENT_MAX_TEXTURE_SIZE,
                    "excluded_dynamic_background_path": "/World/objects",
                    "limitations": "static USD mesh scenery only; source NuRec/Gaussian-splat geometry is not converted",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), directory)
    resource_paths = [directory / "environment.mtl", *obj_paths]
    resource_paths.extend(directory / name for name in sorted(texture_names.values()))
    return urdf_path, mjcf_path, [str(path) for path in resource_paths]


def _write_source_obj(
    path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    uv_values: np.ndarray | None,
    uv_indices: np.ndarray | None,
    material_name: str,
) -> None:
    """Serialize one textured USD material group as a renderer-portable OBJ."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"mtllib environment.mtl\nusemtl {material_name}\n")
        for vertex in vertices:
            stream.write("v {:.9g} {:.9g} {:.9g}\n".format(*vertex))
        if uv_values is not None:
            for uv in uv_values:
                stream.write("vt {:.9g} {:.9g}\n".format(*uv))
        if uv_values is not None and uv_indices is not None:
            for face, uv_face in zip(triangles, uv_indices, strict=True):
                stream.write(
                    "f "
                    + " ".join(f"{int(vertex) + 1}/{int(uv) + 1}" for vertex, uv in zip(face, uv_face, strict=True))
                    + "\n"
                )
        else:
            for face in triangles:
                stream.write("f " + " ".join(str(int(vertex) + 1) for vertex in face) + "\n")


def _studio_panorama_assets(cache_root: Path, panorama_path: Path) -> tuple[Path, Path, list[str]]:
    """Build a visual-only UV sphere from the selected Genie Sim dome HDR."""
    panorama_path = Path(panorama_path).resolve()
    if panorama_path.suffix.lower() != ".hdr" or not panorama_path.is_file():
        raise ValueError(f"Expected a local Genie Sim Radiance HDR environment map: {panorama_path}")
    panorama_hash = _sha256(panorama_path)
    directory = Path(cache_root).resolve() / f"geniesim_studio_backdrop_v8_{panorama_hash[:12]}"
    urdf_path, mjcf_path = directory / "backdrop.urdf", directory / "backdrop.xml"
    mesh_path, mtl_path, texture_path = (
        directory / "panorama.obj",
        directory / "panorama.mtl",
        directory / "panorama.png",
    )
    required = (urdf_path, mjcf_path, mesh_path, mtl_path, texture_path)
    if all(path.is_file() for path in required):
        return urdf_path, mjcf_path, [str(mesh_path), str(mtl_path), str(texture_path)]
    if directory.exists():
        raise FileExistsError(f"Incomplete cached Genie Sim panorama backdrop: {directory}")

    import cv2
    from PIL import Image

    pixels = cv2.imread(str(panorama_path), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError(f"Could not decode Genie Sim HDR panorama: {panorama_path}")
    linear_rgb = np.maximum(pixels[..., :3][..., ::-1].astype(np.float32), 0.0)
    if linear_rgb.shape[1] > _STUDIO_PANORAMA_MAX_WIDTH:
        scale = _STUDIO_PANORAMA_MAX_WIDTH / linear_rgb.shape[1]
        size = (_STUDIO_PANORAMA_MAX_WIDTH, max(1, round(linear_rgb.shape[0] * scale)))
        linear_rgb = cv2.resize(linear_rgb, size, interpolation=cv2.INTER_AREA)
    reinhard = linear_rgb / (1.0 + linear_rgb)
    srgb = np.where(reinhard <= 0.0031308, 12.92 * reinhard, 1.055 * reinhard ** (1.0 / 2.4) - 0.055)
    rgb8 = np.asarray(np.clip(srgb * 255.0 + 0.5, 0, 255), dtype=np.uint8)

    # Keep the shell within the practical far-clip range of the supported
    # renderers. Genie Sim's `default` background layer rotates its dome -90°
    # around world Z; rotating the sphere vertices preserves continuous UVs.
    longitude_steps, latitude_steps, radius = 96, 48, 12.0
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    ring_indices: list[list[int]] = []
    for latitude_index in range(1, latitude_steps):
        latitude = -math.pi / 2 + math.pi * latitude_index / latitude_steps
        ring = []
        for longitude_index in range(longitude_steps + 1):
            longitude = 2 * math.pi * longitude_index / longitude_steps
            ring.append(len(vertices) + 1)
            x = radius * math.cos(latitude) * math.cos(longitude)
            y = radius * math.cos(latitude) * math.sin(longitude)
            vertices.append((y, -x, radius * math.sin(latitude)))
            uvs.append((longitude_index / longitude_steps, latitude_index / latitude_steps))
        ring_indices.append(ring)
    bottom_index = len(vertices) + 1
    vertices.append((0.0, 0.0, -radius))
    uvs.append((0.5, 0.0))
    top_index = len(vertices) + 1
    vertices.append((0.0, 0.0, radius))
    uvs.append((0.5, 1.0))
    faces: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    first_ring = ring_indices[0]
    last_ring = ring_indices[-1]
    for index in range(longitude_steps):
        faces.append((
            (bottom_index, bottom_index),
            (first_ring[index], first_ring[index]),
            (first_ring[index + 1], first_ring[index + 1]),
        ))
        faces.append((
            (last_ring[index], last_ring[index]),
            (top_index, top_index),
            (last_ring[index + 1], last_ring[index + 1]),
        ))
    for lower, upper in pairwise(ring_indices):
        for index in range(longitude_steps):
            a, b, c, d = lower[index], lower[index + 1], upper[index + 1], upper[index]
            faces.extend((((a, a), (d, d), (c, c)), ((a, a), (c, c), (b, b))))

    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geniesim-panorama-", dir=directory.parent) as temporary:
        staged = Path(temporary) / directory.name
        staged.mkdir()
        Image.fromarray(rgb8).save(staged / "panorama.png", optimize=True)
        with (staged / "panorama.obj").open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("mtllib panorama.mtl\nusemtl studio_panorama\n")
            for vertex in vertices:
                stream.write("v {:.9g} {:.9g} {:.9g}\n".format(*vertex))
            for uv in uvs:
                stream.write("vt {:.9g} {:.9g}\n".format(*uv))
            for face in faces:
                stream.write("f " + " ".join(f"{vertex}/{uv}" for vertex, uv in face) + "\n")
        (staged / "panorama.mtl").write_text(
            "newmtl studio_panorama\nKd 1 1 1\nmap_Kd panorama.png\n", encoding="utf-8"
        )
        absolute_mesh = (directory / "panorama.obj").resolve().as_posix()
        absolute_texture = (directory / "panorama.png").resolve().as_posix()
        (staged / "backdrop.urdf").write_text(
            '<?xml version="1.0"?><robot name="geniesim_studio_panorama"><link name="geniesim_studio_panorama">'
            '<inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" '
            'izz="0.1"/></inertial><visual><geometry><mesh filename="'
            f"{absolute_mesh}"
            '"/></geometry><material name="studio_panorama"><color rgba="1 1 1 1"/><texture filename="'
            f"{absolute_texture}"
            '"/></material></visual></link></robot>\n',
            encoding="utf-8",
        )
        (staged / "backdrop.xml").write_text(
            '<mujoco model="geniesim_studio_panorama"><compiler inertiafromgeom="false"/><asset>'
            f'<mesh name="studio_panorama_mesh" file="{absolute_mesh}"/>'
            f'<texture name="studio_panorama_texture" type="2d" file="{absolute_texture}"/>'
            '<material name="studio_panorama_material" texture="studio_panorama_texture" rgba="1 1 1 1" '
            'specular="0" shininess="0"/></asset><worldbody><geom name="studio_panorama" '
            'type="mesh" mesh="studio_panorama_mesh" material="studio_panorama_material" '
            'contype="0" conaffinity="0" group="1"/></worldbody></mujoco>\n',
            encoding="utf-8",
        )
        (staged / "provenance.json").write_text(
            json.dumps(
                {
                    "format": "roboverse.icl.geniesim_studio_hdr_backdrop.v2",
                    "source_hdr_sha256": panorama_hash,
                    "tone_mapping": "Reinhard followed by sRGB encoding",
                    "projection": "equirectangular texture on an inward-facing visual-only sphere",
                    "limitations": "does not reproduce NuRec/Gaussian-splat geometry, parallax, or occlusion",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), directory)
    return urdf_path, mjcf_path, [str(mesh_path), str(mtl_path), str(texture_path)]


def _studio_backdrop_assets(
    cache_root: Path,
    *,
    extent: float = 10.0,
    height: float = 6.0,
    panorama_path: Path | None = None,
) -> tuple[Path, Path, list[str]]:
    """Create a shared, visual-only studio backdrop with room-facing mesh normals."""
    if panorama_path is not None:
        return _studio_panorama_assets(cache_root, panorama_path)
    directory = Path(cache_root).resolve() / "geniesim_studio_backdrop_v5"
    urdf_path, mjcf_path = directory / "backdrop.urdf", directory / "backdrop.xml"
    mesh_paths = [directory / f"wall_{name}.obj" for name in ("x_positive", "x_negative", "y_positive", "y_negative")]
    required = [urdf_path, mjcf_path, *mesh_paths]
    if all(path.is_file() for path in required):
        return urdf_path, mjcf_path, [str(path) for path in mesh_paths]
    if directory.exists():
        raise FileExistsError(f"Incomplete cached Genie Sim studio backdrop: {directory}")

    wall_vertices = (
        ((extent, -extent, 0), (extent, -extent, height), (extent, extent, height), (extent, extent, 0)),
        ((-extent, -extent, 0), (-extent, extent, 0), (-extent, extent, height), (-extent, -extent, height)),
        ((-extent, extent, 0), (extent, extent, 0), (extent, extent, height), (-extent, extent, height)),
        ((-extent, -extent, 0), (-extent, -extent, height), (extent, -extent, height), (extent, -extent, 0)),
    )
    names = ("x_positive", "x_negative", "y_positive", "y_negative")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="geniesim-backdrop-", dir=directory.parent) as temporary:
        staged = Path(temporary) / directory.name
        staged.mkdir()
        for name, vertices in zip(names, wall_vertices, strict=True):
            with (staged / f"wall_{name}.obj").open("w", encoding="utf-8", newline="\n") as stream:
                for vertex in vertices:
                    stream.write("v {:.9g} {:.9g} {:.9g}\n".format(*vertex))
                stream.write("f 1 2 3\nf 1 3 4\n")

        color = (0.82, 0.82, 0.80)
        urdf_visuals = []
        mjcf_assets = []
        mjcf_geoms = []
        for name in names:
            filename = (directory / f"wall_{name}.obj").resolve().as_posix()
            rgb = " ".join(f"{value:.6g}" for value in color)
            urdf_visuals.append(
                f'<visual><geometry><mesh filename="{filename}"/></geometry>'
                f'<material name="studio_gray"><color rgba="{rgb} 1"/></material></visual>'
            )
            # These are intentionally planar visual meshes. MuJoCo requires an
            # explicit shell inertia for zero-volume mesh assets, even though
            # the geoms are world-fixed and have collisions disabled.
            mjcf_assets.append(f'<mesh name="wall_{name}" file="{filename}" inertia="shell"/>')
            mjcf_geoms.append(
                f'<geom name="wall_{name}" type="mesh" mesh="wall_{name}" rgba="{rgb} 1" '
                'contype="0" conaffinity="0" group="1"/>'
            )
        (staged / "backdrop.urdf").write_text(
            '<?xml version="1.0"?><robot name="geniesim_studio_backdrop"><link name="geniesim_studio_backdrop">'
            '<inertial><mass value="1"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" '
            f'izz="0.1"/></inertial>{"".join(urdf_visuals)}</link></robot>\n',
            encoding="utf-8",
        )
        (staged / "backdrop.xml").write_text(
            '<mujoco model="geniesim_studio_backdrop"><compiler inertiafromgeom="false"/><asset>'
            f"{''.join(mjcf_assets)}</asset><worldbody>{''.join(mjcf_geoms)}</worldbody></mujoco>\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), directory)
    return urdf_path, mjcf_path, [str(path) for path in mesh_paths]


def _source_marks_fixed(name: str, asset_id: str, entry: dict, fixed_names: set[str]) -> bool:
    if name in fixed_names or asset_id in fixed_names:
        return True
    description = entry.get("description", {})
    if not isinstance(description, dict):
        return False
    labels = [name, asset_id]
    for key in ("object_category", "semantic_name"):
        value = description.get(key, [])
        if isinstance(value, str):
            labels.append(value)
        elif isinstance(value, list):
            labels.extend(value)
    static_surface_terms = {
        "table",
        "desk",
        "counter",
        "countertop",
        "workbench",
        "floor",
        "ground",
        "cabinet",
        "drawer",
    }
    words = {
        word for label in labels if isinstance(label, str) for word in re.split(r"[^a-z0-9]+", label.lower()) if word
    }
    return bool(words & static_surface_terms)


def _instance_scene_layout(instance: dict) -> dict[str, dict]:
    """Resolve an imported source scene snapshot while preserving its metadata descriptions."""
    metadata_layout = instance.get("scene_info", {}).get("layout")
    if not isinstance(metadata_layout, dict):
        raise ValueError("Genie Sim instance has no object layout")
    snapshot = instance.get("source_scene")
    if snapshot is None:
        return metadata_layout
    if not isinstance(snapshot, dict) or snapshot.get("format") != _SOURCE_SCENE_FORMAT:
        raise ValueError("Unsupported Genie Sim source-scene snapshot")
    source_file = instance.get("source_files", {}).get("scene.usda", {})
    scene_hash = snapshot.get("sha256")
    scene_path = source_file.get("path")
    if (
        not isinstance(scene_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", scene_hash)
        or scene_hash != source_file.get("sha256")
        or not isinstance(scene_path, str)
        or PurePosixPath(scene_path).is_absolute()
        or ".." in PurePosixPath(scene_path).parts
    ):
        raise ValueError("Genie Sim source-scene snapshot does not match the recorded source file")
    objects = snapshot.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Genie Sim source-scene snapshot has no object map")

    result = {}
    for name, item in objects.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name) or not isinstance(item, dict):
            raise ValueError(f"Invalid source-scene object {name!r}")
        if not isinstance(item.get("active"), bool):
            raise ValueError(f"Source-scene object {name!r} has no valid active flag")
        if not item["active"]:
            continue
        asset_id = item.get("asset_id")
        asset_path = item.get("asset_path")
        prim_path = item.get("prim_path")
        asset_parts = PurePosixPath(asset_path).parts if isinstance(asset_path, str) else ()
        if (
            not isinstance(asset_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id)
            or not isinstance(asset_path, str)
            or not asset_path.startswith("/geniesim_assets/objects/")
            or ".." in asset_parts
            or len(asset_parts) < 5
            or asset_parts[-2] != asset_id
            or asset_parts[-1] not in {"Aligned.usda", "Aligned.usd"}
            or not isinstance(prim_path, str)
            or not prim_path.startswith("/World/Objects/")
            or PurePosixPath(prim_path).name != name
        ):
            raise ValueError(f"Invalid source-scene payload for {name!r}")
        scale = np.asarray(item.get("scale", []), dtype=np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f"Invalid source-scene scale for {name!r}")
        if not np.allclose(scale, scale[0], rtol=0.0, atol=1e-5):
            raise NotImplementedError(
                f"Source scene {instance.get('task')}/{instance.get('instance_id')} object {name!r} "
                f"uses nonuniform scale {scale.tolist()}; cross-backend scaled asset loading is not yet supported"
            )
        uniform_scale = 1.0 if np.allclose(scale, 1.0, rtol=0.0, atol=1e-5) else float(scale.mean())
        old = metadata_layout.get(name, {})
        if not isinstance(old, dict):
            raise ValueError(f"Invalid metadata for source-scene object {name!r}")
        entry = deepcopy(old)
        entry.update({"usd": asset_id, "xyz": item.get("xyz"), "xyzw": item.get("xyzw")})
        entry["_source_scene_payload"] = {
            "scene_path": scene_path,
            "scene_sha256": scene_hash,
            "prim_path": prim_path,
            "asset_path": asset_path,
            "metadata_asset_id": old.get("usd"),
            "resolved_asset_id": asset_id,
            "uniform_scale": uniform_scale,
        }
        result[name] = entry
    return result


def _payload_asset_root(assets_root: Path, entry: Mapping) -> Path:
    """Choose the prepared object layer actually referenced by the source scene."""
    payload = entry.get("_source_scene_payload")
    if payload and PurePosixPath(payload["asset_path"]).name == "Aligned.usd":
        return Path(assets_root) / "_source_layers" / "Aligned_usd"
    return Path(assets_root)


def _validate_payload_provenance(entry: Mapping, provenance: Mapping) -> None:
    """Reject a prepared asset from a different source path or USD layer."""
    payload = entry.get("_source_scene_payload")
    if not payload:
        return
    asset_path = PurePosixPath(payload["asset_path"])
    layer = asset_path.name
    asset_id = payload["resolved_asset_id"]
    expected = "/geniesim_assets/" + str(provenance.get("source_asset_directory", "")) + "/" + layer
    source_hash = provenance.get("source_files_sha256", {}).get(layer)
    historical = HISTORICAL_ASSETS.get(asset_id)
    if (
        str(asset_path) != expected
        or provenance.get("asset_id") != asset_id
        or provenance.get("source_revision") != source_revision(asset_id)
        or provenance.get("source_layer", "Aligned.usda") != layer
        or not isinstance(source_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
        or (historical and (
            provenance.get("source_asset_directory") != historical["directory"]
            or any(provenance.get("source_files_sha256", {}).get(name) != digest
                   for name, digest in historical["sha256"].items())
        ))
    ):
        raise ValueError(f"Prepared asset does not match source-scene payload {asset_path}")


def geniesim_object_bounds(instance: dict, assets_root: Path) -> dict[str, np.ndarray]:
    """Load prepared local-frame mesh bounds for each rigid object in one source instance."""
    layout = _instance_scene_layout(instance)
    if not isinstance(layout, dict) or not layout:
        raise ValueError("Genie Sim instance has no object layout")
    result = {}
    for name, entry in layout.items():
        asset_id = entry.get("usd") if isinstance(entry, dict) else None
        if not isinstance(asset_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id):
            raise ValueError(f"Genie Sim scene object {name!r} has an invalid asset ID")
        asset_root = _payload_asset_root(assets_root, entry)
        resolved_asset_id, _ = _resolve_scene_asset(asset_root, asset_id, entry)
        provenance_path = asset_root / resolved_asset_id / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Prepare Genie Sim asset {asset_id!r} before loading its bounds")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_payload_provenance(entry, provenance)
        if provenance.get("format") != _OBJECT_FORMAT or provenance.get("asset_id") != resolved_asset_id:
            raise ValueError(f"Prepared object provenance does not match source asset {asset_id!r}")
        bounds = np.asarray(provenance.get("bounds_m"), dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
            raise ValueError(f"Prepared asset {resolved_asset_id!r} has invalid local geometry bounds")
        result[name] = bounds * float(entry.get("_source_scene_payload", {}).get("uniform_scale", 1.0))
    return result


def geniesim_checker_subframes(instance: dict, assets_root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Resolve source object child frames used by checker graphs from verified asset metadata."""
    references: set[tuple[str, str]] = set()

    def collect(value) -> None:
        if isinstance(value, str):
            for match in re.finditer(r"/(?:Workspace|World)/[Oo]bjects/([A-Za-z0-9_-]+)/([A-Za-z0-9_/.-]+)", value):
                references.add((match.group(1), match.group(2)))
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(instance.get("checker_graphs", {}))
    layout = _instance_scene_layout(instance)
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, child in sorted(references):
        if name not in layout:
            raise ValueError(f"Checker references source subframe of absent scene object {name!r}")
        entry = layout[name]
        asset_id = entry["usd"]
        asset_root = _payload_asset_root(assets_root, entry)
        resolved, _ = _resolve_scene_asset(asset_root, asset_id, entry)
        provenance = json.loads((asset_root / resolved / "provenance.json").read_text(encoding="utf-8"))
        _validate_payload_provenance(entry, provenance)
        value = provenance.get("source_subframes", {}).get(child)
        if value is None:
            raise ValueError(
                f"Prepared asset {resolved!r} lacks verified source subframe {child!r}; "
                "rerun prepare_geniesim_objects.py with --refresh-subframes"
            )
        transform = np.asarray(value, dtype=np.float64)
        rotation = transform[:3, :3] if transform.shape == (4, 4) else None
        if (rotation is None or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
            raise ValueError(f"Prepared asset {resolved!r} has an invalid source subframe {child!r}")
        transform = transform.copy()
        transform[:3, 3] *= float(entry.get("_source_scene_payload", {}).get("uniform_scale", 1.0))
        result.setdefault(name, {})[child] = transform
    return result


def geniesim_checker_background_objects(
    instance: dict, scene_usd: str | None, source_root: Path = _GENIESIM_SOURCE_ROOT
) -> dict:
    """Read referenced fixed background world bounds from the original environment USD.

    These observations are for checker geometry only. They do not create
    physical background collision shapes in the portable simulation.
    """
    references = set(re.findall(r"/World/background/[A-Za-z0-9_/-]+", json.dumps(instance.get("checker_graphs", {}))))
    if not references:
        return {}
    if not scene_usd:
        raise ValueError("Checker references background objects but no source environment USD was configured")
    root = Path(source_root).resolve()
    path = (root / scene_usd.lstrip("/\\")).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Checker needs the source background USD: {path}")
    from pxr import Usd, UsdGeom

    from roboverse_pack.tasks.icl.geniesim_checker import ObjectObservation

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError(f"Cannot open checker background source: {path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                       UsdGeom.Tokens.proxy])
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    result = {}
    for reference in sorted(references):
        prim = stage.GetPrimAtPath(reference)
        if not prim or not prim.IsActive():
            raise ValueError(f"Source background object {reference!r} is absent or inactive in {path}")
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bounds = np.asarray([box.GetMin(), box.GetMax()], dtype=np.float64) * meters
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError(f"Source background object {reference!r} has invalid world geometry bounds")
        result[reference] = ObjectObservation(pose=np.eye(4), bounds=bounds)
    return result


def _pushpull_references(instance: dict) -> list[str]:
    """Return source object references whose checker predicates require joint state."""
    found: list[str] = []

    def visit(value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "PushPull" and isinstance(child, str):
                    found.append(child.split("|", 1)[0].strip())
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(instance.get("checker_graphs", {}))
    return list(dict.fromkeys(found))


def _source_pose_from_row_matrix(
    matrix: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a USD row-vector transform to a world position and WXYZ quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Source articulated object has an invalid USD transform")
    quaternion = Rotation.from_matrix(matrix[:3, :3].T).as_quat(scalar_first=True)
    return tuple(float(v) for v in matrix[3, :3]), tuple(float(v) for v in quaternion)


def _articulated_object_cfg(
    *,
    object_name: str,
    asset_id: str,
    asset_root: Path,
    wrapper_root: Path,
    position: Sequence[float],
    orientation_wxyz: Sequence[float],
) -> ArticulationObjCfg:
    """Load a converted source articulation with backend-specific portable model files."""
    asset_root = Path(asset_root)
    provenance_path = asset_root / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Prepare articulated Genie Sim asset {asset_id!r} first: {asset_root}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("format") != _ARTICULATED_OBJECT_FORMAT
        or provenance.get("source_revision") != _GENIESIM_ASSET_REVISION
        or provenance.get("asset_id") != asset_id
    ):
        raise ValueError(f"Prepared articulation provenance does not match the pinned Genie Sim asset: {asset_id}")
    instance_root = _articulated_instance_wrapper(asset_root, object_name, wrapper_root)
    urdf_path, mjcf_path = instance_root / "object.urdf", instance_root / "object.xml"
    if not urdf_path.is_file() or not mjcf_path.is_file():
        raise FileNotFoundError(f"Prepared articulation is incomplete: {asset_root}")
    resources = [
        str(path.resolve())
        for path in instance_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".obj", ".mtl", ".png", ".jpg", ".jpeg"}
    ]
    cfg = ArticulationObjCfg(
        name=object_name,
        urdf_path=str(urdf_path.resolve()),
        mjcf_path=str(mjcf_path.resolve()),
        extra_resources=resources,
        default_position=tuple(float(v) for v in position),
        default_orientation=tuple(float(v) for v in orientation_wxyz),
        fix_base_link=bool(provenance.get("root_is_fixed", False)),
        enabled_gravity=True,
    )
    cfg.file_type.update({"mujoco": "mjcf", "genesis": "urdf", "newton": "urdf"})
    return cfg


def _articulated_instance_wrapper(asset_root: Path, instance_name: str, output: Path) -> Path:
    """Namespace a reusable articulated asset for one scene object instance."""
    asset_root, output = Path(asset_root).resolve(), Path(output).resolve()
    provenance_path = asset_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source_hash = _sha256(provenance_path)
    manifest_path = output / "instance.json"
    if output.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("format") == _ARTICULATED_WRAPPER_FORMAT
                and manifest.get("asset_id") == provenance.get("asset_id")
                and manifest.get("instance_name") == instance_name
                and manifest.get("source_provenance_sha256") == source_hash
                and all((output / filename).is_file() for filename in ("object.xml", "object.urdf", "provenance.json"))
            ):
                return output
        except (OSError, json.JSONDecodeError):
            pass
        raise FileExistsError(f"Refusing to overwrite stale articulated scene wrapper: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{instance_name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / instance_name
        staged.mkdir()
        mjcf_tree = ET.parse(asset_root / "object.xml")
        mjcf_root = mjcf_tree.getroot()
        mjcf_root.set("model", instance_name)
        root_bodies = mjcf_root.findall("worldbody/body")
        if len(root_bodies) != 1:
            raise ValueError(f"Expected one root MJCF body in articulated asset {asset_root}")
        root_bodies[0].set("name", instance_name)
        for node in mjcf_root.findall(".//mesh") + mjcf_root.findall(".//texture"):
            filename = node.get("file")
            if filename:
                node.set("file", str((asset_root / filename).resolve()))
        mjcf_tree.write(staged / "object.xml", encoding="utf-8", xml_declaration=True)

        urdf_tree = ET.parse(asset_root / "object.urdf")
        urdf_root = urdf_tree.getroot()
        urdf_root.set("name", instance_name)
        for mesh in urdf_root.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh.set("filename", str((asset_root / filename).resolve()))
        for texture in urdf_root.findall(".//texture"):
            filename = texture.get("filename")
            if filename:
                texture.set("filename", str((asset_root / filename).resolve()))
        urdf_tree.write(staged / "object.urdf", encoding="utf-8", xml_declaration=True)
        shutil.copy2(provenance_path, staged / "provenance.json")
        (staged / "instance.json").write_text(
            json.dumps(
                {
                    "format": _ARTICULATED_WRAPPER_FORMAT,
                    "asset_id": provenance["asset_id"],
                    "instance_name": instance_name,
                    "source_provenance_sha256": source_hash,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), output)
    return output


def _background_articulation_pose(scene_usd: str, asset_id: str, root_link: str) -> tuple[tuple, tuple]:
    """Read a background articulation's authored root link pose from its source USD."""
    source_root = _GENIESIM_SOURCE_ROOT.resolve()
    scene_path = (source_root / scene_usd.lstrip("/\\")).resolve()
    if not scene_path.is_relative_to(source_root) or not scene_path.is_file():
        raise FileNotFoundError(f"Missing pinned source scene USD: {scene_path}")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene_path))
    prim = stage.GetPrimAtPath(f"/World/objects/{asset_id}/entity/{root_link}") if stage else None
    if not prim:
        raise ValueError(f"Source background does not contain articulated root link {asset_id}/{root_link}")
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=np.float64)
    return _source_pose_from_row_matrix(transform)


def _resolve_scene_asset(asset_root: Path, source_asset_id: str, entry: Mapping) -> tuple[str, dict | None]:
    """Resolve an exact source asset, or a narrowly verified compatibility alias."""
    if (Path(asset_root) / source_asset_id / "provenance.json").is_file():
        return source_asset_id, None
    alias = _COMPATIBILITY_ASSET_ALIASES.get(source_asset_id)
    if alias is None:
        return source_asset_id, None

    candidate_id = alias["asset_id"]
    provenance_path = Path(asset_root) / candidate_id / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Source asset {source_asset_id!r} is unavailable and its declared compatibility asset "
            f"{candidate_id!r} is not prepared: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("format") != _OBJECT_FORMAT
        or provenance.get("asset_id") != candidate_id
        or provenance.get("source_revision") != alias["source_revision"]
        or provenance.get("source_files_sha256", {}).get("Aligned.usd") != alias["aligned_usd_sha256"]
    ):
        raise ValueError(f"Compatibility asset {candidate_id!r} does not match its pinned GenieSim provenance")
    expected_dimensions = np.asarray(entry.get("description", {}).get("dimensions", []), dtype=np.float64)
    bounds = np.asarray(provenance.get("bounds_m", []), dtype=np.float64)
    if expected_dimensions.shape != (3,) or bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError(f"Cannot verify dimensions for source asset alias {source_asset_id!r}")
    candidate_dimensions = bounds[1] - bounds[0]
    if not np.allclose(np.sort(candidate_dimensions), np.sort(expected_dimensions), rtol=0.0, atol=2e-6):
        raise ValueError(
            f"Compatibility asset {candidate_id!r} dimensions do not match {source_asset_id!r}: "
            f"{candidate_dimensions.tolist()} vs {expected_dimensions.tolist()}"
        )
    return candidate_id, {
        "requested_source_asset_id": source_asset_id,
        "resolved_asset_id": candidate_id,
        "source_revision": alias["source_revision"],
        "aligned_usd_sha256": alias["aligned_usd_sha256"],
        "dimension_match_m": expected_dimensions.tolist(),
        "identity_verified": False,
        "reason": alias["reason"],
    }


def _scale_xml_vector(node: ET.Element, attribute: str, factor: float, size: int, default: str | None = None) -> None:
    """Scale one authored numeric XML vector, rejecting malformed asset data."""
    raw = node.get(attribute, default)
    if raw is None:
        return
    try:
        values = [float(value) for value in raw.split()]
    except ValueError as exc:
        raise ValueError(f"Invalid {attribute} in scaled Genie Sim asset") from exc
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Invalid {attribute} in scaled Genie Sim asset: {raw!r}")
    node.set(attribute, " ".join(f"{value * factor:.17g}" for value in values))


def _scale_rigid_wrapper(mjcf_root: ET.Element, urdf_root: ET.Element, scale: float) -> None:
    """Bake uniform scale into rigid geometry with constant mass and inertia proportional to scale²."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Rigid-object scale must be positive and finite")
    if abs(scale - 1.0) <= 1e-5:
        return
    bodies = mjcf_root.findall("worldbody/body")
    links = urdf_root.findall("link")
    if (
        len(bodies) != 1 or bodies[0].find("body") is not None or bodies[0].find("joint") is not None
        or len(links) != 1 or urdf_root.find("joint") is not None
    ):
        raise NotImplementedError("Uniform source scale currently requires a single rigid body/link")
    body, link = bodies[0], links[0]
    _scale_xml_vector(body, "pos", scale, 3)
    for mesh in mjcf_root.findall("asset/mesh"):
        _scale_xml_vector(mesh, "scale", scale, 3, "1 1 1")
    for geom in body.findall("geom"):
        geom_type = geom.get("type")
        if geom_type == "mesh":
            _scale_xml_vector(geom, "pos", scale, 3)
        elif geom_type == "sphere":
            _scale_xml_vector(geom, "size", scale, 1)
            _scale_xml_vector(geom, "pos", scale, 3)
        else:
            raise NotImplementedError(f"Uniform source scale does not support MJCF {geom_type!r} geometry")
    for inertial in body.findall("inertial"):
        _scale_xml_vector(inertial, "pos", scale, 3)
        _scale_xml_vector(inertial, "diaginertia", scale * scale, 3)
        _scale_xml_vector(inertial, "fullinertia", scale * scale, 6)

    for visual in (*link.findall("visual"), *link.findall("collision")):
        geometry = visual.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        sphere = geometry.find("sphere") if geometry is not None else None
        if geometry is None or len(geometry) != 1:
            raise NotImplementedError("Uniform source scale requires one URDF geometry per visual/collision")
        if mesh is not None:
            _scale_xml_vector(mesh, "scale", scale, 3, "1 1 1")
        elif sphere is not None:
            _scale_xml_vector(sphere, "radius", scale, 1)
        else:
            raise NotImplementedError("Uniform source scale supports URDF mesh and sphere geometry")
        origin = visual.find("origin")
        if origin is not None:
            _scale_xml_vector(origin, "xyz", scale, 3)
    inertial = link.find("inertial")
    if inertial is not None:
        origin = inertial.find("origin")
        if origin is not None:
            _scale_xml_vector(origin, "xyz", scale, 3)
        inertia = inertial.find("inertia")
        if inertia is not None:
            for attr in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                _scale_xml_vector(inertia, attr, scale * scale, 1)


def _sphere_collision_proxy(asset_id: str, provenance: Mapping[str, object]) -> dict[str, object] | None:
    """Use an analytic sphere for pinned Genie Sim billiard meshes authored as balls."""
    if asset_id not in _BILLIARDS_SPHERE_ASSETS:
        return None
    bounds = np.asarray(provenance.get("bounds_m"), dtype=np.float64)
    mass = float(provenance.get("mass_kg", 0.0))
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid source bounds or mass for billiards asset {asset_id!r}")
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0) or not np.allclose(extents, extents.mean(), rtol=0.0, atol=1e-6):
        raise ValueError(f"Billiards asset {asset_id!r} is not spherical within the pinned source bounds")
    radius = float(extents.mean() * 0.5)
    inertia = float(0.4 * mass * radius * radius)
    return {
        "shape": "sphere",
        "source": "verified_billiards_visual_bounds",
        "center_m": center.tolist(),
        "radius_m": radius,
        "mass_kg": mass,
        "inertia_diagonal_kg_m2": [inertia, inertia, inertia],
    }


def _apply_sphere_collision_proxy(
    mjcf_root: ET.Element, urdf_root: ET.Element, collision_proxy: Mapping[str, object]
) -> None:
    """Replace the generated AABB collider and box inertia with a verified sphere."""
    if collision_proxy.get("shape") != "sphere":
        raise ValueError(f"Unsupported Genie Sim collision proxy: {collision_proxy.get('shape')!r}")
    center = np.asarray(collision_proxy.get("center_m"), dtype=np.float64)
    inertia = np.asarray(collision_proxy.get("inertia_diagonal_kg_m2"), dtype=np.float64)
    radius = float(collision_proxy.get("radius_m", 0.0))
    mass = float(collision_proxy.get("mass_kg", 0.0))
    if (
        center.shape != (3,) or inertia.shape != (3,) or not np.isfinite(center).all()
        or not np.isfinite(inertia).all() or np.any(inertia <= 0)
        or not math.isfinite(radius) or radius <= 0 or not math.isfinite(mass) or mass <= 0
    ):
        raise ValueError("Invalid verified sphere collision proxy")
    center_text = " ".join(f"{value:.17g}" for value in center)
    inertia_text = " ".join(f"{value:.17g}" for value in inertia)

    bodies = mjcf_root.findall("worldbody/body")
    if len(bodies) != 1:
        raise ValueError("Sphere collision proxy requires one rigid MJCF body")
    body = bodies[0]
    collision_geoms = [geom for geom in body.findall("geom") if geom.get("name") == "collision"]
    if len(collision_geoms) != 1 or collision_geoms[0].get("type") != "mesh":
        raise ValueError("Sphere collision proxy requires one generated mesh collision geom")
    geom = collision_geoms[0]
    geom.attrib.pop("mesh", None)
    geom.set("type", "sphere")
    geom.set("size", f"{radius:.17g}")
    geom.set("pos", center_text)
    assets = mjcf_root.find("asset")
    if assets is not None:
        for mesh in list(assets.findall("mesh")):
            if mesh.get("name") == "collision":
                assets.remove(mesh)
    inertials = body.findall("inertial")
    if len(inertials) != 1:
        raise ValueError("Sphere collision proxy requires one MJCF inertial element")
    inertials[0].set("pos", center_text)
    inertials[0].set("mass", f"{mass:.17g}")
    inertials[0].set("diaginertia", inertia_text)
    inertials[0].attrib.pop("fullinertia", None)

    links = urdf_root.findall("link")
    if len(links) != 1:
        raise ValueError("Sphere collision proxy requires one rigid URDF link")
    link = links[0]
    collisions = link.findall("collision")
    if len(collisions) != 1:
        raise ValueError("Sphere collision proxy requires one generated URDF collision")
    geometry = collisions[0].find("geometry")
    if geometry is None or len(geometry) != 1 or geometry.find("mesh") is None:
        raise ValueError("Sphere collision proxy requires one generated URDF collision mesh")
    geometry.clear()
    ET.SubElement(geometry, "sphere", {"radius": f"{radius:.17g}"})
    origin = collisions[0].find("origin")
    if origin is None:
        origin = ET.Element("origin")
        collisions[0].insert(0, origin)
    origin.set("xyz", center_text)
    origin.set("rpy", "0 0 0")
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError("Sphere collision proxy requires one URDF inertial element")
    urdf_origin = inertial.find("origin")
    if urdf_origin is None:
        urdf_origin = ET.SubElement(inertial, "origin")
    urdf_origin.set("xyz", center_text)
    urdf_origin.set("rpy", "0 0 0")
    urdf_mass = inertial.find("mass")
    urdf_inertia = inertial.find("inertia")
    if urdf_mass is None or urdf_inertia is None:
        raise ValueError("Sphere collision proxy requires URDF mass and inertia")
    urdf_mass.set("value", f"{mass:.17g}")
    for attr, value in zip(("ixx", "iyy", "izz"), inertia, strict=True):
        urdf_inertia.set(attr, f"{value:.17g}")
    for attr in ("ixy", "ixz", "iyz"):
        urdf_inertia.set(attr, "0")


def _instance_wrapper(
    asset_root: Path,
    asset_id: str,
    instance_name: str,
    output: Path,
    *,
    asset_alias: dict | None = None,
    source_scene_payload: dict | None = None,
    collision_proxy: dict | None = None,
) -> Path:
    asset_dir = asset_root / asset_id
    provenance_path = asset_dir / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Prepared Genie Sim asset is missing: {asset_dir}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("format") != _OBJECT_FORMAT or provenance.get("asset_id") != asset_id:
        raise ValueError(f"Prepared object provenance does not match {asset_id}: {asset_dir}")
    source_hash = _sha256(provenance_path)
    if output.exists():
        manifest_path = output / "instance.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("format") == _SCENE_WRAPPER_FORMAT
                and manifest.get("asset_id") == asset_id
                and manifest.get("instance_name") == instance_name
                and manifest.get("source_provenance_sha256") == source_hash
                and manifest.get("asset_alias") == asset_alias
                and manifest.get("source_scene_payload") == source_scene_payload
                and manifest.get("collision_proxy") == collision_proxy
            ):
                if all((output / name).is_file() for name in ("object.urdf", "object.xml")):
                    return output
        raise FileExistsError(f"Refusing to overwrite scene object wrapper: {output}")
    for filename in ("object.urdf", "object.xml"):
        if not (asset_dir / filename).is_file():
            raise FileNotFoundError(f"Prepared Genie Sim object is incomplete: {asset_dir / filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{instance_name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / instance_name
        staged.mkdir()
        mjcf_tree = ET.parse(asset_dir / "object.xml")
        mjcf_root = mjcf_tree.getroot()
        mjcf_root.set("model", instance_name)
        root_bodies = mjcf_root.findall("worldbody/body")
        if len(root_bodies) != 1:
            raise ValueError(f"Expected one rigid MJCF root body for {asset_id}")
        root_bodies[0].set("name", instance_name)
        for node in mjcf_root.findall(".//mesh") + mjcf_root.findall(".//texture"):
            filename = node.get("file")
            if filename:
                node.set("file", str((asset_dir / filename).resolve()))
        urdf_tree = ET.parse(asset_dir / "object.urdf")
        urdf_root = urdf_tree.getroot()
        urdf_root.set("name", instance_name)
        links = urdf_root.findall("link")
        if len(links) != 1:
            raise ValueError(f"Expected one rigid URDF root link for {asset_id}")
        links[0].set("name", instance_name)
        for mesh in urdf_root.findall(".//mesh"):
            filename = mesh.get("filename")
            if filename:
                mesh.set("filename", str((asset_dir / filename).resolve()))
        for texture in urdf_root.findall(".//texture"):
            filename = texture.get("filename")
            if filename:
                texture.set("filename", str((asset_dir / filename).resolve()))
        if collision_proxy is not None:
            _apply_sphere_collision_proxy(mjcf_root, urdf_root, collision_proxy)
        scale = float((source_scene_payload or {}).get("uniform_scale", 1.0))
        _scale_rigid_wrapper(mjcf_root, urdf_root, scale)
        mjcf_tree.write(staged / "object.xml", encoding="utf-8", xml_declaration=True)
        urdf_tree.write(staged / "object.urdf", encoding="utf-8", xml_declaration=True)
        wrapper_manifest = {
            "format": _SCENE_WRAPPER_FORMAT,
            "asset_id": asset_id,
            "instance_name": instance_name,
            "source_provenance_sha256": source_hash,
        }
        if asset_alias is not None:
            wrapper_manifest["asset_alias"] = asset_alias
        if source_scene_payload is not None:
            wrapper_manifest["source_scene_payload"] = source_scene_payload
        if collision_proxy is not None:
            wrapper_manifest["collision_proxy"] = collision_proxy
        if abs(scale - 1.0) > 1e-5:
            wrapper_manifest["uniform_scale"] = scale
            wrapper_manifest["inertia_scaling"] = "constant source mass; center scales by s and inertia by s^2"
        (staged / "instance.json").write_text(json.dumps(wrapper_manifest, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(staged), output)
    return output


def build_scene_scenario(
    *,
    instance: dict,
    simulator: str,
    robot: RobotCfg,
    assets_root: Path,
    cache_root: Path,
    cameras: Sequence[BaseCameraCfg] = (),
    fixed_object_names: Sequence[str] = (),
    background_panorama: Path | None = None,
    background_scene_usd: str | None = None,
    workspace_offset_xy: Sequence[float] = (0.0, 0.0),
    robot_position: Sequence[float] = (0.0, 0.0, 0.0),
    robot_orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    headless: bool = True,
) -> ScenarioCfg:
    """Build a source-pose RoboVerse scene from one imported rigid-object layout.

    USD scene snapshots already use the source world frame. The caller may
    supply an XY offset for legacy metadata-only snapshots. Camera configs are
    passed separately. Source USD meshes and diffuse textures are imported as
    portable visuals; the Gaussian-splat room itself is not loaded. Articulated
    objects referenced by supported checkers use their dedicated loaders.
    """
    if simulator not in {"mujoco", "genesis", "newton"}:
        raise ValueError("Genie Sim scenes support only MuJoCo, Genesis and Newton")
    task, instance_id = instance.get("task"), str(instance.get("instance_id"))
    if (
        not isinstance(task, str)
        or not re.fullmatch(r"[A-Za-z0-9_-]+", task)
        or not re.fullmatch(r"[0-9]+", instance_id)
    ):
        raise ValueError("Imported scene metadata is missing a safe task or numeric instance ID")
    layout = _instance_scene_layout(instance)
    if not isinstance(layout, dict) or not layout:
        raise ValueError(f"Genie Sim scene {task}/{instance_id} has no object layout to instantiate")
    if len(workspace_offset_xy) != 2:
        raise ValueError("workspace_offset_xy must contain exactly two coordinates")
    workspace_offset = tuple(float(value) for value in workspace_offset_xy)
    if not all(math.isfinite(value) for value in workspace_offset):
        raise ValueError("workspace_offset_xy must contain finite coordinates")
    pushpull_refs = _pushpull_references(instance)
    articulated_layout_refs: dict[str, str] = {}
    background_articulated_refs: list[str] = []
    for reference in pushpull_refs:
        if reference.startswith("/World/objects/"):
            asset_id = reference.rsplit("/", 1)[-1]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", asset_id):
                raise ValueError(f"Unsupported background articulation reference: {reference!r}")
            background_articulated_refs.append(asset_id)
        elif reference in layout:
            asset_id = layout[reference].get("usd")
            if not isinstance(asset_id, str):
                raise ValueError(f"Articulated checker object {reference!r} has no source asset ID")
            articulated_layout_refs[reference] = asset_id
        else:
            raise ValueError(f"PushPull checker object {reference!r} is absent from layout and source background")
    fixed = set(fixed_object_names)
    backdrop_color = (0.78, 0.78, 0.76)
    objects = [
        PrimitiveCubeCfg(
            name="geniesim_studio_floor",
            size=(20.0, 20.0, 0.05),
            color=backdrop_color,
            physics=PhysicStateType.GEOM,
            default_position=(0.0, 0.0, -0.025),
        ),
    ]
    backdrop_urdf, backdrop_mjcf, backdrop_resources = _studio_backdrop_assets(
        Path(cache_root), panorama_path=background_panorama
    )
    backdrop = RigidObjCfg(
        name="geniesim_studio_backdrop",
        urdf_path=str(backdrop_urdf),
        mjcf_path=str(backdrop_mjcf),
        extra_resources=backdrop_resources,
        physics=PhysicStateType.GEOM,
        fix_base_link=True,
    )
    backdrop.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
    objects.append(backdrop)
    room_assets = _source_environment_geometry_assets(
        Path(cache_root),
        background_scene_usd,
        exclude_asset_ids=tuple(str(entry.get("usd")) for entry in layout.values() if isinstance(entry, dict)),
    )
    if room_assets is not None:
        environment_urdf, environment_mjcf, environment_resources = room_assets
        environment = RigidObjCfg(
            name="geniesim_source_environment",
            urdf_path=str(environment_urdf),
            mjcf_path=str(environment_mjcf),
            extra_resources=environment_resources,
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
        )
        environment.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
        objects.append(environment)
    from roboverse_pack.tasks.icl.geniesim_background_collision import prepare_warehouse_collisions

    collision_assets = prepare_warehouse_collisions(
        Path(cache_root), _GENIESIM_SOURCE_ROOT, background_scene_usd,
        exclude_asset_ids=tuple(str(entry.get("usd")) for entry in layout.values() if isinstance(entry, dict)),
    )
    if collision_assets is not None:
        collision_urdf, collision_mjcf, collision_resources = collision_assets
        collision = RigidObjCfg(
            name="geniesim_warehouse_collisions",
            urdf_path=str(collision_urdf),
            mjcf_path=str(collision_mjcf),
            extra_resources=collision_resources,
            physics=PhysicStateType.GEOM,
            fix_base_link=True,
            newton_hide_collision_shapes=True,
        )
        collision.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
        objects.append(collision)
    seen_names = set()
    cache_root = Path(cache_root).resolve() / task / instance_id
    for index, (name, entry) in enumerate(layout.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name) or name in seen_names:
            raise ValueError(f"Unsupported or duplicate Genie Sim scene object name: {name!r}")
        seen_names.add(name)
        if not isinstance(entry, dict) or not entry.get("usd"):
            raise ValueError(f"Genie Sim object {name} has no source asset ID")
        source_asset_id = str(entry["usd"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_asset_id):
            raise ValueError(f"Unsupported Genie Sim asset ID for {name}: {source_asset_id!r}")
        position = tuple(float(value) for value in entry.get("xyz", ()))
        xyzw = tuple(float(value) for value in entry.get("xyzw", ()))
        if len(position) != 3 or len(xyzw) != 4:
            raise ValueError(f"Genie Sim object {name} has malformed position or quaternion")
        if not all(math.isfinite(value) for value in (*position, *xyzw)):
            raise ValueError(f"Genie Sim object {name} has non-finite position or quaternion")
        position = (position[0] + workspace_offset[0], position[1] + workspace_offset[1], position[2])
        quat_norm = math.sqrt(sum(value * value for value in xyzw))
        if quat_norm < 1e-8:
            raise ValueError(f"Genie Sim object {name} has a zero quaternion")
        articulated_asset_id = articulated_layout_refs.get(name)
        if articulated_asset_id is not None:
            articulated_root = Path(assets_root).resolve().parent / "geniesim_articulated" / articulated_asset_id
            provenance_path = articulated_root / "provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(
                    f"Prepare articulated Genie Sim asset {articulated_asset_id!r} before building {task}/{instance_id}"
                )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            entity_pose = np.eye(4, dtype=np.float64)
            entity_pose[:3, :3] = Rotation.from_quat(xyzw).as_matrix()
            entity_pose[:3, 3] = position
            root_link_pose = np.asarray(provenance.get("root_link_transform_in_entity"), dtype=np.float64)
            if root_link_pose.shape != (4, 4) or not np.isfinite(root_link_pose).all():
                raise ValueError(f"Invalid source root-link transform for articulated asset {articulated_asset_id}")
            base_pose = entity_pose @ root_link_pose
            base_position, base_orientation = _source_pose_from_row_matrix(base_pose.T)
            objects.append(
                _articulated_object_cfg(
                    object_name=name,
                    asset_id=articulated_asset_id,
                    asset_root=articulated_root,
                    wrapper_root=cache_root / "articulated" / name / _sha256(provenance_path)[:12],
                    position=base_position,
                    orientation_wxyz=base_orientation,
                )
            )
            continue
        asset_root = _payload_asset_root(assets_root, entry)
        asset_id, asset_alias = _resolve_scene_asset(asset_root, source_asset_id, entry)
        asset_dir = asset_root / asset_id
        source_provenance = asset_dir / "provenance.json"
        if not source_provenance.is_file():
            raise FileNotFoundError(
                f"Prepare source-scene payload {entry.get('_source_scene_payload', {}).get('asset_path', asset_id)!r}: "
                f"{source_provenance}"
            )
        source_provenance_data = json.loads(source_provenance.read_text(encoding="utf-8"))
        _validate_payload_provenance(entry, source_provenance_data)
        collision_proxy = _sphere_collision_proxy(asset_id, source_provenance_data)
        alias_cache_suffix = f"_a{hashlib.sha256(source_asset_id.encode()).hexdigest()[:8]}" if asset_alias else ""
        source_scene_payload = entry.get("_source_scene_payload")
        payload_cache_suffix = (
            f"_s{hashlib.sha256(json.dumps(source_scene_payload, sort_keys=True).encode()).hexdigest()[:10]}"
            if source_scene_payload else ""
        )
        collision_cache_suffix = (
            f"_c{hashlib.sha256(json.dumps(collision_proxy, sort_keys=True).encode()).hexdigest()[:10]}"
            if collision_proxy else ""
        )
        instance_dir = _instance_wrapper(
            asset_root,
            asset_id,
            name,
            cache_root / (
                f"{index:03d}_{name}_{_sha256(source_provenance)[:12]}{alias_cache_suffix}"
                f"{payload_cache_suffix}{collision_cache_suffix}"
            ),
            asset_alias=asset_alias,
            source_scene_payload=source_scene_payload,
            collision_proxy=collision_proxy,
        )
        resources = [
            str(path.resolve())
            for path in asset_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".obj", ".stl", ".mtl", ".png", ".jpg", ".jpeg"}
        ]
        obj = RigidObjCfg(
            name=name,
            urdf_path=str(instance_dir / "object.urdf"),
            mjcf_path=str(instance_dir / "object.xml"),
            extra_resources=resources,
            physics=PhysicStateType.GEOM
            if _source_marks_fixed(name, asset_id, entry, fixed)
            else PhysicStateType.RIGIDBODY,
            default_position=position,
            default_orientation=(xyzw[3] / quat_norm, xyzw[0] / quat_norm, xyzw[1] / quat_norm, xyzw[2] / quat_norm),
        )
        obj.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
        objects.append(obj)

    for asset_id in background_articulated_refs:
        articulated_root = Path(assets_root).resolve().parent / "geniesim_articulated" / asset_id
        provenance_path = articulated_root / "provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(
                f"Prepare articulated Genie Sim background asset {asset_id!r} before building {task}/{instance_id}"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not background_scene_usd:
            raise ValueError(f"Checker references background object {asset_id!r}, but the run has no source scene USD")
        position, orientation = _background_articulation_pose(
            background_scene_usd,
            asset_id,
            str(provenance.get("root_link", "")),
        )
        objects.append(
            _articulated_object_cfg(
                object_name=asset_id,
                asset_id=asset_id,
                asset_root=articulated_root,
                wrapper_root=cache_root / "articulated" / asset_id / _sha256(provenance_path)[:12],
                position=position,
                orientation_wxyz=orientation,
            )
        )

    robot_cfg = deepcopy(robot)
    shifted_robot_position = (
        float(robot_position[0]) + workspace_offset[0],
        float(robot_position[1]) + workspace_offset[1],
        float(robot_position[2]),
    )
    robot_pose = tuple(float(value) for value in (*shifted_robot_position, *robot_orientation_wxyz))
    if len(robot_pose) != 7 or not all(math.isfinite(value) for value in robot_pose):
        raise ValueError("Robot initial pose must contain seven finite position/quaternion values")
    robot_quat_norm = math.sqrt(sum(value * value for value in robot_pose[3:]))
    if robot_quat_norm < 1e-8:
        raise ValueError("Robot initial quaternion must be nonzero")
    robot_cfg.default_position = robot_pose[:3]
    robot_cfg.default_orientation = tuple(value / robot_quat_norm for value in robot_pose[3:])
    return ScenarioCfg(
        robots=[robot_cfg],
        objects=objects,
        cameras=list(cameras),
        simulator=simulator,
        num_envs=1,
        headless=headless,
        add_default_ground=False,
        lights=[DomeLightCfg(intensity=450), DistantLightCfg(intensity=1100, polar=30, azimuth=45)],
        sim_params=SimParamCfg(
            dt=1 / (_BENCHMARK_ACTION_HZ * _BENCHMARK_PHYSICS_SUBSTEPS),
            njmax=512,
            mujoco_use_scenario_lights=True,
            mujoco_integrator="implicitfast",
            # G1/G2 source drives are very stiff and have authored effort
            # limits. Applying those limits at the MuJoCo actuator can cause
            # a tiny position error to saturate the motor and fling the light
            # wrist links across the target. Moving the same direct-drive
            # bounds to the joint preserves the torque cap and stabilizes the
            # implicit solve.
            mujoco_actuator_limits_at_joint=simulator == "mujoco",
            newton_mujoco_enable_multiccd=collision_assets is not None,
        ),
        decimation=_BENCHMARK_PHYSICS_SUBSTEPS,
    )


def build_run_scenario(
    *,
    spec_path: Path,
    run_id: str,
    simulator: str,
    robot: RobotCfg,
    assets_root: Path,
    cache_root: Path,
    instance_index: int = 0,
    cameras: Sequence[BaseCameraCfg] = (),
    fixed_object_names: Sequence[str] = (),
    headless: bool = True,
    allow_robot_substitution: bool = False,
) -> tuple[ScenarioCfg, dict, dict, dict]:
    """Build the requested source run using its original robot spawn pose and selected scene."""
    run, instance, evaluation = load_run_instance(spec_path, run_id, instance_index)
    source_robot_settings = evaluation.get("robot", {})
    source_robot_id = source_robot_settings.get("robot_id")
    target_robot_id = next((name for name in ("G1", "G2") if robot.name.lower().endswith(name.lower())), None)
    substituted = bool(source_robot_id and target_robot_id != source_robot_id)
    if substituted and (not allow_robot_substitution or target_robot_id is None):
        raise ValueError(f"Source run {run_id!r} requires robot {source_robot_id!r}, but received {robot.name!r}")
    robot_cfg = deepcopy(robot)
    source_pose = _robot_spawn_pose(source_robot_settings.get("robot_init_pose", {}), run_id=run_id)
    if source_pose is None:
        raise ValueError(f"Source run {run_id!r} has no robot initial pose")
    position = source_pose.get("position", ())
    orientation = source_pose.get("quaternion", ())
    if len(position) != 3 or len(orientation) != 4:
        raise ValueError(f"Source run {run_id!r} has malformed robot initial pose")
    source_fixed = evaluation.get("objects", {}).get("fix_objects", [])
    if source_fixed is None:
        source_fixed = []
    if not isinstance(source_fixed, list) or not all(isinstance(name, str) for name in source_fixed):
        raise ValueError(f"Source run {run_id!r} has unsupported fixed-object declarations")
    fixed_names = tuple(dict.fromkeys((*fixed_object_names, *source_fixed)))
    source_scene_path = evaluation.get("scene", {}).get("scene_usd")
    # Authored sub-scenes are referenced at /Workspace in the source runtime;
    # their USD poses and robot_init_pose are already in the same world frame.
    # Older metadata-only imports keep their historical workspace offset.
    workspace_offset = (0.0, 0.0) if instance.get("source_scene") is not None else _scene_workspace_offset(evaluation)
    background_panorama = resolve_geniesim_environment_hdr(source_scene_path)
    source_positions = run.get("source_initial_positions")
    pose_task = run.get("subtask")
    if substituted:
        target_key = f"{target_robot_id}_omnipicker"
        source_positions = run.get("initial_pose_options", {}).get(target_key)
        if source_positions is None:
            source_positions = run.get("cross_robot_reference_poses", {}).get(target_key)
            pose_task = "pick_block_color"
        if not isinstance(source_positions, dict):
            raise ValueError(f"No source or reference reset pose is available for {target_robot_id!r}")
    if source_positions is None:
        raise ValueError(f"Source reset pose for run {run_id!r} was not imported; regenerate the benchmark snapshot")
    from roboverse_pack.tasks.icl.control import prepare_geniesim_reset_pose

    reset_pose = prepare_geniesim_reset_pose(robot_cfg, source_positions)
    robot_cfg.default_joint_positions = reset_pose.joint_positions
    run["reset_pose_validation"] = {
        "clipped_source_positions": reset_pose.clipped_source_positions,
        "max_equality_residual": reset_pose.max_equality_residual,
    }
    run["robot_adaptation"] = {
        "source_robot": source_robot_id,
        "runtime_robot": target_robot_id,
        "substituted": substituted,
        "joint_pose_source_task": pose_task,
        "base_pose": "source_run_pose",
        "cross_robot_pose_and_reach_validated": False if substituted else None,
    }
    scenario = build_scene_scenario(
        instance=instance,
        simulator=simulator,
        robot=robot_cfg,
        assets_root=assets_root,
        cache_root=cache_root,
        cameras=cameras,
        fixed_object_names=fixed_names,
        background_panorama=background_panorama,
        background_scene_usd=source_scene_path,
        workspace_offset_xy=workspace_offset,
        robot_position=position,
        robot_orientation_wxyz=orientation,
        headless=headless,
    )
    return scenario, run, instance, evaluation


def _robot_spawn_pose(value, *, run_id: str) -> dict | None:
    """Accept both workspace-keyed and direct source robot pose encodings."""
    if not isinstance(value, dict) or not value:
        return None
    if "position" in value or "quaternion" in value:
        return value
    pose = next(iter(value.values()))
    if not isinstance(pose, dict):
        raise ValueError(f"Source run {run_id!r} has an unsupported robot_init_pose encoding")
    return pose


def _scene_workspace_offset(evaluation: dict) -> tuple[float, float]:
    """Read the task-local XY origin from the source workspace configuration."""
    scene = evaluation.get("scene", {})
    workspaces = scene.get("function_space_objects", {}) if isinstance(scene, dict) else {}
    workspace = workspaces.get("workspace_00") if isinstance(workspaces, dict) else None
    if not isinstance(workspace, dict):
        return 0.0, 0.0
    position = workspace.get("position")
    if position is None:
        return 0.0, 0.0
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("Genie Sim workspace_00 position must contain three coordinates")
    x, y = float(position[0]), float(position[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Genie Sim workspace_00 XY position must be finite")
    return x, y
