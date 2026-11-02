"""Small Genesis compatibility helpers kept independent of the engine import."""

from __future__ import annotations

import inspect
import xml.etree.ElementTree as stdlib_et
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np

from metasim.utils.xml_safe import ET


def genesis_floating_mjcf(path: str) -> str:
    """Add a free root joint to a rigid MJCF without changing its source file.

    Genesis loads MJCF roots exactly as authored, whereas MetaSim's object
    configuration specifies whether the root is fixed. Inline XML keeps this
    adjustment local; absolute compiler/include paths retain asset resolution.
    Multi-root or articulated roots require an explicit source free joint.
    """
    source = Path(path).resolve()
    root = ET.parse(source).getroot()
    world = root.find("worldbody")
    bodies = world.findall("body") if world is not None else []
    if len(bodies) != 1:
        raise ValueError(f"A floating Genesis MJCF rigid object needs one top-level body: {source}")
    body = bodies[0]
    if body.find("freejoint") is not None or any(j.get("type") == "free" for j in body.findall("joint")):
        return str(source)
    if body.findall("joint"):
        raise ValueError(f"A floating Genesis MJCF root cannot have another joint: {source}")
    stdlib_et.SubElement(body, "freejoint")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = stdlib_et.SubElement(root, "compiler")
    for key in ("assetdir", "meshdir", "texturedir"):
        value = Path(compiler.get(key, ""))
        compiler.set(key, str(value if value.is_absolute() else source.parent / value))
    for include in root.findall("include"):
        file = include.get("file")
        if not file:
            raise ValueError(f"MJCF include has no file: {source}")
        value = Path(file)
        include.set("file", str(value if value.is_absolute() else source.parent / value))
    return stdlib_et.tostring(root, encoding="unicode")


def apply_genesis_mesh_texture_compat(genesis_module: Any) -> None:
    """Let Genesis use source mesh textures when no surface override was requested.

    Genesis 1.4.5 initializes a default color texture in ``Mesh.from_trimesh``
    before it reads an OBJ/MJCF material. Its subsequent ``update_texture`` call
    preserves that default, so imported image textures are lost. Passing an
    uninitialized default surface through the existing importer lets its
    source-material conversion install the image. Explicit color or image
    surfaces keep their normal precedence.
    """
    from genesis.engine.mesh import Mesh

    if getattr(Mesh, "_metasim_source_texture_compat", False):
        return

    original = inspect.getattr_static(Mesh, "from_trimesh").__func__
    signature = inspect.signature(original)
    default_surface_type = genesis_module.surfaces.Default
    color_texture_type = genesis_module.textures.ColorTexture
    image_texture_type = genesis_module.textures.ImageTexture

    def has_source_image(mesh: Any) -> bool:
        visual = getattr(mesh, "visual", None)
        material = getattr(visual, "material", None)
        return getattr(visual, "kind", None) == "texture" and getattr(material, "image", None) is not None

    def is_implicit_default_surface(surface: Any, mesh: Any) -> bool:
        if not isinstance(surface, default_surface_type) or not has_source_image(mesh):
            return False
        texture = getattr(surface, "diffuse_texture", None)
        if not isinstance(texture, color_texture_type) or not np.allclose(texture.color, (1.0, 1.0, 1.0)):
            return False
        if any(getattr(surface, name, None) is not None for name in ("color", "roughness", "metallic", "emissive")):
            return False
        # Genesis 1.4.5's default ``update_texture()`` fills every texture slot
        # with a placeholder; explicit ImageTexture overrides are left intact.
        fields_set = getattr(surface, "model_fields_set", getattr(surface, "__fields_set__", set()))
        implicit_texture_slots = {
            "diffuse_texture",
            "emissive_texture",
            "metallic_texture",
            "normal_texture",
            "opacity_texture",
            "roughness_texture",
        }
        return implicit_texture_slots.issubset(fields_set) and not isinstance(texture, image_texture_type)

    @wraps(original)
    def from_trimesh_compat(cls, mesh, *args, **kwargs):
        bound = signature.bind(cls, mesh, *args, **kwargs)
        surface = bound.arguments.get("surface")
        if surface is None or is_implicit_default_surface(surface, mesh):
            bound.arguments["surface"] = genesis_module.surfaces.Default()
        return original(*bound.args, **bound.kwargs)

    Mesh.from_trimesh = classmethod(from_trimesh_compat)
    Mesh._metasim_source_texture_compat = True


def refresh_genesis_visualizer(scene, *, headless: bool) -> None:
    """Synchronize camera geometry after direct state writes, without stepping physics."""
    visualizer = getattr(scene, "visualizer", None)
    update = getattr(visualizer, "update", None)
    if callable(update):
        if "force" in inspect.signature(update).parameters:
            update(force=True)
        else:
            update()
        return
    viewer = getattr(scene, "viewer", None)
    if not headless and viewer:
        viewer.update()
        return
    raise NotImplementedError("This Genesis version cannot refresh headless rendering without a physics step")


def genesis_primitive_options(obj) -> dict[str, Any]:
    """Translate a primitive's pose and physical flags into Genesis morph options."""
    return {
        "pos": tuple(obj.default_position),
        "quat": tuple(obj.default_orientation),
        "fixed": bool(obj.fix_base_link),
        "collision": bool(obj.collision_enabled),
    }


def _rotate_usd_light_direction(quaternion_wxyz: Sequence[float]) -> tuple[float, float, float]:
    """Rotate UsdLux's local -Z emission direction by a wxyz quaternion."""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm < 1e-8:
        raise ValueError("Distant light quaternion must be finite and nonzero")
    w, x, y, z = quaternion / norm
    vector = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    axis = np.asarray((x, y, z), dtype=np.float64)
    rotated = vector + 2.0 * np.cross(axis, np.cross(axis, vector) + w * vector)
    return tuple(float(value) for value in rotated)


def _genesis_scenario_light_kwargs(scenario_lights: Sequence[Any]) -> dict[str, Any]:
    """Translate the shared distant/dome rig into Genesis' camera light options."""
    if not scenario_lights:
        return {}

    from metasim.scenario.lights import DistantLightCfg, DomeLightCfg

    directional = []
    ambient = np.zeros(3, dtype=np.float64)
    for light in scenario_lights:
        if isinstance(light, DistantLightCfg):
            # Genesis' default directional intensity (5) corresponds to a 0.7
            # diffuse MuJoCo light; use the shared UsdLux-to-MuJoCo fit to keep
            # the declared studio exposure in the same range.
            diffuse = float(light.intensity) * (0.95 / 2000.0)
            directional.append({
                "type": "directional",
                "dir": _rotate_usd_light_direction(light.quat),
                "color": tuple(float(value) for value in light.color),
                "intensity": diffuse * (5.0 / 0.7),
            })
        elif isinstance(light, DomeLightCfg) and not light.texture_file:
            ambient += np.asarray(light.color, dtype=np.float64) * float(light.intensity) * (0.45 / 1000.0)
        else:
            # Older/current Genesis vis options cannot express the remaining
            # UsdLux types faithfully; retain its renderer defaults in that case.
            return {}

    if not directional:
        return {}
    kwargs: dict[str, Any] = {"lights": directional}
    if np.any(ambient):
        kwargs["ambient_light"] = tuple(float(value) for value in np.clip(ambient, 0.0, 1.0))
    return kwargs


def genesis_vis_options_kwargs(
    vis_options_cls: type, num_envs: int, scenario_lights: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Build Genesis camera options across releases, including supported scene lights."""
    fields = getattr(vis_options_cls, "model_fields", None)
    if fields is None:
        fields = getattr(vis_options_cls, "__fields__", {})
    kwargs = {}
    if "rendered_envs_idx" in fields:
        kwargs["rendered_envs_idx"] = tuple(range(num_envs))
    else:
        kwargs["n_rendered_envs"] = num_envs
    if scenario_lights and "lights" in fields and "ambient_light" in fields:
        kwargs.update(_genesis_scenario_light_kwargs(scenario_lights))
    return kwargs


def _contains_tuple_annotation(annotation: Any) -> bool:
    """Return whether a possibly wrapped type annotation contains a tuple."""
    if get_origin(annotation) is tuple:
        return True
    return any(_contains_tuple_annotation(arg) for arg in get_args(annotation))


def genesis_mjcf_armature_kwargs(morph_cls: type) -> dict[str, Any]:
    """Avoid Genesis-injected armature when MJCF supplies its own joint inertias.

    Current Genesis accepts None to retain the values parsed from MJCF,
    including the MJCF default of zero. Older releases that lack this option
    must receive no unknown keyword argument.
    """
    fields = getattr(morph_cls, "model_fields", None)
    if fields is None:
        fields = getattr(morph_cls, "__fields__", {})
    field = fields.get("default_armature")
    if field is None:
        return {}
    annotation = getattr(field, "annotation", None)
    if annotation is None:
        annotation = getattr(field, "outer_type_", None)
    if getattr(field, "allow_none", False) or type(None) in get_args(annotation):
        return {"default_armature": None}
    return {}


def genesis_supports_anisotropic_scale(morph_cls: type) -> bool:
    """Detect whether a Genesis morph can apply a scale independently per axis."""
    fields = getattr(morph_cls, "model_fields", None)
    if fields is None:
        fields = getattr(morph_cls, "__fields__", {})
    field = fields.get("scale") if fields else None
    annotation = getattr(field, "annotation", None)
    if annotation is None:
        # Pydantic v1 stores the normalized annotation on ModelField.type_.
        annotation = getattr(field, "outer_type_", None) or getattr(field, "type_", None)
    if not _contains_tuple_annotation(annotation):
        return False

    # Genesis 1.4.5 declares a tuple scale in the field type, then rejects
    # anisotropic values in a model validator. Check the validator as well as
    # the annotation so type hints alone cannot send an invalid scale to it.
    decorators = getattr(morph_cls, "__pydantic_decorators__", None)
    model_validators = getattr(decorators, "model_validators", {})
    if "_enforce_isotropic_scale" in model_validators:
        return False
    legacy_root_validators = (
        *getattr(morph_cls, "__pre_root_validators__", ()),
        *(func for _, func in getattr(morph_cls, "__post_root_validators__", ())),
    )
    if any(getattr(validator, "__name__", "") == "_enforce_isotropic_scale" for validator in legacy_root_validators):
        return False
    return True


def normalize_genesis_scale(scale: Any, *, supports_anisotropic: bool = False) -> tuple[Any, bool]:
    """Prepare object scale, using a volume-equivalent fallback if needed."""
    if not isinstance(scale, (tuple, list, np.ndarray)):
        return scale, False
    values = np.asarray(scale, dtype=float).reshape(-1)
    if values.size == 0:
        return 1.0, False
    anisotropic = not np.allclose(values, values[0], rtol=1e-7, atol=1e-9)
    if anisotropic and supports_anisotropic:
        return tuple(float(value) for value in values), False
    if anisotropic and np.all(values > 0):
        # Genesis versions that restrict file morphs to isotropic scales still
        # get the requested volume change, with no arbitrary preference for X.
        uniform_scale = float(np.exp(np.log(values).mean()))
        return uniform_scale, True
    return float(values[0]), anisotropic


def genesis_position_offsets(absolute_positions: Any, qpos0: Any) -> Any:
    """Convert MetaSim absolute joint positions to Genesis qpos0-relative DOFs."""
    return absolute_positions - qpos0


def genesis_absolute_positions(position_offsets: Any, qpos0: Any) -> Any:
    """Convert Genesis qpos0-relative DOFs to MetaSim absolute joint positions."""
    return position_offsets + qpos0


def apply_genesis_actuator_settings(
    entity: Any,
    actuators: Mapping[str, Any] | None,
    control_types: Mapping[str, str] | None,
) -> None:
    """Apply MetaSim joint gains and effort limits to a built Genesis entity."""
    if not actuators:
        return

    joints_by_name = {joint.name: joint for joint in entity.joints}
    control_types = control_types or {}
    for joint_name, actuator in actuators.items():
        joint = joints_by_name.get(joint_name)
        if joint is None or joint.dofs_idx_local is None:
            continue
        dofs_idx_local = list(joint.dofs_idx_local)
        if not dofs_idx_local:
            continue

        if control_types.get(joint_name, "position") == "effort":
            # Effort-controlled joints must not retain an implicit PD drive.
            entity.set_dofs_kp(0.0, dofs_idx_local=dofs_idx_local)
            entity.set_dofs_kv(0.0, dofs_idx_local=dofs_idx_local)
        else:
            if actuator.stiffness is not None:
                entity.set_dofs_kp(float(actuator.stiffness), dofs_idx_local=dofs_idx_local)
            if actuator.damping is not None:
                entity.set_dofs_kv(float(actuator.damping), dofs_idx_local=dofs_idx_local)

        if actuator.effort_limit_sim is not None:
            effort_limit = abs(float(actuator.effort_limit_sim))
            entity.set_dofs_force_range(-effort_limit, effort_limit, dofs_idx_local=dofs_idx_local)
