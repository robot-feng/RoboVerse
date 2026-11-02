from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import genesis as gs
import numpy as np
import torch
from loguru import logger as log
from scipy.spatial.transform import Rotation

from metasim.utils.xml_safe import ET  # defused parser for untrusted MJCF/URDF

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity, RigidJoint
    from genesis.vis.camera import Camera

from metasim.queries.base import BaseQueryType
from metasim.scenario.cameras import FisheyeCameraCfg
from metasim.scenario.objects import (
    ArticulationObjCfg,
    PrimitiveCubeCfg,
    PrimitiveSphereCfg,
    RigidObjCfg,
    _FileBasedMixin,
)
from metasim.scenario.robot import RobotCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.sim import BaseSimHandler
from metasim.types import CompatActionInput, DictEnvState
from metasim.utils.fisheye import CUBEMAP_ROTATIONS, cubemap_lookup, fisheye_rays, sample_cubemap_rgb
from metasim.utils.gs_util import alpha_blend_rgba
from metasim.utils.state import CameraState, ObjectState, RobotState, TensorState, action_input_to_tensor

from ._genesis_compat import (
    apply_genesis_actuator_settings,
    apply_genesis_mesh_texture_compat,
    genesis_absolute_positions,
    genesis_floating_mjcf,
    genesis_mjcf_armature_kwargs,
    genesis_position_offsets,
    genesis_primitive_options,
    genesis_supports_anisotropic_scale,
    genesis_vis_options_kwargs,
    normalize_genesis_scale,
    refresh_genesis_visualizer,
)

apply_genesis_mesh_texture_compat(gs)


def _mjcf_meshes_are_explicitly_noncolliding(mjcf_path: str | None) -> bool:
    """Return true only when every MJCF geom explicitly opts out of collision."""
    if not mjcf_path:
        return False
    try:
        geoms = ET.parse(mjcf_path).getroot().iter("geom")
        geoms = list(geoms)
    except (OSError, ET.ParseError):
        return False
    return bool(geoms) and all(geom.get("contype") == "0" and geom.get("conaffinity") == "0" for geom in geoms)


def _mjcf_entity_material(mjcf_path: str | None, *, fixed: bool):
    """Avoid geometry-derived inertia for fixed MJCF assets used only for rendering."""
    if fixed and _mjcf_meshes_are_explicitly_noncolliding(mjcf_path):
        return gs.materials.Kinematic()
    return gs.materials.Rigid()


def _fisheye_world_rotation(camera) -> np.ndarray:
    """OpenGL camera-to-world rotation for a world-fixed look-at camera."""
    forward = np.asarray(camera.look_at, dtype=float) - np.asarray(camera.pos, dtype=float)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError(f"Camera {camera.name}: look_at must differ from pos")
    forward /= norm
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(right, forward), -forward))


def _validate_fisheye_support(camera, scenario) -> None:
    """Reject outputs the current cube-face implementation cannot provide."""
    if set(camera.data_types) != {"rgb"}:
        raise NotImplementedError("Genesis fisheye cameras currently support RGB only")
    if scenario.num_envs != 1:
        raise NotImplementedError("Genesis fisheye cameras currently support one environment")
    if scenario.gs_scene is not None:
        raise NotImplementedError("Genesis fisheye cameras do not support Gaussian-splat backgrounds")


def _fisheye_face_mount(transform: np.ndarray, face_rotation: np.ndarray) -> np.ndarray:
    """Keep the lens origin and rotate a face in the lens-local camera frame."""
    face_transform = transform.copy()
    face_transform[:3, :3] = transform[:3, :3] @ face_rotation
    return face_transform

# Apply IGL compatibility patch
try:
    import genesis.engine.entities.rigid_entity.rigid_geom as _rigid_geom_module
    import igl as _igl

    _original_compute_sd = _rigid_geom_module.RigidGeom._compute_sd

    def _patched_compute_sd(self, query_points):
        """Patched version that handles different IGL return values"""
        result = _igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)
        if isinstance(result, tuple):
            return result[0] if len(result) > 0 else None
        return result

    _rigid_geom_module.RigidGeom._compute_sd = _patched_compute_sd
except Exception:
    pass

# Optional: RoboSplatter imports for GS background rendering
try:
    from robo_splatter.models.camera import Camera as SplatCamera
    from robo_splatter.render.scenes import SceneRenderType

    ROBO_SPLATTER_AVAILABLE = True
except ImportError:
    ROBO_SPLATTER_AVAILABLE = False
    log.debug("RoboSplatter not available. GS background rendering will be disabled.")


class GenesisHandler(BaseSimHandler):
    _reports_target_from_action_cache = True  # the engine holds no joint target; see BaseSimHandler
    get_states_honours_env_ids = True  # ``_get_states`` indexes by ``env_ids``
    _default_physics_dt = 1 / 100  # the engine default this backend passes at construction

    # ``_set_states`` indexes ``state["objects"]`` directly, so it only
    # accepts the list-of-dict form; the base converts TensorState input.
    _set_states_input_type = "dict"

    def __init__(self, scenario: ScenarioCfg, optional_queries: dict[str, BaseQueryType] | None = None):
        super().__init__(scenario, optional_queries)
        self._actions_cache: CompatActionInput = []
        self.object_inst_dict: dict[str, RigidEntity] = {}
        self.camera_inst_dict: dict[str, Camera] = {}
        self._fisheye_cameras: dict[str, dict] = {}
        self.robot = self.robots[0] if self.robots else None

    def _get_camera_params(self, camera):
        """Get camera intrinsics and extrinsics from Genesis camera configuration.

        Args:
            camera: PinholeCameraCfg object

        Returns:
            Ks: (3, 3) intrinsic matrix
            c2w: (4, 4) camera-to-world transformation matrix
        """
        # Use an authored calibration when supplied; otherwise CameraCfg derives
        # the centered matrix from focal length and aperture.
        Ks = np.asarray(camera.intrinsics, dtype=np.float64)

        # Extrinsics: build from camera configuration
        cam_pos = np.array(camera.pos)
        forward = np.array(camera.look_at) - cam_pos
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Build c2w matrix (OpenGL convention: camera looks along -Z)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # Z axis points backward
        c2w[:3, 3] = cam_pos

        return Ks, c2w

    def _add_default_ground(self) -> None:
        """Add Genesis' implicit plane unless the scenario supplies its own floor."""
        if not getattr(self.scenario, "add_default_ground", True):
            return
        try:
            self.scene_inst.add_entity(gs.morphs.Plane())
        except Exception as exc:
            log.warning(f"Could not add ground plane: {exc}")

    def launch(self) -> None:
        super().launch()
        # Backend selection: METASIM_GENESIS_BACKEND env var ("cpu" | "gpu", default gpu)
        # falls back to gs.cpu when CUDA isn't available so headless CI doesn't crash.
        backend_env = os.environ.get("METASIM_GENESIS_BACKEND", "gpu").lower()
        # Genesis' decorated banner includes emoji that the default Windows
        # cp936 PowerShell stream cannot encode. The raw theme keeps its log
        # output compatible with the native Windows console.
        genesis_theme = "raw" if os.name == "nt" else "dark"
        if backend_env == "cpu" or not torch.cuda.is_available():
            gs.init(backend=gs.cpu, theme=genesis_theme)
        else:
            gs.init(backend=gs.gpu, theme=genesis_theme)
        self.scene_inst = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.physics_dt,
                substeps=1,  # substeps > 1 unstable upstream — track upstream genesis bug before raising
            ),
            vis_options=gs.options.VisOptions(
                **genesis_vis_options_kwargs(gs.options.VisOptions, self.scenario.num_envs, self.lights)
            ),
            rigid_options=gs.options.RigidOptions(
                # Genesis exposes self-collision as a scene-wide switch. Match
                # RobotCfg when all configured robots disable self-collision;
                # retain Genesis' default for object-only scenes.
                enable_self_collision=(not self.robots or any(robot.enabled_self_collisions for robot in self.robots))
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=not self.headless,
        )

        self._add_default_ground()

        for robot in self.robots:
            robot_file_type = robot.file_type.get("genesis", "urdf")
            if robot_file_type == "mjcf":
                if not robot.mjcf_path:
                    raise ValueError(f"Robot {robot.name} requires an MJCF path for Genesis")
                if not robot.fix_base_link or robot.collapse_fixed_joints:
                    raise NotImplementedError("Genesis MJCF robots currently require a fixed base without link merging")
                morph = gs.morphs.MJCF(
                    file=robot.mjcf_path, **genesis_mjcf_armature_kwargs(gs.morphs.MJCF)
                )
            elif robot_file_type == "urdf":
                # Remove invalid empty collision nodes before the URDF parser.
                robot_urdf_path = self._sanitize_urdf(robot.urdf_path)
                morph = gs.morphs.URDF(
                    file=robot_urdf_path, fixed=robot.fix_base_link, merge_fixed_links=robot.collapse_fixed_joints
                )
            else:
                raise ValueError(f"Unsupported Genesis robot file type {robot_file_type!r}")
            robot_inst: RigidEntity = self.scene_inst.add_entity(
                morph,
                material=gs.materials.Rigid(gravity_compensation=1 if not robot.enabled_gravity else 0),
            )
            self.object_inst_dict[robot.name] = robot_inst
            if robot is self.robot:
                self.robot_inst = robot_inst
            self._sanitize_joint_dofs(robot_inst)

        ## Add objects
        for obj in self.scenario.objects:
            genesis_scale = obj.scale if isinstance(obj, _FileBasedMixin) else None
            if isinstance(obj, _FileBasedMixin):
                morph_cls = gs.morphs.URDF
                if isinstance(obj, RigidObjCfg) and obj.file_type["genesis"] == "mjcf":
                    morph_cls = gs.morphs.MJCF
                genesis_scale, anisotropic = normalize_genesis_scale(
                    obj.scale,
                    supports_anisotropic=genesis_supports_anisotropic_scale(morph_cls),
                )
                if anisotropic:
                    log.warning(
                        f"Installed Genesis does not support different scaling for each axis for {obj.name}; "
                        f"using isotropic scale={genesis_scale}"
                    )
            if isinstance(obj, PrimitiveCubeCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Box(size=obj.size, **genesis_primitive_options(obj)),
                    surface=gs.surfaces.Default(color=obj.color),
                )
            elif isinstance(obj, PrimitiveSphereCfg):
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.Sphere(radius=obj.radius, **genesis_primitive_options(obj)),
                    surface=gs.surfaces.Default(color=obj.color),
                )
            elif isinstance(obj, RigidObjCfg):
                if obj.file_type["genesis"] == "urdf":
                    urdf_path = self._sanitize_urdf(obj.urdf_path) if obj.urdf_path else None
                    obj_inst = self.scene_inst.add_entity(
                        gs.morphs.URDF(file=urdf_path, fixed=obj.fix_base_link, scale=genesis_scale),
                    )
                # Use convex decomposition mjcf file for more accurate physical collision.
                elif obj.file_type["genesis"] == "mjcf":
                    mjcf_path = obj.mjcf_path if obj.fix_base_link else genesis_floating_mjcf(obj.mjcf_path)
                    collision_enabled = not _mjcf_meshes_are_explicitly_noncolliding(mjcf_path)
                    morph = gs.morphs.MJCF(
                        file=mjcf_path,
                        scale=genesis_scale,
                        collision=collision_enabled,
                        convexify=False,
                        decimate=False,
                        **genesis_mjcf_armature_kwargs(gs.morphs.MJCF),
                    )
                    material = _mjcf_entity_material(mjcf_path, fixed=obj.fix_base_link)
                    obj_inst = self.scene_inst.add_entity(morph, material=material)
                else:
                    raise NotImplementedError(f"RigidObjCfg {obj.name} must have either urdf_path or mjcf_path set.")

            elif isinstance(obj, ArticulationObjCfg):
                urdf_path = self._sanitize_urdf(obj.urdf_path) if obj.urdf_path else None
                obj_inst = self.scene_inst.add_entity(
                    gs.morphs.URDF(file=urdf_path, fixed=obj.fix_base_link, scale=genesis_scale),
                )
            else:
                raise NotImplementedError(f"Object type {type(obj)} not supported")
            self.object_inst_dict[obj.name] = obj_inst
            self._sanitize_joint_dofs(obj_inst)

        ## Add cameras
        for camera in self.cameras:
            if isinstance(camera, FisheyeCameraCfg):
                _validate_fisheye_support(camera, self.scenario)
                rays, valid = fisheye_rays(camera)
                face_size = max(camera.width, camera.height)
                # Mounted poses come from mount_quat after build; the look-at
                # fields are only relevant for world-fixed cameras.
                base_rotation = np.eye(3) if camera.mount_to is not None else _fisheye_world_rotation(camera)
                pos = np.asarray(camera.pos, dtype=float)
                faces = []
                for face_rotation in CUBEMAP_ROTATIONS:
                    rotation = base_rotation @ face_rotation
                    faces.append(self.scene_inst.add_camera(
                        res=(face_size, face_size), pos=tuple(pos),
                        lookat=tuple(pos - rotation[:, 2]), up=tuple(rotation[:, 1]), fov=90,
                    ))
                self._fisheye_cameras[camera.name] = {
                    "faces": faces,
                    "valid": valid,
                    "lookup": cubemap_lookup(rays, valid, face_size),
                }
                continue
            camera_inst = self.scene_inst.add_camera(
                res=(camera.width, camera.height),
                pos=camera.pos,
                lookat=camera.look_at,
                fov=camera.vertical_fov,
            )
            self.camera_inst_dict[camera.name] = camera_inst

        self.scene_inst.build(
            n_envs=self.scenario.num_envs, env_spacing=(self.scenario.env_spacing, self.scenario.env_spacing)
        )

        for robot in self.robots:
            self._apply_robot_default_joint_positions(robot)
            apply_genesis_actuator_settings(
                self.object_inst_dict[robot.name],
                robot.actuators,
                robot.control_type,
            )

        for camera in self.cameras:
            if camera.mount_to is None:
                continue
            if self.num_envs != 1:
                raise NotImplementedError("Genesis mounted cameras currently support one environment")
            if not isinstance(camera.mount_to, str) or not isinstance(camera.mount_link, str):
                raise ValueError(f"Camera {camera.name} needs string mount_to and mount_link")
            if camera.mount_pos is None or camera.mount_quat is None:
                raise ValueError(f"Camera {camera.name} needs mount_pos and mount_quat")
            if self.scenario.gs_scene is not None:
                raise NotImplementedError("Mounted Genesis cameras with Gaussian-splat backgrounds are not supported")
            entity = self.object_inst_dict.get(camera.mount_to)
            if entity is None:
                raise ValueError(f"Camera {camera.name} mount object {camera.mount_to!r} does not exist")
            link = entity.get_link(name=camera.mount_link.split("/")[-1])
            if link is None:
                raise ValueError(f"Camera {camera.name} mount link {camera.mount_link!r} does not exist")
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(camera.mount_quat, scalar_first=True).as_matrix()
            transform[:3, 3] = camera.mount_pos
            if isinstance(camera, FisheyeCameraCfg):
                for face, face_rotation in zip(self._fisheye_cameras[camera.name]["faces"], CUBEMAP_ROTATIONS, strict=True):
                    face.attach(link, _fisheye_face_mount(transform, face_rotation))
            else:
                self.camera_inst_dict[camera.name].attach(link, transform)

        # Initialize GS background if enabled
        self._build_gs_background()

    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Return articulation link names in backend or alphabetical order."""
        obj_inst = self.object_inst_dict.get(obj_name)
        if obj_inst is None:
            return []
        names = [link.name for link in obj_inst.links]
        return sorted(names) if sort else names

    @staticmethod
    def _get_body_state(obj_inst, body_names: list[str], env_ids: list[int]) -> torch.Tensor | None:
        """Pack world link pose and velocity in the shared 13-value body-state layout."""
        if not body_names:
            return None
        links_by_name = {link.name: link for link in obj_inst.links}
        missing = [name for name in body_names if name not in links_by_name]
        if missing:
            raise ValueError(f"Genesis articulation is missing configured links: {missing}")
        link_indices = [links_by_name[name].idx_local for name in body_names]
        pos = obj_inst.get_links_pos(links_idx_local=link_indices, envs_idx=env_ids, relative=True)
        quat = obj_inst.get_links_quat(links_idx_local=link_indices, envs_idx=env_ids, relative=True)
        vel = obj_inst.get_links_vel(links_idx_local=link_indices, envs_idx=env_ids, relative=True)
        ang = obj_inst.get_links_ang(links_idx_local=link_indices, envs_idx=env_ids)
        body_state = torch.cat((pos, quat, vel, ang), dim=-1)
        if body_state.ndim != 3 or body_state.shape[-1] != 13:
            raise ValueError(f"Genesis link state must have shape (num_envs, num_links, 13), got {body_state.shape}")
        return body_state

    def _get_states(self, env_ids: list[int] | None = None) -> TensorState:
        if env_ids is None:
            env_ids = list(range(self.num_envs))

        object_states = {}
        for obj in self.objects:
            obj_inst = self.object_inst_dict[obj.name]
            if isinstance(obj, ArticulationObjCfg):
                joint_reindex = self.get_joint_reindex(obj.name)
                joint_names = self._get_joint_names(obj.name, sort=False)
                dofs_idx_local, qpos0_by_dof, _ = self._joint_dof_layout(obj_inst, joint_names)
                joint_pos = obj_inst.get_dofs_position(dofs_idx_local=dofs_idx_local, envs_idx=env_ids)
                joint_pos = genesis_absolute_positions(
                    joint_pos,
                    torch.as_tensor(qpos0_by_dof, dtype=joint_pos.dtype, device=joint_pos.device),
                )
                body_names = self._get_body_names(obj.name, sort=True)
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                    body_names=body_names,
                    body_state=self._get_body_state(obj_inst, body_names, env_ids),
                    joint_pos=joint_pos[:, joint_reindex],
                    joint_vel=obj_inst.get_dofs_velocity(dofs_idx_local=dofs_idx_local, envs_idx=env_ids)[
                        :, joint_reindex
                    ],
                )
            else:
                state = ObjectState(
                    root_state=torch.cat(
                        [
                            obj_inst.get_pos(envs_idx=env_ids),
                            obj_inst.get_quat(envs_idx=env_ids),
                            obj_inst.get_vel(envs_idx=env_ids),
                            obj_inst.get_ang(envs_idx=env_ids),
                        ],
                        dim=-1,
                    ),
                )
            object_states[obj.name] = state

        robot_states = {}
        for obj in self.robots:
            obj_inst = self.object_inst_dict[obj.name]
            joint_reindex = self.get_joint_reindex(obj.name)
            joint_names = self._get_joint_names(obj.name, sort=False)
            dofs_idx_local, qpos0_by_dof, _ = self._joint_dof_layout(obj_inst, joint_names)
            joint_pos = obj_inst.get_dofs_position(dofs_idx_local=dofs_idx_local, envs_idx=env_ids)
            joint_pos = genesis_absolute_positions(
                joint_pos,
                torch.as_tensor(qpos0_by_dof, dtype=joint_pos.dtype, device=joint_pos.device),
            )
            joint_pos_target = self._joint_pos_target_from_action_cache(obj.name)
            effort_target = self._get_effort_targets() if self._get_control_mode(obj.name) == "effort" else None
            body_names = self._get_body_names(obj.name, sort=True)
            if env_ids is not None:  # every other field is sliced by env_ids here
                joint_pos_target = None if joint_pos_target is None else joint_pos_target[env_ids]
                effort_target = None if effort_target is None else effort_target[env_ids]
            state = RobotState(
                root_state=torch.cat(
                    [
                        obj_inst.get_pos(envs_idx=env_ids),
                        obj_inst.get_quat(envs_idx=env_ids),
                        obj_inst.get_vel(envs_idx=env_ids),
                        obj_inst.get_ang(envs_idx=env_ids),
                    ],
                    dim=-1,
                ),
                body_names=body_names,
                body_state=self._get_body_state(obj_inst, body_names, env_ids),
                joint_pos=joint_pos[:, joint_reindex],
                joint_vel=obj_inst.get_dofs_velocity(dofs_idx_local=dofs_idx_local, envs_idx=env_ids)[:, joint_reindex],
                joint_pos_target=joint_pos_target,
                joint_vel_target=None,  # Genesis uses position/effort control; vel target not separately tracked
                joint_effort_target=effort_target,
            )
            robot_states[obj.name] = state

        camera_states = {}
        for camera in self.cameras:
            if isinstance(camera, FisheyeCameraCfg):
                projection = self._fisheye_cameras[camera.name]
                images = []
                for face in projection["faces"]:
                    if camera.mount_to is not None:
                        face.move_to_attach()
                    rgb, _, _, _ = face.render(depth=False, segmentation=False)
                    rgb = rgb.detach().cpu().numpy() if isinstance(rgb, torch.Tensor) else np.asarray(rgb)
                    if np.issubdtype(rgb.dtype, np.floating) and rgb.max(initial=0) <= 1.01:
                        rgb = rgb * 255
                    images.append(np.clip(rgb, 0, 255).astype(np.uint8))
                valid = projection["valid"]
                rgb = sample_cubemap_rgb(images, projection["lookup"], valid)
                camera_states[camera.name] = CameraState(
                    rgb=torch.from_numpy(rgb).unsqueeze(0),
                    depth=None,
                    valid_mask=torch.from_numpy(valid.copy()).unsqueeze(0),
                )
                continue
            camera_inst = self.camera_inst_dict[camera.name]
            if camera.mount_to is not None:
                camera_inst.move_to_attach()
            rgb, depth, segmentation, _ = camera_inst.render(depth=True, segmentation=True)

            # Ensure tensors and normalize RGB to [0, 255] for consistency
            if isinstance(rgb, np.ndarray):
                rgb_t = torch.from_numpy(rgb.copy())
            else:
                rgb_t = torch.as_tensor(rgb)
            if rgb_t.is_floating_point():
                # If already [0,1], scale to [0,255]
                try:
                    maxv = float(rgb_t.max().item())
                except Exception:
                    maxv = 1.0
                if maxv <= 1.01:
                    rgb_t = (rgb_t * 255.0).clamp(0, 255)

            if isinstance(depth, np.ndarray):
                depth_t = torch.from_numpy(depth.copy())
            else:
                depth_t = torch.as_tensor(depth)

            # GS background blending
            if self.scenario.gs_scene is not None and self.scenario.gs_scene.with_gs_background:
                assert ROBO_SPLATTER_AVAILABLE, (
                    "RoboSplatter is not available. GS background rendering will be disabled."
                )
                # Get camera parameters
                Ks, c2w = self._get_camera_params(camera)

                # Render GS background
                gs_cam = SplatCamera.init_from_pose_list(
                    pose_list=c2w,
                    camera_intrinsic=Ks,
                    image_height=int(camera.height),
                    image_width=int(camera.width),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                gs_result = self.gs_background.render(gs_cam, render_type=SceneRenderType.FOREGROUND)
                gs_result.to_numpy()

                # Create foreground mask from segmentation
                if segmentation is not None:
                    seg_np = segmentation if isinstance(segmentation, np.ndarray) else segmentation.cpu().numpy()
                    # Exclude background and ground plane (typically ID 0 = ground, ID 1 = first object)
                    foreground_mask = seg_np > 1
                    mask = np.where(foreground_mask, 255, 0).astype(np.uint8)

                    # Blending RGB
                    sim_rgb = (
                        rgb_t.numpy().astype(np.uint8) if isinstance(rgb_t, torch.Tensor) else rgb_t.astype(np.uint8)
                    )
                    foreground = np.concatenate([sim_rgb, mask[..., None]], axis=-1)
                    background = gs_result.rgb.squeeze(0)
                    blended_rgb = alpha_blend_rgba(foreground, background)
                    rgb_t = torch.from_numpy(np.array(blended_rgb.copy()))

                    # Compose depth
                    bg_depth = gs_result.depth.squeeze(0)
                    if bg_depth.ndim == 3 and bg_depth.shape[-1] == 1:
                        bg_depth = bg_depth[..., 0]
                    depth_np = depth_t.numpy() if isinstance(depth_t, torch.Tensor) else depth_t
                    depth_comp = np.where(foreground_mask, depth_np, bg_depth)
                    depth_t = torch.from_numpy(depth_comp.copy())

            state = CameraState(
                rgb=rgb_t.unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
                depth=depth_t.unsqueeze(0).repeat_interleave(self.num_envs, dim=0),
                intrinsics=torch.as_tensor(camera.intrinsics, dtype=torch.float32, device=rgb_t.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1, -1)
                .clone(),
            )
            camera_states[camera.name] = state

        extras = self.get_extra()  # extra observations TODO: add extra observations
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

    def _set_states(self, states: list[DictEnvState], env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        states_flat = [state["objects"] | state["robots"] for state in states]

        all_objects = self.objects + self.robots
        for obj in all_objects:
            if obj.name not in self.object_inst_dict:
                log.warning(f"Object {obj.name} not found in object_inst_dict")
                continue

            obj_inst = self.object_inst_dict[obj.name]

            positions = torch.stack(
                [torch.as_tensor(states_flat[env_id][obj.name]["pos"]) for env_id in env_ids], dim=0
            )
            rotations = torch.stack(
                [torch.as_tensor(states_flat[env_id][obj.name]["rot"]) for env_id in env_ids], dim=0
            )

            obj_inst.set_pos(positions)
            obj_inst.set_quat(rotations)

            is_articulated = isinstance(obj, (ArticulationObjCfg, RobotCfg))

            if is_articulated:
                joint_names = self._get_joint_names(obj.name, sort=False)
                dofs_idx_local, qpos0_by_dof, _ = self._joint_dof_layout(obj_inst, joint_names)

                if joint_names and "dof_pos" in states_flat[0][obj.name]:
                    joint_positions = []
                    for env_id in env_ids:
                        env_joint_pos = []
                        for jn in joint_names:
                            if jn in states_flat[env_id][obj.name]["dof_pos"]:
                                pos_val = states_flat[env_id][obj.name]["dof_pos"][jn]
                                env_joint_pos.append(float(pos_val))
                            else:
                                env_joint_pos.append(0.0)
                        joint_positions.append(env_joint_pos)

                    joint_pos_array = genesis_position_offsets(
                        torch.tensor(joint_positions, dtype=torch.float32),
                        torch.as_tensor(qpos0_by_dof, dtype=torch.float32),
                    )
                    # Write the named DOFs directly. Free-root pose is handled
                    # above with set_pos/set_quat and is not in this list.
                    if dofs_idx_local:
                        obj_inst.set_dofs_position(
                            joint_pos_array,
                            dofs_idx_local=dofs_idx_local,
                            envs_idx=env_ids,
                        )

        # Entity setters perform forward kinematics by default. Advancing a
        # scene step here would also integrate gravity/contact forces, so a
        # state write would no longer be an exact reset (unlike MuJoCo and
        # Newton's set-state paths).

        if not self.headless and hasattr(self.scene_inst, "viewer") and self.scene_inst.viewer:
            self.scene_inst.viewer.update()

    def _set_dof_targets(self, actions: CompatActionInput) -> None:
        self._actions_cache = actions

        if not self.robots:
            return

        # Fast-path: tensor input (VectorEnv) in handler/API order across all robots.
        if isinstance(actions, (torch.Tensor, np.ndarray)):
            actions_tensor = action_input_to_tensor(self, actions, device=self.device)
            offset = 0
            for robot in self.robots:
                obj_name = robot.name
                obj_inst = self.object_inst_dict[obj_name]
                control_mode = self._get_control_mode(obj_name)
                sim_joint_names = self._get_joint_names(obj_name, sort=False)
                sorted_joint_names = self._get_joint_names(obj_name, sort=True)
                robot_width = len(sorted_joint_names)

                if robot_width == 0:
                    continue

                robot_actions = actions_tensor[:, offset : offset + robot_width]
                offset += robot_width
                if robot_actions.shape[1] == 0:
                    continue

                name_to_sorted_idx = {joint_name: idx for idx, joint_name in enumerate(sorted_joint_names)}
                action_indices = [
                    name_to_sorted_idx[joint_name] for joint_name in sim_joint_names if joint_name in name_to_sorted_idx
                ]
                if not action_indices:
                    continue

                action_values = robot_actions[:, action_indices]
                dofs_idx_local, qpos0_by_dof, _ = self._joint_dof_layout(obj_inst, sim_joint_names)

                if not dofs_idx_local:
                    continue

                action_values = genesis_position_offsets(
                    action_values,
                    torch.as_tensor(qpos0_by_dof, dtype=action_values.dtype, device=action_values.device),
                )

                if control_mode == "effort":
                    obj_inst.control_dofs_force(force=action_values, dofs_idx_local=dofs_idx_local)
                else:
                    obj_inst.control_dofs_position(position=action_values, dofs_idx_local=dofs_idx_local)
            return

        # Dict/list input path
        for robot in self.robots:
            obj_name = robot.name
            obj_inst = self.object_inst_dict[obj_name]
            control_mode = self._get_control_mode(obj_name)
            sim_joint_names = self._get_joint_names(obj_name, sort=False)
            joint_names = sim_joint_names

            if control_mode == "effort":
                available_joints = set()
                for env_action in actions:
                    robot_action = env_action.get(obj_name) or {}
                    effort_targets = robot_action.get("dof_effort_target") or {}
                    available_joints.update(effort_targets.keys())
                if available_joints:
                    joint_names = [jn for jn in joint_names if jn in available_joints]

                effort = []
                for env_id in range(self.num_envs):
                    robot_action = actions[env_id].get(obj_name) or {}
                    effort_targets = robot_action.get("dof_effort_target") or {}
                    effort.append([effort_targets.get(jn, 0.0) for jn in joint_names])

                dofs_idx_local = []
                for j in obj_inst.joints:
                    if j.dofs_idx_local is not None and j.name in joint_names:
                        dofs_idx_local.extend(j.dofs_idx_local)

                if dofs_idx_local:
                    obj_inst.control_dofs_force(
                        force=effort,
                        dofs_idx_local=dofs_idx_local,
                    )
            else:
                available_joints = set()
                for env_action in actions:
                    robot_action = env_action.get(obj_name) or {}
                    pos_targets = robot_action.get("dof_pos_target") or {}
                    available_joints.update(pos_targets.keys())
                if available_joints:
                    joint_names = [jn for jn in joint_names if jn in available_joints]
                dofs_idx_local, qpos0_by_dof, _ = self._joint_dof_layout(obj_inst, joint_names)

                position = []
                for env_id in range(self.num_envs):
                    robot_action = actions[env_id].get(obj_name) or {}
                    pos_targets = robot_action.get("dof_pos_target") or {}
                    position.append(
                        genesis_position_offsets(
                            np.asarray([
                                pos_targets.get(jn, qpos0) for jn, qpos0 in zip(joint_names, qpos0_by_dof, strict=True)
                            ]),
                            np.asarray(qpos0_by_dof),
                        ).tolist()
                    )

                if dofs_idx_local:
                    if self.num_envs == 1:
                        position = position[0]
                    obj_inst.control_dofs_position(
                        position=position,
                        dofs_idx_local=dofs_idx_local,
                    )

    def _simulate(self):
        for _ in range(self.scenario.decimation):
            self.scene_inst.step()

    def refresh_render(self):
        """Refresh camera and viewer geometry after a direct state change."""
        refresh_genesis_visualizer(self.scene_inst, headless=self.headless)

    def _get_effort_targets(self) -> torch.Tensor | None:
        """Get the effort targets from cached actions."""
        if not self._actions_cache or not self.robot:
            return None

        # Sorted order so joint_effort_target aligns with joint_pos/joint_vel
        # (see BaseSimHandler._joint_pos_target_from_action_cache). Values are name-keyed.
        joint_names = self._get_joint_names(self.robot.name, sort=True)
        effort_targets = []
        for action in self._actions_cache:
            if "dof_effort_target" in action[self.robot.name] and action[self.robot.name]["dof_effort_target"]:
                effort_values = [action[self.robot.name]["dof_effort_target"][jn] for jn in joint_names]
                effort_targets.append(effort_values)

        if effort_targets:
            return torch.tensor(effort_targets, dtype=torch.float32)
        return None

    def _get_control_mode(self, obj_name: str) -> str:
        """Get the control mode for the object."""
        control_type = getattr(self.object_dict[obj_name], "control_type", None) or {}
        control_types = list(set(control_type.values()))
        if len(control_types) > 1:
            raise ValueError(f"Multiple control types not supported: {control_types}")
        return control_types[0] if control_types else "position"

    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        obj_cfg = self.object_dict[obj_name]
        if isinstance(obj_cfg, (ArticulationObjCfg, RobotCfg)):
            joints: list[RigidJoint] = self.object_inst_dict[obj_name].joints
            # Exclude FREE and FIXED joints (e.g., floating base), and only include joints with DoFs
            joint_names = []
            for j in joints:
                try:
                    if j.type in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED):
                        continue
                except Exception:
                    pass
                try:
                    dofs = getattr(j, "dofs_idx_local", None)
                    if dofs is None or len(dofs) == 0:
                        continue
                except Exception:
                    continue
                joint_names.append(j.name)

            if sort:
                joint_names.sort()

            return joint_names
        else:
            return []

    @staticmethod
    def _joint_dof_layout(obj_inst, joint_names: list[str]) -> tuple[list[int], list[float], dict[str, float]]:
        """Return the named DOF indices and Genesis qpos0 offsets in joint order."""
        joints_by_name = {joint.name: joint for joint in obj_inst.joints}
        dofs_idx_local: list[int] = []
        qpos0_by_dof: list[float] = []
        qpos0_by_name: dict[str, float] = {}
        for joint_name in joint_names:
            joint = joints_by_name.get(joint_name)
            if joint is None or joint.dofs_idx_local is None:
                continue
            joint_dofs = list(joint.dofs_idx_local)
            if not joint_dofs:
                continue
            qpos0 = np.asarray(joint.desc.init_qpos, dtype=np.float32).reshape(-1)
            if qpos0.size == len(joint_dofs):
                offsets = qpos0.tolist()
            elif qpos0.size == 1:
                offsets = [float(qpos0[0])] * len(joint_dofs)
            else:
                offsets = [0.0] * len(joint_dofs)
            dofs_idx_local.extend(joint_dofs)
            qpos0_by_dof.extend(offsets)
            if len(offsets) == 1:
                qpos0_by_name[joint_name] = offsets[0]
        return dofs_idx_local, qpos0_by_dof, qpos0_by_name

    def _apply_robot_default_joint_positions(self, robot: RobotCfg) -> None:
        """Initialize a robot at its configured absolute joint positions."""
        if not robot.default_joint_positions:
            return

        obj_inst = self.object_inst_dict[robot.name]
        joint_names = self._get_joint_names(robot.name, sort=False)
        dofs_idx_local, qpos0_by_dof, _ = self._joint_dof_layout(obj_inst, joint_names)
        if not dofs_idx_local:
            return

        absolute_positions = [
            robot.default_joint_positions.get(joint_name, qpos0)
            for joint_name, qpos0 in zip(joint_names, qpos0_by_dof, strict=True)
        ]
        offsets = genesis_position_offsets(np.asarray(absolute_positions), np.asarray(qpos0_by_dof))
        if self.num_envs > 1:
            offsets = np.repeat(offsets[None, :], self.num_envs, axis=0)
        obj_inst.set_dofs_position(offsets, dofs_idx_local=dofs_idx_local)

    @property
    def num_envs(self) -> int:
        return self.scene_inst.n_envs

    @property
    def device(self) -> torch.device:
        return gs.device

    def close(self) -> None:
        """Close the simulation and clean up resources."""
        # Genesis keeps process-global state and only permits one initialized
        # runtime. Destroy it so a later handler can initialize another scene.
        gs.destroy()
        self.scene_inst = None
        self.object_inst_dict = {}
        self.camera_inst_dict = {}
        self._fisheye_cameras = {}

    @staticmethod
    def _sanitize_dof_array(values: np.ndarray | list | tuple | None, default: float = 0.0) -> np.ndarray | None:
        """Ensure DOF parameter arrays contain numeric values so Genesis solver can cast to float32."""
        if values is None:
            return None
        arr_obj = np.array(values, dtype=object)
        sanitized = []
        for item in arr_obj.flatten():
            if item is None or (isinstance(item, float) and math.isnan(item)):
                sanitized.append(default)
            else:
                sanitized.append(float(item))
        return np.array(sanitized, dtype=gs.np_float).reshape(arr_obj.shape)

    def _sanitize_joint_dofs(self, entity: RigidEntity | None) -> None:
        """Replace invalid joint DOF parameters that would break Genesis build."""
        if entity is None or not hasattr(entity, "joints"):
            return
        for joint in getattr(entity, "joints", []):
            if not hasattr(joint, "_dofs_frictionloss"):
                continue
            sanitized = self._sanitize_dof_array(joint.dofs_frictionloss, default=0.0)
            if sanitized is not None:
                joint._dofs_frictionloss = sanitized

    def _sanitize_urdf(self, urdf_path: str | None) -> str | None:
        """Removing empty <collision> nodes."""
        if urdf_path is None:
            return None

        try:
            src = Path(urdf_path)
            if not src.exists():
                return urdf_path
            tree = ET.parse(src)
            root = tree.getroot()
            changed = False

            for link in root.findall(".//link"):
                to_remove = []
                for coll in link.findall("collision"):
                    if coll.find("geometry") is None:
                        to_remove.append(coll)
                for coll in to_remove:
                    link.remove(coll)
                    changed = True

            if not changed:
                return urdf_path

            dst = src.with_suffix(".genesis.urdf")
            tree.write(dst, encoding="utf-8", xml_declaration=True)
            return str(dst)
        except Exception as e:
            log.warning(f"URDF sanitize failed for {urdf_path}: {e}. Using original file.")
            return urdf_path
