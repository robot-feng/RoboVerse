"""Registered RoboVerse task adapters for the SynthICL scene profiles."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from metasim.scenario.scenario import ScenarioCfg
from metasim.task.registry import register_task
from metasim.utils.math import quat_apply
from roboverse_pack.tasks.icl.fruit_sorting import TABLE_TOP_M, _world_aabb
from roboverse_pack.tasks.icl.g1_joint_task import G1JointPositionTask
from roboverse_pack.tasks.icl.icl_static_scenes import build_icl_static_scenario


@register_task("icl.synthicl_pick_and_place_gn")
class SynthICLPickAndPlaceTask(G1JointPositionTask):
    """Place one GraspNet source object on a fixed target in the SynthICL scene.

    The initial representative uses the source tea-box / bowl pair at the
    source scale. Its checker follows the source placement sampler: the moving
    object's center lies within the target's local XY bounds and its lower
    surface is within the configured vertical offset of the target's top.
    """

    max_episode_steps = 900
    VERTICAL_OFFSET_M = (-0.05, 0.02)
    FALL_HEIGHT_M = 0.25

    def __init__(
        self,
        scenario: ScenarioCfg | None = None,
        device=None,
        *,
        simulator: str = "mujoco",
        asset_root: Path | None = None,
        image_size: int = 256,
        headless: bool = True,
    ):
        if scenario is None:
            scenario = build_icl_static_scenario(
                simulator=simulator,
                scene_id="pick_and_place_gn",
                asset_root=asset_root,
                robot_profile="g1",
                image_size=image_size,
                headless=headless,
            )
        objects = {obj.name: obj for obj in scenario.objects if getattr(obj, "urdf_path", None)}
        if not {"source", "target"} <= objects.keys():
            raise ValueError("SynthICL pick-and-place requires source and target rigid-object roles")
        asset_metadata = {}
        for role, obj in objects.items():
            provenance_path = Path(obj.urdf_path).parent / "provenance.json"
            try:
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                bounds = provenance["bounds_m"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Object {role!r} has no valid bounds provenance at {provenance_path}") from exc
            if len(bounds) != 2 or any(len(axis) != 3 for axis in bounds):
                raise ValueError(f"Object {role!r} has invalid bounds provenance at {provenance_path}")
            values = torch.as_tensor(bounds, dtype=torch.float64)
            if not torch.isfinite(values).all() or torch.any(values[1] <= values[0]):
                raise ValueError(f"Object {role!r} has invalid bounds provenance at {provenance_path}")
            asset_metadata[role] = {"bounds_m": values}
        self._metadata = {
            "source": "source",
            "destination": "target",
            "objects": asset_metadata,
        }
        self._placement_valid = None
        super().__init__(scenario, device)

    def _observe_outcome(self, state) -> None:
        source = state.objects[self._metadata["source"]].root_state
        destination = state.objects[self._metadata["destination"]].root_state
        source_bounds = self._metadata["objects"]["source"]["bounds_m"].to(source.device, source.dtype)
        destination_bounds = self._metadata["objects"]["target"]["bounds_m"].to(destination.device, destination.dtype)
        source_lower, _ = _world_aabb(source[:, :7], source_bounds)
        _, destination_upper = _world_aabb(destination[:, :7], destination_bounds)

        inverse_destination_rotation = destination[:, 3:7].clone()
        inverse_destination_rotation[:, 1:] *= -1
        source_center_local = quat_apply(inverse_destination_rotation, source[:, :3] - destination[:, :3])
        in_target_xy = (
            (source_center_local[:, :2] >= destination_bounds[0, :2])
            & (source_center_local[:, :2] <= destination_bounds[1, :2])
        ).all(dim=1)
        vertical_offset = source_lower[:, 2] - destination_upper[:, 2]
        lower_offset, upper_offset = self.VERTICAL_OFFSET_M
        placed = in_target_xy & (vertical_offset >= lower_offset) & (vertical_offset <= upper_offset)
        self._placement_valid = placed
        self._inside_frames = torch.where(placed, self._inside_frames + 1, 0)
        self._success |= self._inside_frames >= 2
        self._failure |= (source[:, 2] < TABLE_TOP_M - self.FALL_HEIGHT_M) & ~self._success
