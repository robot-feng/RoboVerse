"""Render one prepared Genie Sim textured object through a MetaSim backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metasim.constants import PhysicStateType
from metasim.scenario.cameras import PinholeCameraCfg
from metasim.scenario.lights import DistantLightCfg, DomeLightCfg
from metasim.scenario.objects import PrimitiveCubeCfg, RigidObjCfg
from metasim.scenario.scenario import ScenarioCfg
from metasim.scenario.simulator_params import SimParamCfg
from metasim.utils.setup_util import get_handler
from tools.icl.prepare_geniesim_objects import _OBJECT_FORMAT, _OBJECT_FORMAT_V1, _OBJECT_FORMAT_V2


def _prepared_asset_resources(asset: Path) -> tuple[dict, list[Path]]:
    """Validate a prepared rigid asset and collect its render and collision resources."""
    provenance = json.loads((asset / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("format") not in {_OBJECT_FORMAT_V1, _OBJECT_FORMAT_V2, _OBJECT_FORMAT}:
        raise ValueError("Asset was not prepared by prepare_geniesim_objects.py")
    required = [asset / name for name in ("object.urdf", "object.xml", "provenance.json")]
    visuals = list(asset.glob("visual_*.obj"))
    if any(not path.is_file() for path in required) or not visuals:
        raise FileNotFoundError(f"Prepared object wrapper is incomplete: {asset}")
    resources = sorted(
        path for path in asset.iterdir()
        if path.is_file() and path.suffix.lower() in {".obj", ".stl", ".mtl", ".png", ".jpg", ".jpeg"}
    )
    return provenance, resources


def probe(simulator: str, asset: Path, output: Path, *, width: int = 640, height: int = 400) -> None:
    if simulator not in {"mujoco", "genesis", "newton"}:
        raise ValueError("Select mujoco, genesis or newton")
    asset, output = asset.resolve(), output.resolve()
    provenance, resources = _prepared_asset_resources(asset)
    obj = RigidObjCfg(
        name=provenance["asset_id"],
        urdf_path=str(asset / "object.urdf"),
        mjcf_path=str(asset / "object.xml"),
        extra_resources=[str(path) for path in resources],
        physics=PhysicStateType.RIGIDBODY,
        default_position=(0.0, 0.0, 0.10),
    )
    obj.file_type.update({"mujoco": "mjcf", "genesis": "mjcf", "newton": "urdf"})
    camera = PinholeCameraCfg(
        name="overview",
        pos=(0.38, -0.55, 0.40),
        look_at=(0.0, 0.0, 0.03),
        width=width,
        height=height,
        data_types=["rgb"],
    )
    floor = PrimitiveCubeCfg(
        name="floor",
        size=(1.2, 1.2, 0.04),
        color=(0.22, 0.25, 0.29),
        physics=PhysicStateType.GEOM,
        default_position=(0.0, 0.0, -0.02),
    )
    scenario = ScenarioCfg(
        objects=[obj, floor],
        cameras=[camera],
        simulator=simulator,
        num_envs=1,
        headless=True,
        lights=[DomeLightCfg(intensity=450), DistantLightCfg(intensity=1100, polar=30, azimuth=45)],
        sim_params=SimParamCfg(mujoco_use_scenario_lights=True),
    )
    handler = get_handler(scenario)
    try:
        handler.set_states([
            {
                "objects": {
                    obj.name: {"pos": torch.tensor(obj.default_position), "rot": torch.tensor(obj.default_orientation)},
                    "floor": {
                        "pos": torch.tensor(floor.default_position),
                        "rot": torch.tensor(floor.default_orientation),
                    },
                },
                "robots": {},
            }
        ])
        handler.refresh_render()
        rgb = handler.get_states(mode="tensor").cameras["overview"].rgb[0].detach().cpu().numpy()
        if rgb.shape != (height, width, 3) or rgb.dtype != "uint8":
            raise ValueError(f"Invalid backend RGB frame {rgb.shape}, {rgb.dtype}")
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(output)
        print(f"{simulator}: {provenance['asset_id']} RGB {width}x{height}; saved {output}")
    finally:
        handler.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    args = parser.parse_args()
    probe(args.sim, args.asset, args.output, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
