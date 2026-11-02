"""Prepare source-scaled YCB meshes and wrappers for The Imitator Game H19."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import trimesh

from metasim.utils.hf_util import LOCAL_DIR
from tools.icl.prepare_assets import prepare_assets

DEFAULT_OUTPUT = Path(LOCAL_DIR) / "icl" / "objects" / "imitator_game" / "h19"
YCB_VARIANTS = {
    "026_sponge": {"task_scale": 0.8, "mass_kg": 0.5, "mass_rule": "H19 source task sets sponge mass to 0.5 kg"},
    "036_wood_block": {
        "task_scale": 0.5,
        "mass_kg": 0.5,
        "mass_rule": "H19 source task sets L2 replacement mass to 0.5 kg",
    },
    "061_foam_brick": {
        "task_scale": 0.4,
        "mass_kg": None,
        "mass_rule": "1000 kg/m^3 YCB density times the volume of the scaled convex collision hull",
    },
}
SOURCE_ARCHIVE_SHA256 = "1a4a5a2e4b6b9866ab5d75969ae0c8cec7dadf57ee20a2907d694f43a4fd0d92"
SOURCE_REVISION = "2d7a339b27aff14a0780db4bd4d5e3a68c6358bc"
SOURCE_URL = "https://huggingface.co/datasets/imitator-game/IG-10K-Assets"
ASSET_LICENSE = "CC-BY-NC-4.0"


def _scale_obj_vertices(path: Path, scale: float) -> None:
    """Bake one source uniform scale into OBJ vertex coordinates without changing UV/material data."""
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "v" and len(fields) >= 4:
            scaled = [float(value) * scale for value in fields[1:4]]
            line = "v " + " ".join(f"{value:.12g}" for value in scaled)
            if len(fields) > 4:
                line += " " + " ".join(fields[4:])
        output.append(line)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def _set_task_mass_and_provenance(asset_dir: Path, source_id: str) -> None:
    """Bake the source task scale, mass, inertia and license into portable wrappers."""
    variant = YCB_VARIANTS[source_id]
    scale = float(variant["task_scale"])
    for name in ("textured.obj", "collision.obj"):
        _scale_obj_vertices(asset_dir / name, scale)

    collision = trimesh.load(asset_dir / "collision.obj", force="mesh", process=False)
    if collision.is_empty or not collision.is_watertight or collision.volume <= 0:
        raise ValueError(f"{source_id} has no finite, watertight convex collision hull")
    bounds = collision.bounds
    extents = collision.extents
    mass = float(variant["mass_kg"] or (1000.0 * collision.volume))
    ixx, iyy, izz = (mass / 12 * (extents[(index + 1) % 3] ** 2 + extents[(index + 2) % 3] ** 2) for index in range(3))

    urdf_path = asset_dir / "object.urdf"
    urdf_tree = ET.parse(urdf_path)
    urdf_masses = urdf_tree.findall(".//inertial/mass")
    urdf_inertias = urdf_tree.findall(".//inertial/inertia")
    if len(urdf_masses) != 1 or len(urdf_inertias) != 1:
        raise ValueError(f"Expected one inertial body in {urdf_path}")
    urdf_masses[0].set("value", f"{mass:.12g}")
    for field, value in (("ixx", ixx), ("iyy", iyy), ("izz", izz), ("ixy", 0), ("ixz", 0), ("iyz", 0)):
        urdf_inertias[0].set(field, f"{value:.12g}")
    urdf_tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    mjcf_path = asset_dir / "object.xml"
    mjcf_tree = ET.parse(mjcf_path)
    mjcf_inertials = mjcf_tree.findall(".//inertial")
    if len(mjcf_inertials) != 1:
        raise ValueError(f"Expected one inertial body in {mjcf_path}")
    mjcf_inertials[0].set("mass", f"{mass:.12g}")
    mjcf_inertials[0].set("diaginertia", f"{ixx:.12g} {iyy:.12g} {izz:.12g}")
    mjcf_tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)

    provenance_path = asset_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update({
        "source_repository": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_object": source_id,
        "asset_license": ASSET_LICENSE,
        "task_scale_baked_in": scale,
        "mass_kg": mass,
        "mass_rule": variant["mass_rule"],
        "collision": "convex hull of source collision.ply with source task scale baked into vertices",
        "inertia": "uniform box approximation from scaled convex-hull bounds",
        "bounds_m": bounds.tolist(),
    })
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def prepare_h19_ycb_assets(*, archive: Path, output: Path = DEFAULT_OUTPUT) -> list[Path]:
    """Prepare H19's three exact YCB variants in a task-local cache without overwriting data."""
    archive, output = Path(archive), Path(output)
    destination = output / "ycb"
    variants = tuple(YCB_VARIANTS)
    conflicts = [destination / source_id for source_id in variants if (destination / source_id).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to replace existing H19 source assets: {conflicts}")

    output.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="imitator-h19-assets-", dir=output) as temporary:
        staged_root = Path(temporary) / "ycb"
        prepare_assets(archive=archive, output=staged_root, objects=list(variants))
        for source_id in variants:
            _set_task_mass_and_provenance(staged_root / source_id, source_id)
        prepared = []
        for source_id in variants:
            final = destination / source_id
            if final.exists():
                raise FileExistsError(f"Refusing to replace existing H19 source asset: {final}")
            shutil.move(str(staged_root / source_id), final)
            prepared.append(final)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Pinned IG-10K simulator assets.tar.zst")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"Asset cache root (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    for path in prepare_h19_ycb_assets(archive=args.archive, output=args.output):
        print(path)


if __name__ == "__main__":
    main()
