"""Prepare the pinned YCB objects required by The Imitator Game H7 task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from metasim.utils.hf_util import LOCAL_DIR
from tools.icl.prepare_assets import prepare_assets

DEFAULT_OUTPUT = Path(LOCAL_DIR) / "icl" / "objects" / "imitator_game"
YCB_OBJECTS = ("013_apple", "011_banana", "014_lemon")
SOURCE_APPLE_MASS_KG = 0.5


def _set_apple_mass(asset_dir: Path) -> None:
    """Apply H7's actor mass override to the generated wrappers and provenance."""
    urdf_path = asset_dir / "object.urdf"
    urdf_tree = ET.parse(urdf_path)
    urdf_masses = urdf_tree.findall(".//inertial/mass")
    urdf_inertias = urdf_tree.findall(".//inertial/inertia")
    if len(urdf_masses) != 1 or len(urdf_inertias) != 1:
        raise ValueError(f"Expected a single URDF inertial body in {urdf_path}")

    mjcf_path = asset_dir / "object.xml"
    mjcf_tree = ET.parse(mjcf_path)
    mjcf_inertias = mjcf_tree.findall(".//inertial")
    if len(mjcf_inertias) != 1:
        raise ValueError(f"Expected a single MJCF inertial body in {mjcf_path}")

    old_mass = float(urdf_masses[0].get("value"))
    if old_mass <= 0:
        raise ValueError("Generated apple mass must be positive")
    ratio = SOURCE_APPLE_MASS_KG / old_mass
    urdf_masses[0].set("value", str(SOURCE_APPLE_MASS_KG))
    for field in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        value = urdf_inertias[0].get(field)
        if value is not None:
            urdf_inertias[0].set(field, str(float(value) * ratio))
    urdf_tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    mjcf_inertia = mjcf_inertias[0]
    if not mjcf_inertia.get("mass"):
        raise ValueError(f"Missing MJCF inertial mass in {mjcf_path}")
    mjcf_inertia.set("mass", str(SOURCE_APPLE_MASS_KG))
    diagonal = mjcf_inertia.get("diaginertia")
    if diagonal:
        mjcf_inertia.set("diaginertia", " ".join(str(float(value) * ratio) for value in diagonal.split()))
    for field in ("fullinertia",):
        value = mjcf_inertia.get(field)
        if value:
            mjcf_inertia.set(field, " ".join(str(float(component) * ratio) for component in value.split()))
    mjcf_tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)

    provenance_path = asset_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["mass_kg"] = SOURCE_APPLE_MASS_KG
    provenance["mass_override"] = {
        "source_task": "human_H7 / TwoRobotPickAppleBasket-v1",
        "source_mass_kg": SOURCE_APPLE_MASS_KG,
        "description": "The source task sets the YCB apple actor mass to 0.5 kg.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def prepare_h7_ycb_assets(*, archive: Path, output: Path = DEFAULT_OUTPUT) -> list[Path]:
    """Prepare the three YCB assets without replacing existing user data.

    The shared RoboVerse YCB preparer verifies the pinned archive checksum and extracts only these
    model IDs. The task-specific apple wrapper then applies the source H7 mass override.
    """
    archive, output = Path(archive), Path(output)
    destination = output / "ycb"
    conflicts = [destination / name for name in YCB_OBJECTS if (destination / name).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to replace existing Imitator Game assets: {conflicts}")

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="imitator-h7-assets-", dir=output) as temporary:
        staged = Path(temporary) / "ycb"
        prepare_assets(archive=archive, output=staged, objects=list(YCB_OBJECTS))
        _set_apple_mass(staged / "013_apple")
        destination.mkdir(parents=True, exist_ok=True)
        prepared = []
        for name in YCB_OBJECTS:
            final = destination / name
            if final.exists():
                raise FileExistsError(f"Refusing to replace existing Imitator Game asset: {final}")
            (staged / name).rename(final)
            prepared.append(final)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Pinned IG-10K assets.tar.zst archive")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"Asset cache directory (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()
    for path in prepare_h7_ycb_assets(archive=args.archive, output=args.output):
        print(path)


if __name__ == "__main__":
    main()
