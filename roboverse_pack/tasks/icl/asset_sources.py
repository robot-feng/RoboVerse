"""Pinned GenieSim rigid-asset sources used by the benchmark importer."""

DEFAULT_REVISION = "a0813aad7c16165daffbe6c2737754e0344809ee"
HISTORICAL_ASSETS = {
    "benchmark_garage_kit_064": {
        "revision": "26e4c900a49baa283b9ae08eaa168329566db8ab",
        "directory": "objects/benchmark/garage_kit/benchmark_garage_kit_064",
        "sha256": {
            "Aligned.usda": "a461e02cbcf6830c2fd6af98720527cd138b8fa842d6527815006ce06634e2ac",
            "Aligned.usd": "eff7178c7a60702e45d15812bf0bcf42eac5c53d5157296b1d94a668fe2f3836",
            "_Aligned.usd": "e52eea2988ffde676bc45b51059533a70334ec996a664cf408aef25c2e08296d",
            "item.py": "fabf5a95d7b955274acfb7f4956695477f95866679709fd29ad024c498d7f447",
            "object_parameters.json": "106aa7f9cf1e8f47aececec9b64d2ef4749fa2cd32faf9b71b053707f2dde240",
            "textures/Scan.jpg": "a6c291d7d2469f75217589684bf5b5cdffaf150e1f349b35ccbbf15ab67e4442",
        },
    },
}


def source_revision(asset_id: str) -> str:
    """Return the pinned official snapshot for this particular asset."""
    return HISTORICAL_ASSETS.get(asset_id, {}).get("revision", DEFAULT_REVISION)
