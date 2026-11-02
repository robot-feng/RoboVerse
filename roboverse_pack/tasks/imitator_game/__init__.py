"""RoboVerse task adapters for The Imitator Game."""

from .clean_desk import (
    ASSET_ROOT as H19_ASSET_ROOT,
)
from .clean_desk import (
    LEVEL_SPECS as H19_LEVEL_SPECS,
)
from .clean_desk import (
    LOCAL_SOURCE_TREE_SHA256 as H19_LOCAL_SOURCE_TREE_SHA256,
)
from .clean_desk import (
    UPSTREAM_SHA as H19_UPSTREAM_SHA,
)
from .clean_desk import (
    ImitatorGameCleanDeskL0,
    ImitatorGameCleanDeskL1,
    ImitatorGameCleanDeskL2,
    ImitatorGameCleanDeskL3,
)
from .clean_desk import (
    build_scenario as build_clean_desk_scenario,
)
from .clean_desk import (
    validate_assets as validate_clean_desk_assets,
)
from .pick_apple_basket import (
    ASSET_ROOT,
    EMBODIMENT,
    LEVEL_SPECS,
    LOCAL_SOURCE_TREE_SHA256,
    SOURCE,
    SOURCE_EMBODIMENT,
    UPSTREAM_SHA,
    ImitatorGamePickAppleBasketL0,
    ImitatorGamePickAppleBasketL3,
    build_scenario,
    validate_assets,
)

__all__ = [
    "ASSET_ROOT",
    "EMBODIMENT",
    "H19_ASSET_ROOT",
    "H19_LEVEL_SPECS",
    "H19_LOCAL_SOURCE_TREE_SHA256",
    "H19_UPSTREAM_SHA",
    "LEVEL_SPECS",
    "LOCAL_SOURCE_TREE_SHA256",
    "SOURCE",
    "SOURCE_EMBODIMENT",
    "UPSTREAM_SHA",
    "ImitatorGameCleanDeskL0",
    "ImitatorGameCleanDeskL1",
    "ImitatorGameCleanDeskL2",
    "ImitatorGameCleanDeskL3",
    "ImitatorGamePickAppleBasketL0",
    "ImitatorGamePickAppleBasketL3",
    "build_clean_desk_scenario",
    "build_scenario",
    "validate_assets",
    "validate_clean_desk_assets",
]
