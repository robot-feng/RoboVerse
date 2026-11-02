"""Pure checks for the local ICL inventory builder's source-alias normalization."""

from tools.icl.build_phase1_inventory import _revision_for_env_alias


def test_imitator_source_level_aliases_normalize_to_registered_env_ids():
    assert _revision_for_env_alias("L0_TwoRobotPickAppleBasket-v1") == "TwoRobotPickAppleBasket-v1"
    assert _revision_for_env_alias("L2_TwoRobotPickAppleBasket-v1") == "TwoRobotPickAppleBasket-v1"
    assert _revision_for_env_alias("L3_TwoRobotPickAppleBasket-v1") == "TwoRobotPickAppleBasketL3-v1"


def test_unknown_env_ids_are_preserved():
    assert _revision_for_env_alias("CustomTask-v1") == "CustomTask-v1"
