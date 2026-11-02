# Genie Sim G1 pack-in-bag checker case

This page records one source-verified semantic test for
`g1op_s2r_pack_in_supermarket`, instance `pack_in_supermarket/0`. It is a
checker-only unit test; it is not a physics run or a successful robot demo.

## Source and provenance

The reduced source graph fixture is from the MPL-2.0 Genie Sim benchmark at
commit `a2bbec0c0e2d05f7cb5c13c9a75bcbc16ea229be`:

| Source file | SHA-256 |
| --- | --- |
| `config/g1op_s2r_pack_in_supermarket.yaml` | `16bed9a5fe0bdc3c1e11775b08c6ac21a7411d89845bdef1a3d2dbd28b1dd06b` |
| `benchmark/config/llm_task/pack_in_supermarket/0/problems.json` | `8f030c9350b74b857b54e5146c452b136507580175de5680e56482f44be31c7d` |
| `benchmark/config/llm_task/pack_in_supermarket/0/instructions.json` | `c467e49757df63b3feb08bed6f8ef4ff84f00c6df82cafff04b6f40e9aede75d` |
| `benchmark/config/llm_task/pack_in_supermarket/0/scene_info.json` | `c139043d11198b36860c4ca93d32928c3214b353651c150c2255eb9478329c71` |

The task YAML assigns the G1 omnipicker to `table_task_g1_op` and
`pack_in_supermarket`. The source instruction is “Pack objects on the desk”.
The checker graph names the green apple `benchmark_green_apple_42d8457e`, the
handbag `benchmark_handbag_4eac07c6`, and `right_gripper`. It requires, in
sequence, proximity to the apple, lifting that apple on the right gripper, and
the apple center remaining inside the handbag's scaled world AABB for two
consecutive samples. The graph contains a `StepOut` failure alternative at
each stage.

The semantic test constructs synthetic `CheckerFrame`s from these source
checker predicates. It verifies the successful sequence, the wrong-hand and
wrong-container negative cases, and the first-stage timeout. `Inside` currently
uses a bounding box; this test does not establish that the source handbag's
physical cavity, contact behavior, or release dynamics are reproduced.

Run the unit check from the RoboVerse package root:

```powershell
python -m pytest -q tests/test_icl_geniesim_pack_task.py
python -m ruff check tests/test_icl_geniesim_pack_task.py
```

## Local source snapshot caveat

The local Genie Sim source directory supplied for the broader migration is an
export without Git metadata. Its 86 task YAML files match the SHA-256 values in
the historical RoboVerse catalog, but the catalog's old revision
`226804caba35d8021f5d7c2b2a6701746619aa5f` cannot be fetched from the current
upstream repository. The current upstream main commit is
`a2bbec0c0e2d05f7cb5c13c9a75bcbc16ea229be`; all 86 YAML hashes differ from the
local export. Therefore this test pins its own verified current-upstream
fixture and does not claim the local 86-run import came from that revision.
The full source-import provenance remains under review.
