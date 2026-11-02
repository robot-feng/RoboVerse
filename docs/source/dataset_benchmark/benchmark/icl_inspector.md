# ICL live task inspection

`examples/viser/icl_inspector.py` connects the existing MetaSim `TaskViserWrapper`
to one task registered in RoboVerse. Run it from the RoboVerse repository root
after preparing that task's assets and simulator dependencies:

Install MetaSim's existing visualization extra in the active RoboVerse
environment to enable Viser and optional Rerun recording:

```powershell
uv pip install -e "packages/metasim[visualization]"
```

```powershell
python examples/viser/icl_inspector.py `
  --task icl.geniesim_benchmark `
  --sim mujoco `
  --task-kwargs '{"run_id":"g1op_s2r_sort_fruit"}'
```

Inspect the supplied G1 apple-on-plate scene through the same registered task
and joint-target controls:

```powershell
python examples/viser/icl_inspector.py `
  --task icl.geniesim_g1_apple_delivery `
  --sim mujoco `
  --task-kwargs '{"image_size":512}'
```

The initial version accepts one registered task and one backend per process.
`--list-tasks` lists task IDs discoverable through MetaSim's existing registry.
The task constructor receives `simulator=<backend>` and `headless=True`; its
remaining required values are passed through `--task-kwargs` as a JSON object.
Asset roots follow RoboVerse's normal `ROBOVERSE_DATA_DIR` / repository-root
rules.

The Viser panel provides:

- **Reset**: calls the actual task's `reset(seed=...)` and restores the joint
  target sliders from the returned simulator state.
- **Step**: sends the selected joint target through the task's actual
  `step(...)`, preserving its checker, reward, termination, and failure info.
- **Run / Pause**: repeatedly submits the current joint target at the backend's
  reported environment step interval (30 Hz when the backend does not report
  one); it stops on task termination or timeout.
- A selector for the task's returned camera observations, when the selected
  task/backend provides them, alongside the Viser 3D scene.

The first control adapter supports tasks that expose a bounded continuous Box
action and an ordered `control_joints` or module-level `CONTROL_JOINTS` contract.
Tasks with Cartesian, bimanual, discrete, or unnamed actions fail with an
explicit message instead of receiving guessed controls. Dynamic task/backend
switching and object dragging are not exposed; directly moving a rendered
object would bypass the physics task.

For timeline replay, pass an optional Rerun output path:

```powershell
python examples/viser/icl_inspector.py `
  --task icl.geniesim_benchmark `
  --sim mujoco `
  --task-kwargs '{"run_id":"g1op_s2r_sort_fruit"}' `
  --rerun-output examples/output/qa_runs/icl_inspector/sort_fruit.rrd `
  --rerun-sample-every 1
```

This records actual task poses and available camera RGB/depth observations with
MetaSim's existing `RerunVisualizer`. Replay with `rerun <path-to-file.rrd>`.
Keep these QA recordings outside `examples/output/ICL_Benchmark/`; creating a
Rerun file or opening the live panel is not a task-success or human-approval
claim. The Viser service listens on localhost by default. Select another local
port with `--port` if needed.

The CPU-only adapter contracts are tested with:

```powershell
python -m pytest tests/test_icl_viser_inspector.py -q
```
