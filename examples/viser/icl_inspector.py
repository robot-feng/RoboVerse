"""Inspect one registered ICL task with live RoboVerse stepping in Viser.

Run from the RoboVerse repository root, for example:

    python examples/viser/icl_inspector.py --task icl.geniesim_benchmark --sim mujoco \
        --task-kwargs '{"run_id":"g1op_s2r_sort_fruit"}'

The scene is rendered by MetaSim's existing Viser wrapper. Joint sliders submit
actions through the selected task's ``step`` method so task checkers and episode
termination remain active; no visual-only pose editing is provided.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as log


@dataclass(frozen=True)
class JointActionLayout:
    """Joint labels and bounds for an ICL task's one-dimensional position action."""

    names: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray


def _control_names(task: Any) -> tuple[str, ...]:
    names = getattr(task, "control_joints", None)
    if names is None:
        module = importlib.import_module(type(task).__module__)
        names = getattr(module, "CONTROL_JOINTS", None)
    if names is None:
        raise ValueError(
            f"{type(task).__name__} does not expose a named joint action contract; "
            "this Viser inspector currently supports joint-target ICL tasks"
        )
    return tuple(str(name) for name in names)


def joint_action_layout(task: Any) -> JointActionLayout:
    """Validate the task's action space and return its joint labels and limits."""
    action_space = task.action_space
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise ValueError("Live joint controls require a bounded continuous action space")
    low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
    names = _control_names(task)
    if low.shape != high.shape or low.shape != (len(names),):
        raise ValueError(
            f"Joint action contract mismatch: action bounds have {low.size} values, named joints have {len(names)}"
        )
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Live joint controls require finite, strictly ordered action limits")
    return JointActionLayout(names=names, low=low, high=high)


def _vector_from_observation(task: Any, observation: Any, layout: JointActionLayout) -> np.ndarray:
    """Read current joint positions in the task's declared command order."""
    robot_name = task.scenario.robots[0].name
    if robot_name not in observation.robots:
        raise ValueError(f"Reset observation does not contain robot {robot_name!r}")
    joint_names = task.handler._get_joint_names(robot_name, sort=True)
    positions = observation.robots[robot_name].joint_pos[0].detach().cpu().numpy()
    state = dict(zip(joint_names, positions, strict=True))
    missing = set(layout.names) - set(state)
    if missing:
        raise ValueError(f"Task control joints are missing from the live state: {sorted(missing)}")
    return np.asarray([state[name] for name in layout.names], dtype=np.float64)


def format_step_status(result: tuple[Any, Any, Any, Any, Any], step: int) -> str:
    """Summarize task reward, termination and checker information for the UI."""
    _, reward, terminated, timeout, info = result
    reward_values = np.asarray(reward.detach().cpu() if hasattr(reward, "detach") else reward).reshape(-1)
    terminated_values = np.asarray(terminated.detach().cpu() if hasattr(terminated, "detach") else terminated).reshape(
        -1
    )
    timeout_values = np.asarray(timeout.detach().cpu() if hasattr(timeout, "detach") else timeout).reshape(-1)
    reward_value = float(reward_values[0]) if reward_values.size else 0.0
    is_terminated = bool(terminated_values[0]) if terminated_values.size else False
    is_timeout = bool(timeout_values[0]) if timeout_values.size else False
    checker = info.get("geniesim_checker_status") if isinstance(info, dict) else None
    failure = info.get("task_failure") if isinstance(info, dict) else None
    if hasattr(failure, "detach"):
        failure = failure.detach().cpu().reshape(-1).tolist()
    task_failure = bool(failure[0]) if isinstance(failure, (tuple, list)) and failure else bool(failure)
    return (
        f"step={step} reward={reward_value:.3f} terminated={is_terminated} timeout={is_timeout} "
        f"task_failure={task_failure} checker={checker or 'not exposed'}"
    )


class IclTaskInspector:
    """Connect Viser GUI commands to a real RoboVerse task environment."""

    def __init__(
        self,
        task: Any,
        *,
        seed: int = 0,
        port: int = 8080,
        viewer: Any | None = None,
        run_interval: float | None = None,
        rerun_recorder: Any | None = None,
    ) -> None:
        self.task = task
        self.seed = seed
        self.layout = joint_action_layout(task)
        if viewer is None:
            from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

            viewer = TaskViserWrapper(task, port=port, update_freq=1)
        self.viewer = viewer
        self._lock = threading.RLock()
        self._pause_event = threading.Event()
        self._run_thread: threading.Thread | None = None
        self._step_count = 0
        self._status_handles: list[Any] = []
        self._camera_images: list[tuple[Any, Any]] = []
        self._latest_observation: Any | None = None
        self._latest_result: tuple[Any, Any, Any, Any, Any] | None = None
        self._rerun_recorder = rerun_recorder
        step_seconds = getattr(task.handler, "env_step_s", None)
        self.run_interval = run_interval or (float(step_seconds) if step_seconds and step_seconds > 0 else 1 / 30)
        self._latest_observation = task.handler.get_states(mode="tensor")
        self._command = np.clip(
            _vector_from_observation(task, self._latest_observation, self.layout), self.layout.low, self.layout.high
        )
        self._register_gui()

    def _register_gui(self) -> None:
        server = self.viewer.visualizer.server

        @server.on_client_connect
        def setup_controls(client) -> None:
            with client.gui.add_folder("ICL Task Inspector"):
                task_info = client.gui.add_text(
                    "Selected task",
                    initial_value=f"{self.task.task_name if hasattr(self.task, 'task_name') else type(self.task).__name__}",
                )
                status = client.gui.add_text("Task status", initial_value="Ready")
                reset_button = client.gui.add_button("Reset")
                step_button = client.gui.add_button("Step")
                run_button = client.gui.add_button("Run")
                pause_button = client.gui.add_button("Pause")
                camera_names = self._camera_names()
                initial_camera = (
                    "overview" if "overview" in camera_names else (camera_names[0] if camera_names else "No task cameras")
                )
                camera_select = client.gui.add_dropdown(
                    "Task camera",
                    options=camera_names or ["No task cameras"],
                    initial_value=initial_camera,
                )
                camera_image = client.gui.add_image(
                    self._camera_frame(initial_camera) if camera_names else np.zeros((128, 128, 3), dtype=np.uint8),
                    label="Live task observation",
                )
                client.gui.add_text(
                    "Unavailable controls",
                    initial_value=(
                        "Task/backend switching and object dragging are not available in this session. "
                        "Use RoboVerse task actions; geometry edits would bypass physics."
                    ),
                )

                sliders = []
                for index, name in enumerate(self.layout.names):
                    value = float(np.clip(self._command[index], self.layout.low[index], self.layout.high[index]))
                    slider = client.gui.add_slider(
                        name,
                        min=float(self.layout.low[index]),
                        max=float(self.layout.high[index]),
                        step=float(max((self.layout.high[index] - self.layout.low[index]) / 400, 1e-4)),
                        initial_value=value,
                    )
                    slider.on_update(lambda _event, i=index, handle=slider: self._set_command(i, handle.value))
                    sliders.append(slider)

            self._status_handles.append(status)
            self._camera_images.append((camera_select, camera_image))
            reset_button.on_click(lambda _event: self._reset_gui(sliders))
            step_button.on_click(lambda _event: self._step_gui())
            run_button.on_click(lambda _event: self.start(status))
            pause_button.on_click(lambda _event: self.pause())
            camera_select.on_update(lambda _event: self._update_camera_for(camera_select, camera_image))
            self._update_camera_for(camera_select, camera_image)
            task_info.value = (
                f"{type(self.task).__name__} | sim={self.task.scenario.simulator} | "
                f"robots={','.join(robot.name for robot in self.task.scenario.robots)}"
            )

    def _camera_names(self) -> list[str]:
        if self._latest_observation is not None and getattr(self._latest_observation, "cameras", None):
            return list(self._latest_observation.cameras)
        try:
            states = self.task.handler.get_states(mode="tensor")
            return list(getattr(states, "cameras", {}))
        except Exception:
            return []

    def _camera_frame(self, name: str) -> np.ndarray:
        observation = self._latest_observation
        if observation is None:
            observation = self.task.handler.get_states(mode="tensor")
        camera = observation.cameras[name]
        rgb = camera.rgb[0]
        return rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)

    def _update_camera_for(self, select: Any, image: Any) -> None:
        name = select.value
        if name in self._camera_names():
            image.image = self._camera_frame(name)

    def _set_status(self, value: str, status: Any | None = None) -> None:
        if status is not None and not any(handle is status for handle in self._status_handles):
            self._status_handles.append(status)
        live = []
        for handle in self._status_handles:
            try:
                handle.value = value
                live.append(handle)
            except Exception:
                continue
        self._status_handles = live

    def _set_command(self, index: int, value: float) -> None:
        with self._lock:
            low, high = self.layout.low[index], self.layout.high[index]
            if not low <= float(value) <= high:
                raise ValueError(f"Target for {self.layout.names[index]} is outside task limits")
            self._command[index] = float(value)

    def _publish_observation(self, observation: Any) -> None:
        self._latest_observation = observation
        for select, image in tuple(self._camera_images):
            try:
                if select.value in getattr(observation, "cameras", {}):
                    rgb = observation.cameras[select.value].rgb[0]
                    image.image = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
            except Exception:
                continue
        if self._rerun_recorder is not None:
            self._rerun_recorder.add(observation, self._step_count)

    def _reset_gui(self, sliders: list[Any]) -> None:
        try:
            self.pause(wait=True)
            with self._lock:
                result = self.viewer.reset(seed=self.seed)
                observation = result[0] if isinstance(result, tuple) else result
                self._command = _vector_from_observation(self.task, observation, self.layout)
                self._publish_observation(observation)
                self._step_count = 0
                self._latest_result = None
                for index, slider in enumerate(sliders):
                    slider.value = float(self._command[index])
            self._set_status(f"reset(seed={self.seed}) completed; task checker reset through env.reset")
        except Exception as exc:
            self._set_status(f"Reset failed: {type(exc).__name__}: {exc}")

    def _step_once(self) -> str:
        with self._lock:
            if self._latest_result is not None:
                _, _, terminated, timeout, _ = self._latest_result
                if bool(
                    np.asarray(terminated.detach().cpu() if hasattr(terminated, "detach") else terminated).reshape(-1)[
                        0
                    ]
                ) or bool(np.asarray(timeout.detach().cpu() if hasattr(timeout, "detach") else timeout).reshape(-1)[0]):
                    raise RuntimeError("Episode is terminated; reset before stepping again")
            result = self.viewer.step(self._command.astype(np.float32, copy=True))
            self._latest_result = result
            observation = result[0]
            self._step_count += 1
            self._publish_observation(observation)
            return format_step_status(result, self._step_count)

    def _step_gui(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            self._set_status("Pause the run loop before a manual step")
            return
        try:
            self._set_status(self._step_once())
        except Exception as exc:
            self._set_status(f"Step failed: {type(exc).__name__}: {exc}")

    def start(self, status: Any | None = None) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            self._set_status("Run loop is already active", status)
            return
        self._pause_event.clear()
        self._run_thread = threading.Thread(target=self._run_loop, args=(status,), daemon=True)
        self._run_thread.start()
        self._set_status("Running task through env.step", status)

    def _run_loop(self, status: Any | None) -> None:
        while not self._pause_event.is_set():
            start = time.monotonic()
            try:
                message = self._step_once()
                self._set_status(message, status)
                if self._latest_result is not None:
                    _, _, terminated, timeout, _ = self._latest_result
                    done = np.asarray(
                        terminated.detach().cpu() if hasattr(terminated, "detach") else terminated
                    ).reshape(-1)
                    timed_out = np.asarray(timeout.detach().cpu() if hasattr(timeout, "detach") else timeout).reshape(
                        -1
                    )
                    if (done.size and bool(done[0])) or (timed_out.size and bool(timed_out[0])):
                        self._pause_event.set()
                        break
            except Exception as exc:
                self._set_status(f"Run stopped: {type(exc).__name__}: {exc}", status)
                self._pause_event.set()
                break
            self._pause_event.wait(max(0.0, self.run_interval - (time.monotonic() - start)))
        self._set_status("Paused" if self._pause_event.is_set() else "Stopped", status)

    def pause(self, *, wait: bool = False) -> None:
        self._pause_event.set()
        worker = self._run_thread
        if wait and worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=max(1.0, self.run_interval * 2))
            if worker.is_alive():
                raise RuntimeError("Run loop did not stop before reset")

    def close(self) -> None:
        self.pause(wait=True)
        if self._rerun_recorder is not None:
            self._rerun_recorder.close()
        self.viewer.close()
        server = getattr(getattr(self.viewer, "visualizer", None), "server", None)
        if server is not None and hasattr(server, "stop"):
            server.stop()


class RerunTaskRecorder:
    """Record real task states and camera observations through MetaSim's Rerun utility."""

    def __init__(self, task: Any, viewer: Any, path: str, *, sample_every: int = 1) -> None:
        from metasim.utils.rerun.rerun_util import RerunVisualizer

        if sample_every <= 0:
            raise ValueError("sample_every must be positive")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.viewer = viewer
        self.sample_every = sample_every
        self._recorder = RerunVisualizer(app_name="RoboVerse ICL Inspector", spawn=False, save_path=path)
        self._timeline_step = 0
        states = task.handler.get_states(mode="tensor")
        self._recorder.visualize_scenario_items(
            task.scenario.objects, viewer._extract_states_from_obs(states, "objects")
        )
        self._recorder.visualize_scenario_items(task.scenario.robots, viewer._extract_states_from_obs(states, "robots"))

    def add(self, observation: Any, step: int) -> None:
        timeline_step = self._timeline_step
        self._timeline_step += 1
        if step % self.sample_every:
            return
        self._recorder.set_time(timeline_step)
        states = self.task.handler.get_states(mode="tensor")
        for kind in ("objects", "robots"):
            for name, state in self.viewer._extract_states_from_obs(states, kind).items():
                self._recorder.update_item_pose(name, state)
        for name, camera in getattr(observation, "cameras", {}).items():
            rgb = camera.rgb[0]
            rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
            depth = getattr(camera, "depth", None)
            if depth is not None:
                depth = depth[0]
                depth = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
            self._recorder.log_camera_image(name, rgb, depth)

    def close(self) -> None:
        self._recorder.close()


def _parse_task_kwargs(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--task-kwargs must decode to a JSON object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="icl.geniesim_benchmark")
    parser.add_argument("--sim", choices=("mujoco", "genesis", "newton"), default="mujoco")
    parser.add_argument("--task-kwargs", default="{}", help='JSON kwargs for the task (e.g. {"run_id":"..."})')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rerun-output", help="Optional .rrd recording path for offline replay")
    parser.add_argument("--rerun-sample-every", type=int, default=1)
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    from metasim.task.registry import get_task_class, list_tasks

    if args.list_tasks:
        sys.stdout.write("\n".join(list_tasks()) + "\n")
        return
    task_kwargs = _parse_task_kwargs(args.task_kwargs)
    task_cls = get_task_class(args.task)
    supported = getattr(task_cls, "supported_simulators", None)
    if supported is not None and args.sim not in supported:
        parser.error(f"Task {args.task!r} does not declare support for {args.sim!r}; supported: {supported}")
    task = task_cls(simulator=args.sim, headless=True, **task_kwargs)
    viewer = None
    recorder = None
    inspector = None
    try:
        observation, _ = task.reset(seed=args.seed)
        from metasim.utils.viser.viser_env_wrapper import TaskViserWrapper

        viewer = TaskViserWrapper(task, port=args.port, update_freq=1)
        if args.rerun_output:
            recorder = RerunTaskRecorder(task, viewer, args.rerun_output, sample_every=args.rerun_sample_every)
        inspector = IclTaskInspector(task, seed=args.seed, port=args.port, viewer=viewer, rerun_recorder=recorder)
        inspector._publish_observation(observation)
        log.info(f"Task: {args.task} | backend: {args.sim} | Viser: http://localhost:{args.port}")
        if args.rerun_output:
            log.info(f"Rerun recording: {args.rerun_output}")
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if inspector is not None:
            inspector.close()
        else:
            if recorder is not None:
                recorder.close()
            if viewer is not None:
                viewer.close()
            else:
                task.close()


if __name__ == "__main__":
    main()
