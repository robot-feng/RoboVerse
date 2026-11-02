"""CPU-only contracts for the live ICL Viser task controls."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.viser.icl_inspector import IclTaskInspector, format_step_status, joint_action_layout
from metasim.task.registry import get_task_class


class _Server:
    def on_client_connect(self, callback):
        self.callback = callback
        return callback


class _GuiElement:
    def __init__(self, *, value=None):
        self.value = value
        self.callback = None

    def on_click(self, callback):
        self.callback = callback

    def on_update(self, callback):
        self.callback = callback

    def click(self):
        self.callback(None)


class _Gui:
    def __init__(self):
        self.elements = {}

    def add_folder(self, _label):
        return nullcontext()

    def add_text(self, label, *, initial_value):
        element = _GuiElement(value=initial_value)
        self.elements[label] = element
        return element

    def add_button(self, label):
        element = _GuiElement()
        self.elements[label] = element
        return element

    def add_dropdown(self, label, *, options, initial_value):
        element = _GuiElement(value=initial_value)
        element.options = options
        self.elements[label] = element
        return element

    def add_image(self, image, *, label):
        element = _GuiElement(value=image)
        element.image = image
        self.elements[label] = element
        return element

    def add_slider(self, label, *, initial_value, **_kwargs):
        element = _GuiElement(value=initial_value)
        self.elements[label] = element
        return element


class _Viewer:
    def __init__(self, task):
        self.task = task
        self.visualizer = SimpleNamespace(server=_Server())

    def step(self, action):
        return self.task.step(action)

    def reset(self, **kwargs):
        return self.task.reset(**kwargs)

    def close(self):
        pass


class _Task:
    control_joints = ("joint_a", "joint_b")

    def __init__(self):
        self.action_space = SimpleNamespace(low=np.array([-1.0, -2.0]), high=np.array([1.0, 2.0]))
        self.scenario = SimpleNamespace(simulator="mujoco", robots=[SimpleNamespace(name="robot")])
        self.handler = SimpleNamespace(
            env_step_s=0.01,
            _get_joint_names=lambda name, sort=True: ["joint_a", "joint_b"],
            get_states=lambda mode: self.state,
        )
        self.state = SimpleNamespace(
            robots={"robot": SimpleNamespace(joint_pos=torch.tensor([[0.2, -0.4]]))},
            cameras={
                "head_rgb": SimpleNamespace(rgb=torch.zeros((1, 4, 4, 3))),
                "overview": SimpleNamespace(rgb=torch.ones((1, 4, 4, 3))),
            },
        )
        self.steps = []
        self.reset_seeds = []

    def step(self, action):
        self.steps.append(np.asarray(action).copy())
        return (
            self.state,
            torch.tensor([0.5]),
            torch.tensor([False]),
            torch.tensor([False]),
            {"geniesim_checker_status": "running", "task_failure": torch.tensor([False])},
        )

    def reset(self, seed=0):
        self.reset_seeds.append(seed)
        return self.state, {}


def test_joint_action_layout_uses_task_joint_order_and_bounds():
    layout = joint_action_layout(_Task())
    assert layout.names == ("joint_a", "joint_b")
    np.testing.assert_array_equal(layout.low, [-1.0, -2.0])
    np.testing.assert_array_equal(layout.high, [1.0, 2.0])


def test_live_step_command_reaches_task_step_and_reports_checker_status():
    task = _Task()
    inspector = IclTaskInspector(task, seed=7, viewer=_Viewer(task))
    message = inspector._step_once()
    np.testing.assert_allclose(task.steps, [[0.2, -0.4]])
    assert "checker=running" in message
    assert "task_failure=False" in message


def test_reset_routes_through_task_env_and_restores_actual_joint_targets():
    task = _Task()
    inspector = IclTaskInspector(task, seed=23, viewer=_Viewer(task))
    inspector._command[:] = [0.9, 1.5]
    inspector._reset_gui([])
    assert task.reset_seeds == [23]
    np.testing.assert_allclose(inspector._command, [0.2, -0.4])


def test_viser_buttons_call_the_task_reset_and_step_contracts():
    task = _Task()
    viewer = _Viewer(task)
    inspector = IclTaskInspector(task, seed=31, viewer=viewer)
    gui = _Gui()
    viewer.visualizer.server.callback(SimpleNamespace(gui=gui))

    gui.elements["Step"].click()
    np.testing.assert_allclose(task.steps, [[0.2, -0.4]])
    assert "step=1" in gui.elements["Task status"].value

    gui.elements["Reset"].click()
    assert task.reset_seeds == [31]
    assert "reset(seed=31) completed" in gui.elements["Task status"].value


def test_viser_inspector_defaults_to_overview_camera_when_available():
    task = _Task()
    viewer = _Viewer(task)
    inspector = IclTaskInspector(task, viewer=viewer)
    gui = _Gui()
    viewer.visualizer.server.callback(SimpleNamespace(gui=gui))

    assert gui.elements["Task camera"].value == "overview"
    np.testing.assert_array_equal(gui.elements["Live task observation"].image, np.ones((4, 4, 3)))


def test_unbounded_or_unnamed_action_contract_fails_explicitly():
    task = _Task()
    task.control_joints = None
    task.action_space = SimpleNamespace(low=np.array([-np.inf]), high=np.array([np.inf]))
    with pytest.raises(ValueError, match="named joint action contract"):
        joint_action_layout(task)


def test_status_formatter_includes_result_and_failure_fields():
    result = (
        None,
        torch.tensor([1.0]),
        torch.tensor([True]),
        torch.tensor([False]),
        {"geniesim_checker_status": "failure", "task_failure": torch.tensor([True])},
    )
    text = format_step_status(result, 12)
    assert "step=12" in text
    assert "reward=1.000" in text
    assert "terminated=True" in text
    assert "task_failure=True" in text
    assert "checker=failure" in text


def test_cataloged_geniesim_task_resolves_without_starting_a_simulator():
    task_class = get_task_class("icl.geniesim_benchmark")
    assert task_class.__name__ == "GenieSimBenchmarkTask"
    assert task_class.supported_simulators == ("mujoco", "genesis", "newton")
