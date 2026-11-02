# Cross-Simulator Support

MetaSim enables you to write code once and run it across multiple physics simulators without modifications. This is one of the core features that makes RoboVerse a unified platform for robot learning.

Newton can display collision-only URDF geometry when no visual shapes are
present. Set `newton_hide_collision_shapes=True` on a rigid-object config to
hide its Newton collision shapes while retaining collision behavior. The option
defaults to `False` and does not change other simulator backends.

For Newton's MuJoCo solver, `SimParamCfg(newton_mujoco_enable_multiccd=True)`
opts in to multiple contacts for convex meshes. This can improve face-to-face
contact stability; it requires MuJoCo contacts and leaves the existing solver
default unchanged when off. It does not make material and friction behavior
identical across engines.

MuJoCo scenarios may opt in to
`SimParamCfg(mujoco_actuator_limits_at_joint=True)`. For a robot joint with
exactly one unit-gear direct actuator, the backend applies the intersection of
the existing actuator and joint force ranges at the joint. At a fixed state,
the resulting joint actuator force is unchanged; the location of the clamp can
change implicit-integration dynamics for stiff drives. Passive, contact and
constraint forces are not included in this actuator-force limit. The option
defaults to `False`; unsupported transmissions, multiple actuators on a
converted joint and actuator-routed gravity compensation raise an error. It
does not implement a joint velocity limit.

---

## Supported Simulators

| Simulator | Backend | GPU Parallel | Best Use Case |
|-----------|---------|--------------|---------------|
| `mujoco` | MuJoCo | Via MJX | Fast prototyping, CPU training |
| `mjx` | MuJoCo MJX | Native | Massively parallel RL |
| `isaacsim` | Isaac Sim | Native | High-fidelity sim2real |
| `isaacgym` | Isaac Gym | Native | GPU-accelerated RL |
| `sapien3` | SAPIEN 3 | Native | Manipulation research |
| `sapien2` | SAPIEN 2 | - | Legacy support |
| `genesis` | Genesis | Native | Large-scale training |
| `pybullet` | PyBullet | - | Debugging, education |

---

## Basic Usage

### Switching Simulators

Change the simulator by setting the `simulator` parameter in your scenario:

```python
from metasim.scenario.scenario import ScenarioCfg

# Run with MuJoCo
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="mujoco",  # Change this to switch simulators
    num_envs=1,
)

# Or switch to Isaac Sim
scenario = ScenarioCfg(
    robots=["franka"],
    simulator="isaacsim",
    num_envs=1,
)
```

### Command Line Examples

Most scripts support the `--sim` argument to select the simulator:

```bash
# Run with MuJoCo
python scripts/advanced/replay_demo.py --sim=mujoco --task=StackCube

# Run with Isaac Gym
python scripts/advanced/replay_demo.py --sim=isaacgym --task=StackCube

# Run with SAPIEN 3
python scripts/advanced/replay_demo.py --sim=sapien3 --task=PickCube
```

---

## Considerations

### Feature Parity

Not all features are available across all simulators. Check the [Support Matrix](support_matrix.rst) for detailed compatibility information.
