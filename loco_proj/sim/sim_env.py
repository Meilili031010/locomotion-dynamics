"""
[NEW FILE] MuJoCo setup + simulation loop.

The loop uses mj_step1 / mj_step2 so the controller sees kinematics
(Jacobians, foot positions) of the *current* state, not the previous step.
"""
import time
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

GO2_SCENE = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/scene.xml"
FOOT_RADIUS = 0.022


def make_model(timestep: float = 0.001, xml: str = GO2_SCENE, obstacles: bool = False):
    """
    Go2 + floor. The stock unitree scene.xml also has two 8 cm bars
    (x = 1.2 m, 1.6 m) and stairs from x = 2.1 m; they are removed unless
    obstacles=True, so flat-ground gait studies are not hit by them.
    """
    spec = mujoco.MjSpec.from_file(xml)
    if not obstacles:
        for g in [g for g in spec.worldbody.geoms if g.name != "floor"]:
            spec.delete(g)
    model = spec.compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    reset_robot(model, data)
    return model, data


def reset_robot(model, data, key: int = 0):
    """Home keyframe, base lifted so the feet just touch the floor (no initial penetration)."""
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    feet = [model.geom(n).id for n in ("FL", "FR", "RL", "RR")]
    data.qpos[2] += FOOT_RADIUS - data.geom_xpos[feet, 2].min()
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def get_sim_state(model, data) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    s = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, s, spec)
    return s


def set_sim_state(model, data, s: np.ndarray):
    mujoco.mj_setState(model, data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)


@dataclass
class SimLog:
    t: list = field(default_factory=list)
    qpos: list = field(default_factory=list)
    qvel: list = field(default_factory=list)
    contacts: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    footholds: list = field(default_factory=list)
    tau: list = field(default_factory=list)
    v_cmd: list = field(default_factory=list)

    def append(self, data, info, v_cmd):
        self.t.append(data.time)
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.contacts.append(info["contacts"])
        self.forces.append(info["forces"])
        self.footholds.append(info["footholds"])
        self.tau.append(info["tau"])
        self.v_cmd.append(np.asarray(v_cmd, dtype=float))

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}

    def save(self, path: str):
        np.savez(path, **self.arrays())


def is_fallen(data) -> bool:
    w, x, y, z = data.qpos[3:7]
    up_z = 1.0 - 2.0 * (x * x + y * y)          # z component of the body z axis
    return data.qpos[2] < 0.12 or up_z < 0.5


def run(model, data, controller, duration: float,
        v_cmd: Callable[[float], np.ndarray] | np.ndarray,
        log_every: int = 10, callbacks: list = (), viewer: bool = False,
        realtime: bool = True, stop_on_fall: bool = True) -> tuple[SimLog, bool]:
    """
    Run `controller` for `duration` seconds of sim time.
        v_cmd     : [vx, vy, wz] (heading frame) or a function t -> that
        callbacks : f(model, data, controller) called after each step;
                    return True to stop the run early
    Returns (log, fallen).
    """
    cmd = v_cmd if callable(v_cmd) else (lambda t, c=np.asarray(v_cmd, float): c)
    log = SimLog()
    n_steps = int(round(duration / model.opt.timestep))
    fallen = False

    view = None
    if viewer:
        from mujoco import viewer as mj_viewer
        view = mj_viewer.launch_passive(model, data)
    try:
        for n in range(n_steps):
            wall0 = time.perf_counter()
            mujoco.mj_step1(model, data)
            v = cmd(data.time)
            controller.step(v)
            mujoco.mj_step2(model, data)

            if n % log_every == 0:
                log.append(data, controller.info, v)
            if stop_on_fall and is_fallen(data):
                fallen = True
                break
            if any(cb(model, data, controller) for cb in callbacks):
                break
            if view is not None:
                if not view.is_running():
                    break
                if n % 10 == 0:
                    view.sync()
                if realtime:
                    dt = model.opt.timestep - (time.perf_counter() - wall0)
                    if dt > 0:
                        time.sleep(dt)
    finally:
        if view is not None:
            view.close()
            time.sleep(0.1)
    return log, fallen
