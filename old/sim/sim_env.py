"""
[NEW FILE] MuJoCo model building (with SimCfg settings), simulation loop,
logging, snapshots and perturbations.

The loop uses mj_step1 / mj_step2 so the controller sees the kinematics
(Jacobians, foot positions) of the *current* state.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import mujoco
import numpy as np

from structs import SimCfg, LEGS, rot_z

FOOT_RADIUS = 0.022
_INTEGRATORS = {"euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST}
_CONES = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
          "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_model(cfg: SimCfg | None = None):
    """Build (model, data) from SimCfg and reset the robot to standing."""
    cfg = cfg or SimCfg()
    spec = mujoco.MjSpec.from_file(cfg.xml)
    if not cfg.obstacles:      # stock scene.xml has two bars and stairs
        for g in [g for g in spec.worldbody.geoms if g.name != "floor"]:
            spec.delete(g)
    model = spec.compile()
    apply_sim_cfg(model, cfg)
    data = mujoco.MjData(model)
    reset_robot(model, data, cfg.init_height_offset)
    return model, data


def contact_geoms(model) -> list[int]:
    return [model.geom("floor").id] + [model.geom(n).id for n in LEGS]


def apply_sim_cfg(model, cfg: SimCfg):
    """Write SimCfg into an existing MjModel (in place)."""
    opt = model.opt
    opt.timestep = cfg.timestep
    opt.gravity[:] = [0.0, 0.0, cfg.gravity]
    if cfg.integrator is not None:
        opt.integrator = _INTEGRATORS[cfg.integrator]
    if cfg.cone is not None:
        opt.cone = _CONES[cfg.cone]
    if cfg.impratio is not None:
        opt.impratio = cfg.impratio
    if cfg.noslip_iterations is not None:
        opt.noslip_iterations = cfg.noslip_iterations
    if cfg.solver_iterations is not None:
        opt.iterations = cfg.solver_iterations

    gids = contact_geoms(model)
    feet = gids[1:]
    if cfg.contact_stiffness is not None:            # direct mode: solref = (-k, -b)
        b = cfg.contact_damping if cfg.contact_damping is not None else 0.0
        model.geom_solref[gids] = [-cfg.contact_stiffness, -b]
    elif cfg.contact_solref is not None:             # standard mode: (timeconst, dampratio)
        tc = cfg.contact_solref[0]
        if tc < 2 * cfg.timestep:
            print(f"warning: solref timeconst {tc} < 2*timestep; MuJoCo will clamp it")
        model.geom_solref[gids] = cfg.contact_solref
    if cfg.contact_solimp is not None:
        model.geom_solimp[gids] = cfg.contact_solimp
    if cfg.friction is not None:
        model.geom_friction[gids, 0] = cfg.friction
    if cfg.condim is not None:
        model.geom_condim[feet] = cfg.condim
    if cfg.mass_scale != 1.0:
        model.body_mass[1:] *= cfg.mass_scale
        model.body_inertia[1:] *= cfg.mass_scale
    if cfg.torque_scale != 1.0:
        model.actuator_ctrlrange[:] *= cfg.torque_scale


def foot_friction(model) -> float:
    """Sliding friction MuJoCo uses for floor-foot contacts (max of the pair, or priority)."""
    f, g = model.geom("floor").id, model.geom(LEGS[0]).id
    pf, pg = model.geom_priority[f], model.geom_priority[g]
    if pf != pg:
        return float(model.geom_friction[f if pf > pg else g, 0])
    return float(max(model.geom_friction[f, 0], model.geom_friction[g, 0]))


def describe(model, cfg: SimCfg) -> str:
    f, g = model.geom("floor").id, model.geom(LEGS[0]).id
    return (f"timestep={model.opt.timestep} integrator={mujoco.mjtIntegrator(model.opt.integrator).name} "
            f"cone={mujoco.mjtCone(model.opt.cone).name} impratio={model.opt.impratio}\n"
            f"floor solref={model.geom_solref[f]} solimp={model.geom_solimp[f]}\n"
            f"foot  solref={model.geom_solref[g]} condim={model.geom_condim[g]} "
            f"contact friction={foot_friction(model)}  mass={model.body_subtreemass[1]:.2f} kg")


def reset_robot(model, data, height_offset: float = 0.0, key: int = 0):
    """Home keyframe, base lifted so the feet just touch the floor."""
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    feet = [model.geom(n).id for n in LEGS]
    data.qpos[2] += FOOT_RADIUS - data.geom_xpos[feet, 2].min() + height_offset
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------------
# Snapshots / perturbations
# --------------------------------------------------------------------------
def get_sim_state(model, data) -> np.ndarray:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    s = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, s, spec)
    return s


def set_sim_state(model, data, s: np.ndarray):
    mujoco.mj_setState(model, data, s, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)


def snapshot(model, data, ctrl) -> tuple:
    """(sim state, controller state) - restore with restore()."""
    return get_sim_state(model, data), ctrl.get_state(ctrl.init_mpc_info())


def restore(model, data, ctrl, snap):
    set_sim_state(model, data, snap[0])
    ctrl.set_state(snap[1])


def heading_yaw(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def push_velocity(model, data, dv: np.ndarray):
    """
    Instantaneous push: add a heading-frame base velocity change
    dv = [dvx, dvy, dvz, dwx, dwy, dwz] (joint velocities unchanged).
    """
    dv = np.asarray(dv, dtype=float)
    Rz = rot_z(heading_yaw(data))
    mujoco.mj_forward(model, data)
    R_wb = data.xmat[1].reshape(3, 3)
    data.qvel[0:3] += Rz @ dv[:3]
    data.qvel[3:6] += R_wb.T @ (Rz @ dv[3:6])     # free joint: body-frame angular velocity
    mujoco.mj_forward(model, data)


@dataclass
class ForcePush:
    """External force/torque on the base, world frame, during [t0, t0 + duration)."""
    t0: float
    duration: float
    force: tuple = (0.0, 0.0, 0.0)
    torque: tuple = (0.0, 0.0, 0.0)

    def apply(self, data):
        on = self.t0 <= data.time < self.t0 + self.duration
        data.xfrc_applied[1, :3] = self.force if on else 0.0
        data.xfrc_applied[1, 3:] = self.torque if on else 0.0


# --------------------------------------------------------------------------
# Logging + loop
# --------------------------------------------------------------------------
@dataclass
class SimLog:
    t: list = field(default_factory=list)
    qpos: list = field(default_factory=list)
    qvel: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    x_mpc: list = field(default_factory=list)      # centroidal MPC state (for periodic references)
    contacts: list = field(default_factory=list)
    forces: list = field(default_factory=list)
    footholds: list = field(default_factory=list)
    tau: list = field(default_factory=list)
    v_cmd: list = field(default_factory=list)
    mpc_status: list = field(default_factory=list)
    period_residual: list = field(default_factory=list)

    def append(self, data, ctrl, v_cmd):
        info = ctrl.info_leg
        self.t.append(data.time)
        self.qpos.append(data.qpos.copy())
        self.qvel.append(data.qvel.copy())
        self.phase.append(info["phase"])
        self.x_mpc.append(ctrl.st.mpc_state_centroidal(ctrl.params["I_body"], ctrl.params["com_b"]))
        self.contacts.append(info["contacts"].copy())
        self.forces.append(info["forces"].copy())
        self.footholds.append(info["foothold"].copy())
        self.tau.append(info["tau"].copy())
        self.v_cmd.append(np.asarray(v_cmd, dtype=float))
        self.mpc_status.append(ctrl.mpc_result.status)
        self.period_residual.append(ctrl.mpc_result.period_residual)

    def arrays(self) -> dict:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}

    def save(self, path: str, **extra):
        np.savez(path, **self.arrays(), **extra)


def is_fallen(data) -> bool:
    w, x, y, z = data.qpos[3:7]
    up_z = 1.0 - 2.0 * (x * x + y * y)          # z component of the body z axis
    return data.qpos[2] < 0.12 or up_z < 0.5


def run(model, data, ctrl, duration: float,
        v_cmd: Callable[[float], np.ndarray] | np.ndarray,
        log_every: int | None = 10, callbacks: list = (), pushes: list = (),
        viewer: bool = False, realtime: bool = True,
        stop_on_fall: bool = True) -> tuple[SimLog, bool]:
    """
    Run `ctrl` (QuadrupedController) for `duration` s of sim time.
        v_cmd     : [vx, vy, wz] (heading frame) or a function t -> that
        callbacks : f(model, data, ctrl) after each step; return True to stop
        pushes    : ForcePush list
        log_every : log each n steps (None = no log)
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
            for p in pushes:
                p.apply(data)
            mujoco.mj_step1(model, data)
            v = cmd(data.time)
            ctrl.step(v)
            mujoco.mj_step2(model, data)

            if log_every and n % log_every == 0:
                log.append(data, ctrl, v)
            if stop_on_fall and is_fallen(data):
                fallen = True
                break
            if any(cb(model, data, ctrl) for cb in callbacks):
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
