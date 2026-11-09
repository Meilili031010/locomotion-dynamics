import numpy as np
from Def import NU, NX
from Def import SimCfg

def skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def continuous_dynamics(yaw: float, r_foot: np.ndarray, mass: float,
                        I_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ac (13, 13), Bc (13, 12) of the single-rigid-body model.
    r_feet: (4, 3) foot position minus CoM, world frame.
    """
    Rz = rot_z(yaw)
    I_w_inv = np.linalg.inv(Rz @ I_body @ Rz.T)

    Ac = np.zeros((NX, NX))
    Ac[0:3, 6:9] = np.eye(3)
    Ac[3:6, 9:12] = Rz.T
    Ac[8, 12] = 1.0                     # v_z_dot += g  (g state holds -9.81)

    Bc = np.zeros((NX, NU))
    for i in range(4):
        Bc[6:9, 3 * i:3 * i + 3] = np.eye(3) / mass
        Bc[9:12, 3 * i:3 * i + 3] = I_w_inv @ skew(r_foot[i])
    return Ac, Bc


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    # [FIXED comment] not Euler: exact zero-order hold expm([[A, B], [0, 0]] dt).
    # M^3 = 0 for this model, so I + M dt + (M dt)^2 / 2 is the full series.
    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc
    Mdt = M * dt
    E = np.eye(NX + NU) + Mdt + 0.5 * Mdt @ Mdt
    return E[:NX, :NX], E[:NX, NX:]

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
