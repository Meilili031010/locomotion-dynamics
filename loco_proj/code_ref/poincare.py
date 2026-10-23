"""
[NEW FILE] Poincare (return map) analysis of the closed-loop trot.

Section  : start of each gait cycle = touchdown of the FL/RR pair
           (trot table row 0 begins), sampled once per period T.
Coords   : s = [z, roll, pitch, vx, vy, vz, wx, wy, wz]
           velocities in the heading (yaw) frame, so x, y, yaw drop out and
           a steady trot is a fixed point s* = P(s*).
           extended=True appends the touchdown feet (FL, RR) xy relative to
           the base, in the heading frame (+4 coords): this is where the foot
           placement policy writes its decision.
Stability: eigenvalues of DP(s*). |lambda| < 1 for all -> orbitally stable;
           max |lambda| is the per-cycle contraction factor.

DP is estimated two ways:
  jacobian_fd     central differences, perturbing the velocity coordinates
                  (z/roll/pitch cannot be perturbed without moving the feet)
  jacobian_lstsq  least squares over randomly pushed trajectories:
                  ds_{k+1} = A ds_k, using every section pair. Gives the full
                  square matrix -> eigenvalues.

Experiments are described by plain data (ExperimentSpec), so trials can run
in worker processes.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: threads only add overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import mujoco
import numpy as np

from controller import ControllerConfig, LocomotionController
from foot_placement import POLICIES
from mpc_solver import rot_z
from sim_env import make_model, run, get_sim_state, set_sim_state

BASE_NAMES = ["z", "roll", "pitch", "vx", "vy", "vz", "wx", "wy", "wz"]
FOOT_NAMES = ["FL_x", "FL_y", "RR_x", "RR_y"]
VEL_IDX = [3, 4, 5, 6, 7, 8]           # coordinates that jacobian_fd perturbs


# --------------------------------------------------------------------------
# Experiment description
# --------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    policy: str = "cheetah3"                       # key of foot_placement.POLICIES
    policy_kw: dict = field(default_factory=dict)  # e.g. {"k_fb": 0.03}
    v_cmd: tuple = (0.5, 0.0, 0.0)                 # [vx, vy, wz], heading frame
    cfg_kw: dict = field(default_factory=dict)     # ControllerConfig overrides
    extended: bool = False                         # add foothold coordinates
    t_ramp: float = 1.0                            # command ramp after standing

    @property
    def label(self):
        kw = ",".join(f"{k}={v}" for k, v in self.policy_kw.items())
        return f"{self.policy}({kw})"

    def build(self):
        model, data = make_model()
        ctrl = LocomotionController(model, data, ControllerConfig(**self.cfg_kw),
                                    POLICIES[self.policy](**self.policy_kw))
        return model, data, ctrl

    def command(self, t_stand: float):
        v = np.asarray(self.v_cmd, dtype=float)
        return lambda t: v * np.clip((t - t_stand) / self.t_ramp, 0.0, 1.0)

    @property
    def names(self):
        return BASE_NAMES + (FOOT_NAMES if self.extended else [])


# --------------------------------------------------------------------------
# Section coordinates
# --------------------------------------------------------------------------
def section_state(model, data, ctrl, extended: bool = False) -> np.ndarray:
    mujoco.mj_forward(model, data)     # kinematics of the current qpos (pure, no state change)
    st = ctrl.st
    pose = st.base_pose
    Rz = rot_z(pose[5])
    v_h = Rz.T @ st.base_lin_vel_world
    w_h = Rz.T @ st.base_ang_vel_world
    s = [pose[2], pose[3], pose[4], *v_h, *w_h]
    if extended:
        rel = (st.foot_positions[[0, 3]] - pose[:3]) @ Rz    # FL, RR in heading frame
        s += list(rel[:, :2].reshape(-1))
    return np.array(s)


def perturb_velocity(model, data, ds_vel: np.ndarray):
    """Add a heading-frame velocity change [dvx, dvy, dvz, dwx, dwy, dwz] to the base."""
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    Rz = rot_z(yaw)
    mujoco.mj_forward(model, data)
    R_wb = data.xmat[1].reshape(3, 3)
    data.qvel[0:3] += Rz @ ds_vel[:3]
    data.qvel[3:6] += R_wb.T @ (Rz @ ds_vel[3:6])          # free joint: body-frame w
    mujoco.mj_forward(model, data)


class SectionRecorder:
    """run() callback: records s (and optionally a restorable snapshot) at each section."""

    def __init__(self, extended=False, keep_snapshots=False, stop_after=None):
        self.extended = extended
        self.keep = keep_snapshots
        self.stop_after = stop_after
        self.S, self.t, self.snaps = [], [], []
        self._last_cycle = None

    def __call__(self, model, data, ctrl):
        cyc = ctrl.cycle_index()
        if cyc >= 0 and cyc != self._last_cycle:
            self._last_cycle = cyc
            self.S.append(section_state(model, data, ctrl, self.extended))
            self.t.append(data.time)
            if self.keep:
                self.snaps.append((get_sim_state(model, data), ctrl.get_state()))
        return self.stop_after is not None and len(self.S) >= self.stop_after


# --------------------------------------------------------------------------
# Simulation helpers (top-level functions so they can run in worker processes)
# --------------------------------------------------------------------------
def simulate_sections(spec: ExperimentSpec, n_sections: int, snapshot=None,
                      ds_vel: np.ndarray = None, keep_snapshots=False):
    """
    From rest (snapshot=None) or from a snapshot taken at a section, optionally
    apply a velocity push, and record the next n_sections section states.
    Returns (S (n, d), fallen, recorder). When starting from a snapshot, S[0] is
    the (pushed) state at that section.
    """
    model, data, ctrl = spec.build()
    period = ctrl.cfg.gait.period
    rec = SectionRecorder(spec.extended, keep_snapshots, stop_after=n_sections)
    if snapshot is None:
        duration = ctrl.cfg.t_stand + (n_sections + 1) * period
    else:
        sim_s, ctrl_s = snapshot
        set_sim_state(model, data, sim_s)
        ctrl.set_state(ctrl_s)
        if ds_vel is not None:
            perturb_velocity(model, data, ds_vel)
        rec._last_cycle = ctrl.cycle_index()
        rec.S.append(section_state(model, data, ctrl, spec.extended))
        rec.t.append(data.time)
        duration = (n_sections + 0.5) * period
    _, fallen = run(model, data, ctrl, duration, spec.command(ctrl.cfg.t_stand),
                    log_every=10**9, callbacks=[rec])
    return np.array(rec.S), fallen, rec


def find_limit_cycle(spec: ExperimentSpec, n_warmup: int = 16, n_avg: int = 6):
    """
    Trot from rest until the section state settles.
    Returns s_star (mean of the last n_avg sections), the snapshot at the last
    section, the full section history, and the spread (max abs deviation from
    s_star over the averaging window; small -> period-1 orbit reached).
    """
    S, fallen, rec = simulate_sections(spec, n_warmup + n_avg, keep_snapshots=True)
    if fallen or len(S) < n_warmup + n_avg:
        raise RuntimeError(f"{spec.label}: robot fell / no steady gait")
    tail = S[-n_avg:]
    s_star = tail.mean(axis=0)
    spread = np.abs(tail - s_star).max(axis=0)
    return s_star, rec.snaps[-1], S, spread


def _one_step(args):
    spec, snapshot, ds = args
    S, fallen, _ = simulate_sections(spec, 2, snapshot, ds)
    return None if fallen else (S[0], S[1])


def _trajectory(args):
    spec, snapshot, ds, n = args
    S, fallen, _ = simulate_sections(spec, n + 1, snapshot, ds)
    return None if fallen else S


def jacobian_fd(spec, snapshot, delta=None, workers=None):
    """
    Columns of DP for the velocity coordinates (central differences).
    Returns J (d, 6): row = output coordinate, column = perturbed VEL_IDX entry.
    """
    delta = np.array([0.03, 0.03, 0.02, 0.1, 0.1, 0.1]) if delta is None else delta
    jobs = []
    for j in range(6):
        for sgn in (+1, -1):
            ds = np.zeros(6)
            ds[j] = sgn * delta[j]
            jobs.append((spec, snapshot, ds))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one_step, jobs))
    d = len(spec.names)
    J = np.full((d, 6), np.nan)
    for j in range(6):
        rp, rm = res[2 * j], res[2 * j + 1]
        if rp is None or rm is None:
            continue
        (s0p, s1p), (s0m, s1m) = rp, rm
        J[:, j] = (s1p - s1m) / (s0p[VEL_IDX[j]] - s0m[VEL_IDX[j]])
    return J


def jacobian_lstsq(spec, snapshot, s_star, n_trials=24, n_steps=4, scale=None,
                   seed=0, workers=None, skip_first=True):
    """
    Fit ds_{k+1} = A ds_k (ds = s - s_star) over random pushes at the section.
    skip_first: drop the pair that starts at the push itself. Right after the
    push the base velocity changed but the legs did not, so that state is off
    the set of states the closed loop actually visits at the section.
    Returns A (d, d), the relative fit residual and the pairs used.
    """
    rng = np.random.default_rng(seed)
    scale = np.array([0.05, 0.05, 0.03, 0.15, 0.15, 0.15]) if scale is None else scale
    jobs = [(spec, snapshot, rng.uniform(-1, 1, 6) * scale, n_steps) for _ in range(n_trials)]
    with ProcessPoolExecutor(workers) as ex:
        trajs = [T for T in ex.map(_trajectory, jobs) if T is not None]
    k0 = 1 if skip_first else 0
    X = np.vstack([T[k0:-1] for T in trajs]) - s_star
    Y = np.vstack([T[k0 + 1:] for T in trajs]) - s_star
    # Y = X A^T  ->  A^T = lstsq(X, Y)
    At, *_ = np.linalg.lstsq(X, Y, rcond=None)
    A = At.T
    resid = np.linalg.norm(Y - X @ At) / max(np.linalg.norm(Y), 1e-12)
    return A, resid, (X, Y)


@dataclass
class PoincareResult:
    spec: ExperimentSpec
    s_star: np.ndarray
    spread: np.ndarray
    history: np.ndarray
    J_fd: np.ndarray
    A: np.ndarray
    fit_residual: float
    eigvals: np.ndarray

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.eigvals)))

    def summary(self) -> str:
        n = self.spec.names
        lines = [f"== {self.spec.label}  v_cmd={self.spec.v_cmd}",
                 "  fixed point s*: " + ", ".join(f"{k}={v:+.3f}" for k, v in zip(n, self.s_star)),
                 f"  orbit spread (max) = {self.spread.max():.2e}",
                 f"  lstsq fit residual = {self.fit_residual:.3f}",
                 "  |eig| sorted     = " + " ".join(f"{a:.3f}" for a in np.sort(np.abs(self.eigvals))[::-1]),
                 f"  spectral radius  = {self.spectral_radius:.3f}  "
                 f"({'stable' if self.spectral_radius < 1 else 'NOT stable'})",
                 "  dP/d(velocity) diag (fd): " + ", ".join(
                     f"{n[i]}={self.J_fd[i, j]:+.3f}" for j, i in enumerate(VEL_IDX))]
        return "\n".join(lines)


def analyze(spec: ExperimentSpec, workers=None, n_trials=24, n_steps=4) -> PoincareResult:
    s_star, snap, hist, spread = find_limit_cycle(spec)
    J = jacobian_fd(spec, snap, workers=workers)
    A, resid, _ = jacobian_lstsq(spec, snap, s_star, n_trials=n_trials, n_steps=n_steps,
                                 workers=workers)
    return PoincareResult(spec, s_star, spread, hist, J, A, resid, np.linalg.eigvals(A))


if __name__ == "__main__":
    import sys
    spec = ExperimentSpec(policy=sys.argv[1] if len(sys.argv) > 1 else "cheetah3")
    print(analyze(spec).summary())
