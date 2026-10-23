"""
[NEW CONTENT] Convex MPC for ground reaction forces, following
"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive
Control" (Di Carlo et al., IROS 2018), Sec. III-IV.

State (this project's layout, see ref_gen.py):
    x = [p(3), rpy(3), v(3), w_world(3), g]        13
Input:
    u = [f_FL, f_FR, f_RL, f_RR]  ground reaction forces on the robot, world frame
        (fz > 0 means the ground pushes the robot up)

Simplified dynamics (paper eq. 16, small roll/pitch):
    p_dot   = v
    rpy_dot = Rz(yaw)^T w
    v_dot   = sum(f_i) / m + [0, 0, g]
    w_dot   = I_w^-1 sum(r_i x f_i),   I_w = Rz I_body Rz^T
    g_dot   = 0
Discretized per step with a zero-order hold (matrix exponential), condensed
into a QP over the forces only, and solved with OSQP. Swing-leg forces are
removed from the decision vector (as in the Cheetah code), so a trot QP has
2 legs x 3 x h variables.
"""
from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from ref_gen import MPC_Config, ref_gen, IDX_P, IDX_RPY, GRAVITY
from gait import GaitConfig, GaitScheduler
from state import State

NX, NU = 13, 12


def skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def continuous_dynamics(yaw: float, r_feet: np.ndarray, mass: float,
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
        Bc[9:12, 3 * i:3 * i + 3] = I_w_inv @ skew(r_feet[i])
    return Ac, Bc


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Zero-order hold: expm([[A, B], [0, 0]] dt).
    For this model M^3 = 0 (u, g -> v -> p and u -> w -> rpy are at most two
    integrators deep), so the series I + M dt + M^2 dt^2 / 2 is exact and
    much faster than scipy.linalg.expm.
    """
    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc
    Mdt = M * dt
    E = np.eye(NX + NU) + Mdt + 0.5 * Mdt @ Mdt
    return E[:NX, :NX], E[:NX, NX:]


@dataclass
class MPCResult:
    forces: np.ndarray        # (h, 4, 3) planned GRFs (zeros on swing legs)
    X_pred: np.ndarray        # (h, 13) predicted states x_1..x_h
    status: str
    solve_time: float


class ConvexMPC:
    """
    Usage:
        mpc = ConvexMPC(cfg, mass, I_body)
        res = mpc.solve(x0, Xref, r_feet, contacts)
        f_now = res.forces[0]          # (4, 3) apply until the next solve
    """

    def __init__(self, cfg: MPC_Config, mass: float, I_body: np.ndarray):
        self.cfg = cfg
        self.mass = mass
        self.I_body = np.asarray(I_body)
        self.Q_diag = np.asarray(cfg.Q_diag, dtype=float)
        self.R_leg = np.asarray(cfg.R_diag, dtype=float).reshape(4, 3)
        self.osqp_settings = dict(verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                                  max_iter=4000, polishing=False)
        self._last = None

    # ------------------------------------------------------------------ QP
    def _prediction_matrices(self, Xref: np.ndarray, r_feet: np.ndarray):
        """
        Stacked x_{1..h} = Sx x0 + Su U  with time-varying (A_k, B_k):
        yaw taken from the reference, foot arms from r_feet[k].
        """
        h, dt = self.cfg.h, self.cfg.dt
        Sx = np.zeros((NX * h, NX))
        Su = np.zeros((NX * h, NU * h))
        A_prev_row_x = np.eye(NX)
        prev_row_u = np.zeros((NX, NU * h))
        for k in range(h):
            Ac, Bc = continuous_dynamics(Xref[k, 5], r_feet[k], self.mass, self.I_body)
            A, B = discretize(Ac, Bc, dt)
            row_x = A @ A_prev_row_x
            row_u = A @ prev_row_u
            row_u[:, NU * k:NU * (k + 1)] = B
            Sx[NX * k:NX * (k + 1)] = row_x
            Su[NX * k:NX * (k + 1)] = row_u
            A_prev_row_x, prev_row_u = row_x, row_u
        return Sx, Su

    def _constraints(self, n_active: int):
        """Friction pyramid + normal force bounds for each active (stance) force."""
        mu = self.cfg.mu
        C_f = np.array([[1.0, 0.0, -mu],
                        [-1.0, 0.0, -mu],
                        [0.0, 1.0, -mu],
                        [0.0, -1.0, -mu],
                        [0.0, 0.0, 1.0]])
        lo = np.array([-np.inf, -np.inf, -np.inf, -np.inf, self.cfg.fz_min])
        hi = np.array([0.0, 0.0, 0.0, 0.0, self.cfg.fz_max])
        C = sparse.kron(sparse.eye(n_active), sparse.csc_matrix(C_f), format="csc")
        return C, np.tile(lo, n_active), np.tile(hi, n_active)

    def solve(self, x0: np.ndarray, Xref: np.ndarray, r_feet: np.ndarray,
              contacts: np.ndarray) -> MPCResult:
        """
        x0       (13,)      current state
        Xref     (13, h+1)  from ref_gen (column 0 = x0) or (h+1, 13)
        r_feet   (h, 4, 3)  foot minus CoM over the horizon, world frame
        contacts (h, 4)     bool, True = stance at step k
        """
        import time
        t_start = time.perf_counter()
        h = self.cfg.h
        Xref = np.asarray(Xref)
        if Xref.shape[0] == NX:
            Xref = Xref.T                       # -> (h+1, 13)
        contacts = np.asarray(contacts, dtype=bool)
        assert Xref.shape == (h + 1, NX) and contacts.shape == (h, 4)

        Sx, Su = self._prediction_matrices(Xref, r_feet)

        # keep only stance-leg force columns
        active = np.repeat(contacts.reshape(-1), 3)        # (12h,)
        n_act = int(active.sum()) // 3
        if n_act == 0:                                      # full flight phase
            return self._result(np.zeros((h, 4, 3)), Sx @ x0, "no_contact", t_start)
        Su_a = Su[:, active]

        Qbar = np.tile(self.Q_diag, h)                      # diagonal of block-diag Q
        Rbar = np.tile(self.R_leg.reshape(-1), h)[active]
        err0 = Sx @ x0 - Xref[1:].reshape(-1)

        QSu = Qbar[:, None] * Su_a
        P = Su_a.T @ QSu + np.diag(Rbar)
        q = QSu.T @ err0

        C, lo, hi = self._constraints(n_act)
        prob = osqp.OSQP()
        prob.setup(P=sparse.csc_matrix(np.triu(P)), q=q, A=C, l=lo, u=hi,
                   **self.osqp_settings)
        res = prob.solve()
        status = res.info.status

        if res.x is None or not np.all(np.isfinite(res.x)) or "solved" not in status:
            # fall back to the previous plan shifted by one step, or a static split
            return self._fallback(x0, contacts, Sx, status, t_start)

        U = np.zeros(NU * h)
        U[active] = res.x
        forces = U.reshape(h, 4, 3)
        X_pred = (Sx @ x0 + Su @ U).reshape(h, NX)
        self._last = forces
        return MPCResult(forces, X_pred, status, time.perf_counter() - t_start)

    # ------------------------------------------------------------ helpers
    def _result(self, forces, X_flat, status, t_start):
        import time
        return MPCResult(forces, X_flat.reshape(self.cfg.h, NX), status,
                         time.perf_counter() - t_start)

    def _fallback(self, x0, contacts, Sx, status, t_start):
        h = self.cfg.h
        forces = np.zeros((h, 4, 3))
        n = contacts.sum(axis=1, keepdims=True).clip(min=1)
        forces[..., 2] = contacts * (self.mass * -GRAVITY) / n
        return self._result(forces, Sx @ x0, f"fallback({status})", t_start)


# ---------------------------------------------------------------------------
# Foot positions over the horizon (needed for the B_k matrices)
# ---------------------------------------------------------------------------
def predict_foot_positions(gait: GaitScheduler, t_gait: float, contacts_h: np.ndarray,
                           p_feet_now: np.ndarray, footholds: np.ndarray,
                           Xref: np.ndarray, shoulder_b: np.ndarray,
                           com_b: np.ndarray, ground_z: float = 0.022) -> np.ndarray:
    """
    r_feet (h, 4, 3) = foot position - CoM over the horizon (world frame).

    Per leg, along the horizon (a foot never moves while in stance):
      * stance now                -> current foot position
      * swinging now, after its touchdown -> planned foothold (placement policy)
      * any later touchdown       -> nominal foothold, frozen at that touchdown:
        shoulder + v_ref * T_stance / 2 under the reference body (Raibert guess)
    Swing steps get a position too, but their force is zero so it is unused.
    The CoM is predicted from the reference base pose + body-frame CoM offset.
    """
    Xref = np.asarray(Xref)
    if Xref.shape[0] == NX:
        Xref = Xref.T
    h = contacts_h.shape[0]
    r = np.zeros((h, 4, 3))
    for leg in range(4):
        t_st = gait.stance_duration(leg)
        planned_used = bool(contacts_h[0, leg])
        foot = (p_feet_now[leg] if contacts_h[0, leg] else footholds[leg]).copy()
        for k in range(h):
            xr = Xref[k + 1]
            Rz = rot_z(xr[5])
            if k > 0 and contacts_h[k, leg] and not contacts_h[k - 1, leg]:   # touchdown
                if not planned_used:
                    foot = footholds[leg].copy()
                    planned_used = True
                else:
                    foot = xr[IDX_P] + Rz @ shoulder_b[leg]
                    foot[:2] += xr[6:8] * t_st / 2.0
                    foot[2] = ground_z
            r[k, leg] = foot - (xr[IDX_P] + Rz @ com_b)
    return r


if __name__ == "__main__":
    # Standalone check: standing robot, all feet down, zero velocity command.
    import mujoco
    from state import robot_params
    GO2_XML = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/scene.xml"
    model = mujoco.MjModel.from_xml_path(GO2_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    params = robot_params(model)
    st = State(model, data)

    cfg = MPC_Config()
    mpc = ConvexMPC(cfg, params["mass"], params["I_body"])
    x0 = st.mpc_state()
    Xref = ref_gen(x0, np.array([0.0, 0.0, 0.0]), cfg)
    contacts = np.ones((cfg.h, 4), dtype=bool)
    r = np.repeat((st.foot_positions - st.com_pos)[None], cfg.h, axis=0)
    out = mpc.solve(x0, Xref, r, contacts)
    print(out.status, f"{out.solve_time * 1e3:.1f} ms")
    print("f0 =\n", np.round(out.forces[0], 2))
    print("sum fz =", out.forces[0, :, 2].sum(), " weight =", params["mass"] * 9.81)

    gait = GaitScheduler(GaitConfig())
    contacts = gait.contacts_horizon(0.0, cfg.h, cfg.dt)
    r = predict_foot_positions(gait, 0.0, contacts, st.foot_positions, st.foot_positions,
                               Xref, params["foot_b"], params["com_b"])
    out = mpc.solve(x0, ref_gen(x0, np.array([0.5, 0.0, 0.0]), cfg), r, contacts)
    print("trot:", out.status, f"{out.solve_time * 1e3:.1f} ms")
    print("f0 =\n", np.round(out.forces[0], 2))
