from Def import MPCSol, MPC_Cfg, GaitCfg, LegControllerCfg, FootPlacementCfg, MPCPlan
from Def import ControllerCfg, PeriodicReference
from Def import IDX_P, IDX_RPY, G, NX, NU
from Def import PERIOD_SHIFT_IDX, _PHASE_EPS
from Utils import rot_z, continuous_dynamics, discretize
from Interface import Interface, robot_params
from Reference_traj import ref_gen, ref_gen_periodic
from Gait import GaitScheduler, foot_trajectory, p_terrain
from Foot_placement import FootPlacement
import numpy as np
import mujoco
from scipy import sparse
import time
import osqp

class ConvexMPC:
    def __init__(self, cfg: MPC_Cfg, mass: float, Ib: np.ndarray,
                 period: float | None = None):     # [ADDED] gait period, for the phase variable
        self.cfg = cfg
        self.mass = mass
        self.Ib = np.asarray(Ib)
        self.Q = cfg.Q_diag
        self.R = cfg.R_diag.reshape((4,3))
        self.osqp_opt = dict(verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                                  max_iter=4000, polishing=False)
        self._last = None

        # ---- [ADDED] begin: phase / period constraint setup ----
        self.period = period
        self.n_period = None                       # N = T / dt  (steps per gait cycle)
        if cfg.period_constraint not in ("none", "soft", "hard"):
            raise ValueError(f"period_constraint must be none/soft/hard, got {cfg.period_constraint}")
        if cfg.period_pairs not in ("last", "section", "all"):
            raise ValueError(f"period_pairs must be last/section/all, got {cfg.period_pairs}")
        if cfg.period_constraint != "none":
            if period is None:
                raise ValueError("period constraint needs the gait period")
            n = period / cfg.dt
            if abs(n - round(n)) > 1e-9:
                raise ValueError(f"gait period {period} must be a multiple of mpc dt {cfg.dt}")
            self.n_period = int(round(n))
            if cfg.h < self.n_period + cfg.period_k0:
                raise ValueError(f"period constraint needs h >= T/dt + period_k0 = "
                                 f"{self.n_period + cfg.period_k0} (h = {cfg.h})")
        # ---- [ADDED] end ----

    def _model_prediction(self, X_ref: np.ndarray, r_foot: np.ndarray):
        h, dt = self.cfg.h, self.cfg.dt
        Sx = np.zeros((NX * h, NX))
        Su = np.zeros((NX * h, NU * h))
        A_prev = np.eye(NX)
        B_prev = np.zeros((NX, NU*h))
        for k in range(h):
            Ac, Bc = continuous_dynamics(X_ref[k,5], r_foot[k], self.mass, self.Ib)
            Ad, Bd = discretize(Ac, Bc, dt)
            A_block = Ad @ A_prev
            B_block = Ad @ B_prev
            B_block[ : , NU*k : NU * (k + 1)] = Bd
            Sx[NX*k : NX*(k+1)] = A_block
            Su[NX * k:NX * (k + 1)] = B_block
            A_prev = A_block
            B_prev = B_block

        return Sx, Su

    def _constraints(self, n_active: int):
        mu = self.cfg.mu
        C_f = np.array([[1.0, 0.0, -mu],
                        [-1.0, 0.0, -mu],
                        [0.0, 1.0, -mu],
                        [0.0, -1.0, -mu],
                        [0.0, 0.0, 1.0]])
        lb = np.array([-np.inf, -np.inf, -np.inf, -np.inf, self.cfg.fz_min])
        ub = np.array([0.0, 0.0, 0.0, 0.0, self.cfg.fz_max])
        # [CHANGED]: replace iteration with kron
        C = sparse.kron(sparse.eye(n_active), sparse.csc_matrix(C_f), format="csc")
        return C, np.tile(lb, n_active), np.tile(ub, n_active)

    # ---- [ADDED] begin: period constraint ----
    def _period_pair_starts(self) -> list[int]:
        """First index k of each constrained pair (k, k+N), see MPC_Cfg.period_pairs."""
        cfg, N = self.cfg, self.n_period
        last = cfg.h - N
        if cfg.period_pairs == "all":
            return list(range(cfg.period_k0, last + 1))
        if cfg.period_pairs == "section" and self._phase is not None:
            frac = np.abs(self._phase - np.round(self._phase))
            ks = [k for k in range(cfg.period_k0, last + 1) if frac[k] < 1e-6]
            if ks:
                return ks[:1]
        return [last]                       # "last", or no section in reach

    def _period_rows(self, x0: np.ndarray, X_ref: np.ndarray, Sx: np.ndarray,
                     Su: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear rows G U = g  for  x_{k+N} - Rot_k x_k = shift_k,  k = k0 .. h-N.
          * x, y, yaw (PERIOD_SHIFT_IDX): shift = X_ref[k+N] - X_ref[k]
          * xy parts of v and w: rotated by the reference yaw change over one
            period (periodic in the heading frame), shift = 0
          * other entries: shift = 0
        x_0 is the measured state (constant), x_j = Sx_j x0 + Su_j U for j >= 1.
        Returns (G (m, 12h), g (m,)), full input columns.
        """
        cfg, N = self.cfg, self.n_period
        idx = list(cfg.period_idx)
        G_list, g_list = [], []
        for k in self._period_pair_starts():
            a, b = k, k + N
            dpsi = X_ref[b, 5] - X_ref[a, 5]
            Rot = np.eye(NX)
            c, s_ = np.cos(dpsi), np.sin(dpsi)
            for j in (6, 9):                       # v_xy, w_xy blocks
                Rot[j:j + 2, j:j + 2] = [[c, -s_], [s_, c]]
            D_b = np.eye(NX)[idx]                  # selects x_b
            D_a = -Rot[idx]                        # -Rot x_a
            shift = np.zeros(len(idx))
            for r, i in enumerate(idx):
                if i in PERIOD_SHIFT_IDX:
                    shift[r] = X_ref[b, i] - X_ref[a, i]
            rows_b = slice(NX * (b - 1), NX * b)
            G = D_b @ Su[rows_b]
            g = shift - D_b @ (Sx[rows_b] @ x0)
            if a == 0:
                g -= D_a @ x0
            else:
                rows_a = slice(NX * (a - 1), NX * a)
                G = G + D_a @ Su[rows_a]
                g -= D_a @ (Sx[rows_a] @ x0)
            G_list.append(G); g_list.append(g)
        return np.vstack(G_list), np.concatenate(g_list)
    # ---- [ADDED] end ----

    def solve(self, x0: np.ndarray, X_ref: np.ndarray, r_foot: np.ndarray,
              contacts: np.ndarray, phase0: float | None = None,
              r_hip: np.ndarray | None = None) -> MPCSol:
        """
        X_ref  (h+1, 13)  reference, row 0 = x0  (ref_gen returns (13, h+1): pass .T)
        phase0            [ADDED] gait phase of x0 (cycles, unwrapped); optional
        r_hip  (h, 4, 3)  [ADDED] hip point - CoM over the horizon; needed when
                          cfg.align_weight > 0 (leg line = r_hip - r_foot)
        """
        t_start = time.perf_counter()
        h = self.cfg.h
        assert X_ref.shape == (h+1, NX), f"The shape of reference state is {X_ref.shape}"
        assert contacts.shape == (h, 4), f"The shape of contact is {contacts.shape}"
        # [ADDED] phase variable along the horizon: phi_k = phi_0 + k dt / T
        self._phase = None
        if phase0 is not None and self.period is not None:
            self._phase = phase0 + self.cfg.dt * np.arange(h + 1) / self.period

        Sx, Su = self._model_prediction(X_ref, r_foot)
        active  = np.repeat(contacts.reshape(-1), 3)
        n_contact = int(active.sum()) // 3
        if n_contact == 0:                                      # Four swing legs
            return self._result(np.zeros((h, 4, 3)), Sx @ x0, "no_contact", t_start)
        Su_a = Su[:, active]
        Q_stacked = np.tile(self.Q, h)
        R_stacked = np.tile(self.R.reshape(-1), h)[active]
        err0 = Sx @ x0 - X_ref[1:].reshape(-1)

        QSu = Q_stacked[:, None] * Su_a
        P = Su_a.T @ QSu + np.diag(R_stacked)
        q = QSu.T @ err0
        if self.cfg.align_weight > 0.0 and r_hip is not None:          # [ADDED]
            P = P + self._align_cost(contacts, r_foot, r_hip)[np.ix_(active, active)]

        C, lb, ub = self._constraints(n_contact)

        # ---- [ADDED] begin: period constraint ----
        mode = self.cfg.period_constraint
        G = g = None
        if mode != "none":
            G_full, g = self._period_rows(x0, X_ref, Sx, Su)
            G = G_full[:, active]
        res, status = self._solve_qp(P, q, C, lb, ub, G, g, mode)
        if mode == "hard" and ("solved" not in status or res.x is None):
            res, status = self._solve_qp(P, q, C, lb, ub, G, g, "soft")   # infeasible -> relax
            status = f"{status}(hard->soft)"
        # ---- [ADDED] end ----

        if res.x is None or not np.all(np.isfinite(res.x)) or "solved" not in status:
            return self._fallback(x0, contacts, Sx, status, t_start)
        
        U = np.zeros(NU * h)
        U[active] = res.x
        forces = U.reshape(h, 4, 3)
        X_pred = (Sx @ x0 + Su @ U).reshape(h, NX)
        self._last = forces
        resid = float(np.abs(G @ res.x - g).max()) if G is not None else 0.0   # [ADDED]
        return MPCSol(forces, X_pred, status, time.perf_counter() - t_start,
                      self._phase, resid)                                        # [CHANGED] + phase, residual

    # ---- [ADDED] begin ----
    def _align_cost(self, contacts, r_foot, r_hip):
        """(12h, 12h) block diagonal: w (I - l l^T) for every stance leg and step."""
        h = self.cfg.h
        P = np.zeros((NU * h, NU * h))
        line = r_hip - r_foot                                   # (h, 4, 3) foot -> hip
        for k in range(h):
            for i in np.flatnonzero(contacts[k]):
                l = line[k, i] / np.linalg.norm(line[k, i])
                j = NU * k + 3 * i
                P[j:j + 3, j:j + 3] = self.cfg.align_weight * (np.eye(3) - np.outer(l, l))
        return P

    def _solve_qp(self, P, q, C, lb, ub, G, g, mode):
        """OSQP solve; adds the period rows as a penalty (soft) or equalities (hard)."""
        if mode == "soft":
            w = self.cfg.period_weight
            P = P + w * (G.T @ G)
            q = q - w * (G.T @ g)
        elif mode == "hard":
            C = sparse.vstack([C, sparse.csc_matrix(G)], format="csc")
            lb = np.concatenate([lb, g])
            ub = np.concatenate([ub, g])
        sol = osqp.OSQP()
        sol.setup(P=sparse.csc_matrix(np.triu(P)), q=q, A=C, l=lb, u=ub, **self.osqp_opt)
        res = sol.solve()
        return res, res.info.status
    # ---- [ADDED] end ----

    def _result(self, forces, X_flat, status, t_start):
        return MPCSol(forces, X_flat.reshape(self.cfg.h, NX), status,
                         time.perf_counter() - t_start, self._phase)   # [CHANGED] + phase

    def _fallback(self, x0, contacts, Sx, status, t_start):
        h = self.cfg.h
        forces = np.zeros((h, 4, 3))
        n = contacts.sum(axis=1, keepdims=True).clip(min=1)
        forces[..., 2] = contacts * (self.mass * -G) / n   # [FIXED] GRAVITY was undefined (NameError)
        return self._result(forces, Sx @ x0, f"fallback({status})", t_start)

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
    h = contacts_h.shape[0]
    # [FIXED] was "if shape[0] == NX: transpose", which is ambiguous when
    # h + 1 == 13 (h = 12). Decide from the expected (h+1, NX) shape.
    if Xref.shape == (NX, h + 1) and Xref.shape != (h + 1, NX):
        Xref = Xref.T
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

class LegController:
    def __init__(self, state: Interface, cfg: LegControllerCfg = None):
        self.state = state
        self.cfg = cfg if cfg is not None else LegControllerCfg()   # [FIXED] cfg=None crashed at cfg.kp_swing
        m = state.m
        self.fric = m.dof_damping[state.vadr]          # viscous damping [N m s/rad]
        # [FIXED] MjModel has no `dof_frictionless`; Coulomb joint friction is dof_frictionloss
        self.no_fric = m.dof_frictionloss[state.vadr]
        self.M_s = np.zeros((m.nv, m.nv))

    def torques(self, contacts: np.ndarray, forces: np.ndarray,
                swing_ref: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
            contacts  (4,) bool   True = stance
            forces    (4, 3)      MPC solved forces (world)
            swing_ref (4, 3, 3)   swing leg [pos; vel; acc] (world)
            J         (4, 6, nv)  foot Jacobians
            returns   (4, 3)      joint torques, 4 x [hip, thigh, calf]
        """
        st, cfg = self.state, self.cfg
        p_foot = st.foot_pos
        v_foot = J[:, :3, :] @ st.d.qvel
        bias = st.leg_bias      # gravity
        qdot = st.jnt_vel
        if not contacts.all():                 # [CHANGED] mass matrix only needed for swing legs
            mujoco.mj_fullM(st.m, st.d, self.M_s)   # [FIXED] Interface has .m/.d, not .model/.data

        tau = np.zeros((4,3))
        for i in range(4):
            # [FIXED] was J[i][:, dofs] -> (6, 3) incl. the angular rows; Ji.T @ f
            # and solve(Ji, .) need the (3, 3) translational block.
            Ji = J[i][:3][:, st.vadr[i]]
            if contacts[i]:
                tau[i] += -Ji.T @ forces[i] + bias[i]
            else:
                f = cfg.kp_swing * (swing_ref[i,0] - p_foot[i]) + cfg.kd_swing * (swing_ref[i,1] - v_foot[i])
                tau[i] = Ji.T @ f
                idx = st.vadr[i]
                M_leg = self.M_s[np.ix_(idx, idx)]
                tau[i] += M_leg @ np.linalg.solve(Ji, swing_ref[i, 2]) + bias[i]
            # [FIXED] joint damping/friction compensation was inside the swing
            # branch only. Stance legs need it most: uncompensated it removed
            # ~4 N/leg of the MPC's horizontal force (see earlier version).
            tau[i] += self.fric[i] * qdot[i] + self.no_fric[i] * np.tanh(qdot[i] / cfg.jnt_fric_coeff)

        return np.clip(tau, -st.tau_limit, st.tau_limit)

    def command(self, tau: np.ndarray):
        self.state.sim_ctrl(tau)   # [CHANGED] one place writes data.ctrl (clip + actuator order)

class QuadrupedController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 cfg: ControllerCfg, foot_placement: FootPlacement,
                 periodic_ref: PeriodicReference | None = None):   # [ADDED]
        self.model = model
        self.data = data
        self.cfg = cfg
        self.foot_placement = foot_placement
        self.periodic_ref = periodic_ref      # [ADDED] used when cfg.mpc_cfg.periodic_ref
        self.st = Interface(model, data)
        self.params = robot_params(model)

        self.sim_dt = model.opt.timestep
        self.step_per_mpc = int(round(self.cfg.mpc_cfg.dt / self.sim_dt))
        assert abs(self.step_per_mpc * self.sim_dt - self.cfg.mpc_cfg.dt) < 1e-9, \
            "mpc.dt must be a multiple of the sim timestep"    # [FIXED] self.cfg.mpc -> mpc_cfg

        if self.cfg.step_per_mpc:
            self.step_per_mpc = self.cfg.step_per_mpc
        self.gait = GaitScheduler(self.cfg.gait_cfg)
        # [FIXED] ConvexMPC needs the inertia; [ADDED] gait period for the phase variable
        self.mpc = ConvexMPC(self.cfg.mpc_cfg, self.params["mass"], self.params["I_body"],
                             period=self.cfg.gait_cfg.period)
        self.leg = LegController(self.st, self.cfg.leg_cfg)
        if self.cfg.mpc_cfg.periodic_ref and self.periodic_ref is None:      # [ADDED]
            raise ValueError("mpc_cfg.periodic_ref=True needs a PeriodicReference")
        # [ADDED] hip point per leg in the base frame (shoulder x, y from the
        # nominal foot; z of the hip joint). Given to foot placement policies.
        self.hip_point_b = self.params["foot_b"].copy()
        self.hip_point_b[:, 2] = self.st.hip_b[:, 2]
        if self.cfg.hip_point == "abduction":
            self.hip_point_b[:, 1] = self.st.hip_b[:, 1]
        elif self.cfg.hip_point != "thigh":
            raise ValueError(f"hip_point must be thigh/abduction, got {self.cfg.hip_point}")
        # [ADDED] contact switches must fall on MPC steps (see README)
        solve_dt = self.step_per_mpc * self.sim_dt
        off = np.mod(self.gait.boundaries() / solve_dt + 0.5, 1.0) - 0.5
        if np.any(np.abs(off) > 1e-6):
            print(f"warning: gait boundaries {np.round(self.gait.boundaries(), 4)} s are not "
                  f"multiples of the MPC step {solve_dt} s; contacts switch between MPC steps")
        self.reset()

    def reset(self):
        # [FIXED] was self.step = 0: an int attribute named `step` hides the
        # step() method, so ctrl.step(v) raised "'int' object is not callable".
        self.n_step = 0   # time step count
        self.prev_contacts = np.ones(4, dtype=bool)
        p = self.st.foot_pos
        self.lift_off = p.copy()              # lift-off positions (4, 3)
        self.foothold = p.copy()
        self.forces = np.zeros((4,3))
        # [FIXED] was None: init_mpc_info() calls MPCSol.empty_like(None) and
        # copy_into(None) fails at the first solve.
        self.mpc_result = MPCSol.zeros(self.cfg.mpc_cfg.h)
        # [ADDED] gait clock offset [s]: gait time = t - t_stand - clock_offset.
        # Shifting it perturbs the gait phase (time-augmented Poincare / ROA).
        self.clock_offset = 0.0
        # [ADDED] index of the last MPC solve on the gait-clock grid (see step())
        self.mpc_tick = None
        self.plan = None                      # [ADDED] MPCPlan of the last solve (foot placement input)
        self.info_mpc = self.init_mpc_info()
        self.info_leg = self.init_leg_info()

    def get_state(self, info_dict: dict):
        info_dict["step"] = self.n_step
        np.copyto(info_dict["prev_contacts"], self.prev_contacts)
        np.copyto(info_dict["lift_off"], self.lift_off)
        np.copyto(info_dict["foothold"], self.foothold)
        np.copyto(info_dict["forces"], self.forces)
        self.mpc_result.copy_into(info_dict["mpc_result"])
        info_dict["clock_offset"] = self.clock_offset     # [ADDED]
        info_dict["mpc_tick"] = self.mpc_tick             # [ADDED]
        info_dict["plan"] = self.plan                     # [ADDED] never modified in place, so a reference is enough
        return info_dict                                  # [ADDED] convenience

    def set_state(self, s: dict):
        self.n_step = s["step"]
        np.copyto(self.prev_contacts, s["prev_contacts"])
        np.copyto(self.lift_off, s["lift_off"])
        np.copyto(self.foothold, s["foothold"])
        np.copyto(self.forces, s["forces"])
        s["mpc_result"].copy_into(self.mpc_result)
        self.clock_offset = s.get("clock_offset", 0.0)    # [ADDED]
        self.mpc_tick = s.get("mpc_tick", None)           # [ADDED]
        self.plan = s.get("plan", None)                   # [ADDED]

    def init_mpc_info(self) -> dict:
        return {
            "step": 0,
            "prev_contacts": np.empty_like(self.prev_contacts),
            "lift_off": np.empty_like(self.lift_off),
            "foothold": np.empty_like(self.foothold),
            "forces": np.empty_like(self.forces),
            "mpc_result": MPCSol.empty_like(self.mpc_result),
            "clock_offset": 0.0,                          # [ADDED]
            "mpc_tick": None,                             # [ADDED]
            "plan": None,                                 # [ADDED]
        }

    def init_leg_info(self) -> dict:
        return {
            "t": 0,
            "contacts": np.empty((4,), dtype=bool),
            "forces": np.empty_like(self.forces),
            "foothold": np.empty_like(self.foothold),
            "swing_ref": np.empty((4,3,3), dtype=float),
            "tau": np.zeros((4, 3)),                      # [CHANGED] array, copied each tick
            "v_cmd_w": np.empty((3,), dtype=float),
            "phase": 0.0,                                 # [ADDED]
        }

    @property
    def t_sim(self) -> float:
        return self.n_step * self.sim_dt

    def get_gait_time(self, t: float) -> float:
        return t - self.cfg.t_stand - self.clock_offset   # [CHANGED] + clock_offset

    # ---- [ADDED] begin ----
    def get_phase(self, t: float | None = None) -> float:
        """Unwrapped gait phase [cycles] (negative while standing)."""
        t = self.t_sim if t is None else t
        return self.get_gait_time(t) / self.cfg.gait_cfg.period

    def shift_phase(self, d_phase: float) -> float:
        """
        Perturb the gait clock by d_phase cycles (positive = clock ahead).
        Rounded to whole sim steps: a sub-step shift changes where the gait /
        MPC grid falls between sim steps and leaves a small permanent offset.
        Returns the phase shift actually applied.
        """
        n = round(d_phase * self.cfg.gait_cfg.period / self.sim_dt)
        self.clock_offset -= n * self.sim_dt
        return n * self.sim_dt / self.cfg.gait_cfg.period
    # ---- [ADDED] end ----

    def get_contacts(self, t: float) -> np.ndarray:
        t_gait = self.get_gait_time(t)
        # [FIXED] self.gait(t_gait) -> self.gait.contacts(t_gait) (GaitScheduler is not callable)
        return np.ones(4, dtype=bool) if t_gait < -_PHASE_EPS else self.gait.contacts(t_gait)

    def get_contacts_horizon(self, t: float) -> np.array:
        h = self.cfg.mpc_cfg.h
        dt = self.cfg.mpc_cfg.dt                          # [FIXED] self.cfg.mpc -> mpc_cfg
        contacts_horizon = np.ones((h, 4), dtype=bool)
        ts_gait = self.get_gait_time(t) + dt * np.arange(h)
        contacts_flag = ts_gait > -_PHASE_EPS
        if contacts_flag.any():
            contacts_start = int(np.argmax(contacts_flag))
            # [FIXED] was contacts_horizon[contacts_start] = (h - start, 4) block
            # -> shape error; the block fills rows start .. h-1.
            contacts_horizon[contacts_start:] = self.gait.contacts_horizon(ts_gait[contacts_start], h - contacts_start, dt)
        return contacts_horizon

    def get_cycle_index(self) -> int:
        t_gait = self.get_gait_time(self.t_sim)           # [FIXED] self.t does not exist
        return -1 if t_gait < -1e-9 else self.gait.cycle_index(t_gait)

    def _update_footholds(self, contacts, liftoff, elapsed, p_foot, base_pose,
                          v_base, w_base, v_cmd_w, force: bool = False):
        """
        [CHANGED] (moved out of step()) Stance legs: foothold = current foot.
        Swing legs: ask the foot placement policy, at lift-off, or every step
        if cfg.update_foothold_in_swing (or force=True). The policy gets the
        latest MPC plan in FootPlacementCfg.plan.
        """
        cfg = self.cfg
        yaw = base_pose[5]
        Rz = rot_z(yaw)
        p_hip_b = self.params["foot_b"]
        for i in range(4):
            if contacts[i]:
                self.foothold[i] = p_foot[i]
                continue
            if not (liftoff[i] or cfg.update_foothold_in_swing or force):   # [FIXED] used positions as a bool
                continue
            r_hip = Rz @ np.array([p_hip_b[i,0], p_hip_b[i,1], 0.])
            t_swing = self.gait.swing_duration(i)
            self.foothold[i] = self.foot_placement(
                FootPlacementCfg(
                leg=i, p_hip=base_pose[:3] + r_hip, r_hip=r_hip,
                v_base=v_base, w_base=w_base, v_cmd=v_cmd_w,
                height=base_pose[2], yaw=yaw,
                t_stance=self.gait.stance_duration(i), t_swing=t_swing,
                t_swing_remaining=max(t_swing - elapsed[i], 0.0),
                hip_b=self.hip_point_b[i], plan=self.plan)
            )

    def step(self, v_cmd_b: np.ndarray) -> np.ndarray:
        st = self.st
        t_sim = self.t_sim
        cfg = self.cfg
        v_cmd_b = np.asarray(v_cmd_b, dtype=float)        # [ADDED] lists/tuples
        base_pose = st.base_pose_w
        yaw = base_pose[5]
        Rz = rot_z(yaw)
        V = st.base_v_w
        v_base = V[:3]
        w_base = V[3:]
        p_foot = st.foot_pos
        J = st.foot_jac                           # [ADDED] LegController.torques needs J
        v_cmd_w = np.array([*(Rz[:2, :2] @ v_cmd_b[:2]), v_cmd_b[2]])

        contacts = self.get_contacts(t_sim)
        # [FIXED] lift-off events: was written into self.liftoff (undefined) and
        # mixed with the lift-off *positions* self.lift_off; positions were never
        # recorded and prev_contacts never updated, so swing legs started their
        # trajectory from the initial foot position forever.
        liftoff = ~contacts & self.prev_contacts
        self.lift_off[liftoff] = p_foot[liftoff]
        self.prev_contacts = contacts

        t_gait = self.get_gait_time(t_sim)
        elapsed = self.gait.swing_elapsed(t_gait) if t_gait >= -_PHASE_EPS else np.zeros(4)
        p_hip_b = self.params["foot_b"]
        foot_args = (contacts, liftoff, elapsed, p_foot, base_pose, v_base, w_base, v_cmd_w)
        self._update_footholds(*foot_args)

        terrain_z = p_terrain(None)[2]

        # [FIXED] was `self.t_sim % self.step_per_mpc == 0`: t_sim is a float
        # time, so this was true only at t = 0 (and at whole seconds).
        # [CHANGED] solve on the gait-clock grid (every step_per_mpc * dt of gait
        # time), not on the raw step counter. Same behaviour when clock_offset = 0,
        # but a gait phase shift now moves the MPC sampling with the contact
        # schedule. With the counter, a shift misaligned solves and contact
        # switches and the robot settled on a *different* limit cycle (a neutral
        # direction, eigenvalue ~1, in the time-augmented return map).
        solve_dt = self.step_per_mpc * self.sim_dt
        tick = int(np.floor(t_gait / solve_dt + 1e-6))
        if tick != self.mpc_tick:
            self.mpc_tick = tick
            if cfg.centroidal_state:
                x0 = st.mpc_state_centroidal(self.params["I_body"], self.params["com_b"])
            else:
                x0 = st.mpc_state(G)
            phase0 = self.get_phase(t_sim)                                   # [ADDED]
            if cfg.mpc_cfg.periodic_ref:                                     # [ADDED]
                X_ref = ref_gen_periodic(x0, phase0, v_cmd_b, cfg.mpc_cfg, self.periodic_ref)
            else:
                X_ref = ref_gen(x0, v_cmd_b, cfg.mpc_cfg)
            X_ref = X_ref.T   # [FIXED] ref_gen returns (13, h+1); ConvexMPC.solve asserts (h+1, 13)
            contacts_horizon = self.get_contacts_horizon(t_sim)
            # [ADDED] predicted hip point - CoM (for the MPC alignment cost)
            r_hip = np.stack([(self.hip_point_b - self.params["com_b"]) @ rot_z(xr[5]).T
                              for xr in X_ref[1:]])
            foot_prediction = predict_foot_positions(
                self.gait, t_gait, contacts_horizon, p_foot, self.foothold,
                X_ref, p_hip_b, self.params["com_b"], terrain_z
            )
            # [FIXED] was MPCSol.copy_into(self.mpc_result, new) = copy the OLD
            # result into the new one; the controller never saw a new solution.
            sol = self.mpc.solve(x0, X_ref, foot_prediction, contacts_horizon,
                                 phase0=phase0, r_hip=r_hip)                 # [CHANGED] + phase0, r_hip
            self.plan = MPCPlan(contacts_horizon, sol.forces, sol.X_pred,
                                cfg.mpc_cfg.dt, t_gait, sol.status)          # [ADDED]
            # [ADDED] fixed-point iteration for plan-aware placement policies:
            # new plan -> new swing footholds -> new forces ...
            if getattr(self.foot_placement, "uses_plan", False):
                for _ in range(cfg.placement_iters):
                    self._update_footholds(*foot_args, force=True)
                    foot_prediction = predict_foot_positions(
                        self.gait, t_gait, contacts_horizon, p_foot, self.foothold,
                        X_ref, p_hip_b, self.params["com_b"], terrain_z)
                    sol = self.mpc.solve(x0, X_ref, foot_prediction, contacts_horizon,
                                         phase0=phase0, r_hip=r_hip)
                    self.plan = MPCPlan(contacts_horizon, sol.forces, sol.X_pred,
                                        cfg.mpc_cfg.dt, t_gait, sol.status)
            sol.copy_into(self.mpc_result)
            np.copyto(self.forces, self.mpc_result.forces[0])   # [CHANGED] copy, not a view into mpc_result

        swing_ref = np.zeros((4,3,3))
        for i in range(4):
            if contacts[i]:
                swing_ref[i,0] = p_foot[i]
            else:
                # [FIXED] passed the full (4, 3) arrays lift_off / foothold; one leg's row is needed
                swing_ref[i] = foot_trajectory(
                    elapsed[i], self.lift_off[i], self.foothold[i], cfg.swing_cfg, self.gait.swing_duration(i)
                )

        tau = self.leg.torques(contacts, self.forces, swing_ref, J)   # [FIXED] self.legs -> self.leg, + J
        self.leg.command(tau)
        self.info_leg["t"] = t_sim
        np.copyto(self.info_leg["contacts"], contacts)
        np.copyto(self.info_leg["forces"], self.forces)
        np.copyto(self.info_leg["foothold"], self.foothold)
        np.copyto(self.info_leg["swing_ref"], swing_ref)
        np.copyto(self.info_leg["tau"], tau)              # [CHANGED] copy into the preallocated array
        np.copyto(self.info_leg["v_cmd_w"], v_cmd_w)
        self.info_leg["phase"] = self.get_phase(t_sim)    # [ADDED]
        self.n_step += 1   # [FIXED] was self.t_sim += 1 (read-only property, and it is a time, not a count)

        return tau
