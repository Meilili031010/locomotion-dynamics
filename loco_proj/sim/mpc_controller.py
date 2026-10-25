from structs import MPCSol, MPC_Cfg, GaitCfg
from structs import IDX_P, IDX_RPY, GRAVITY, NX, NU
from structs import PERIOD_SHIFT_IDX   # [ADDED]
from structs import rot_z, continuous_dynamics, discretize
import numpy as np
from ref_gen import ref_gen
from gait import GaitScheduler
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
              contacts: np.ndarray, phase0: float | None = None) -> MPCSol:
        """
        X_ref  (h+1, 13)  reference, row 0 = x0  (ref_gen returns (13, h+1): pass .T)
        phase0            [ADDED] gait phase of x0 (cycles, unwrapped); optional
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
        # [FIXED] sparse.csc_martix -> sparse.csc_matrix (AttributeError on every solve)
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
        forces[..., 2] = contacts * (self.mass * -GRAVITY) / n
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
