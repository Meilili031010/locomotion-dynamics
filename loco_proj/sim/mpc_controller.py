from structs import MPCSol, MPC_Cfg, GaitConfig
from structs import IDX_P, IDX_RPY, GRAVITY, NX, NU
from structs import rot_z, continuous_dynamics, discretize
import numpy as np
from ref_gen import ref_gen
from gait import GaitScheduler
from scipy import sparse
import time
import osqp

class ConvexMPC:
    def __init__(self, cfg: MPC_Cfg, mass: float, Ib: np.ndarray):
        self.cfg = cfg
        self.mass = mass
        self.Ib = np.asarray(Ib)
        self.Q = cfg.Q_diag
        self.R = cfg.R_diag.reshape((4,3))
        self.osqp_opt = dict(verbose=False, eps_abs=1e-5, eps_rel=1e-5,
                                  max_iter=4000, polishing=False)
        self._last = None

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
            B_block[ : , NU*k : NU * (k + 1)] = B
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

    def solve(self, x0: np.ndarray, X_ref: np.ndarray, r_foot: np.ndarray,
              contacts: np.ndarray) -> MPCSol:
        t_start = time.perf_counter()
        h = self.cfg.h
        assert X_ref.shape == (h+1, NX), f"The shape of reference state is {X_ref.shape}"
        assert contacts.shape == (h, 4), f"The shape of contact is {contacts.shape}"

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
        sol = osqp.OSQP()
        sol.setup(P=sparse.csc_martix(np.triu(P)), q=q, A=C, l=lb, u=ub, **self.osqp_opt)
        res = sol.solve()
        status = res.info.status

        if res.x is None or not np.all(np.isfinite(res.x)) or "solved" not in status:
            return self._fallback(x0, contacts, Sx, status, t_start)
        
        U = np.zeros(NU * h)
        U[active] = res.x
        forces = U.reshape(h, 4, 3)
        X_pred = (Sx @ x0 + Su @ U).reshape(h, NX)
        self._last = forces
        return MPCSol(forces, X_pred, status, time.perf_counter() - t_start)

    def _result(self, forces, X_flat, status, t_start):
        return MPCSol(forces, X_flat.reshape(self.cfg.h, NX), status,
                         time.perf_counter() - t_start)

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




