import numpy as np
from Gait import compute_foot_placement, p_terrain
from Def import FootPlacementCfg, SwingTrajectoryCfg
from Utils import rot_z

class FootPlacement:
    name = "base"
    uses_plan = False      # [ADDED] True: the policy reads FootPlacementCfg.plan

    def __init__(self, max_stride: float = 0.2):
        self.max_stride = max_stride

    def __call__(self, cfg: FootPlacementCfg) -> np.ndarray:
        p = np.array(self._raw(cfg), dtype=float)
        hip_td = _hip_at_touchdown(cfg)
        d = p[:2] - hip_td[:2]
        stide_plan = np.linalg.norm(d)
        if stide_plan > self.max_stride:   # [FIXED] self.offset is not defined
            p[:2] = hip_td[:2] + d * self.max_stride / stide_plan
        p[2] = p_terrain(p)[2]
        return p

    def _raw(self, inp: FootPlacementCfg) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"

def _hip_at_touchdown(cfg: FootPlacementCfg) -> np.ndarray:
    return cfg.p_hip + np.array([cfg.v_cmd[0], cfg.v_cmd[1], 0.0]) * cfg.t_swing_remaining

class openloop(FootPlacement):
    name = "openloop"
    def _raw(self, cfg):
        p = _hip_at_touchdown(cfg)
        p[:2] += cfg.v_cmd[:2] * cfg.t_stance / 2.0
        return p

class Raibert(FootPlacement):
    def __init__(self, k: float = 0.03, max_offset: float = 0.2):
        super().__init__(max_offset)
        self.k = k
        self.name = f"Raibert_k{k:g}"   # [CHANGED] gain in the name, so results of a k sweep are distinguishable

    def _raw(self, cfg):
        p = _hip_at_touchdown(cfg)
        p[:2] += cfg.v_base[:2] * cfg.t_stance / 2.
        p[:2] += self.k * (cfg.v_base[:2] - cfg.v_cmd[:2])
        return p


class lipm(FootPlacement):
    def __init__(self, k: float = 0.03, max_offset: float = 0.2):
        super().__init__(max_offset)
        self.swing_cfg = SwingTrajectoryCfg(k_fb=k)
        self.name = f"lipm_k{k}"                     # [CHANGED] gain in the name

    def _raw(self, cfg):
        return compute_foot_placement(
            cfg, self.swing_cfg
        )

# ---- [ADDED] begin: plan-aware policy ----
class ForceAligned(FootPlacement):
    """
    Put the foot so that the leg (foot -> hip line) is parallel to the ground
    force the MPC plans for the coming stance. A force along the leg makes no
    moment about the hip, so hip/thigh torques (and energy) drop; this is the
    leg-axis force alignment seen in running animals.

    For leg i, with the plan rows k of its NEXT stance (MPCPlan.next_stance_rows):
        h_k = predicted hip point  (X_pred[k] position + Rz(yaw_k) hip_b)
        n_k = f_k / |f_k|          (planned force direction)
        P_k = I - n_k n_k^T        (removes the part of a vector along n_k)
    ||P_k (h_k - p)|| = distance from the foot p to the line through h_k along n_k.
    Weighted least squares over the foot's x, y (z fixed on the ground),
    regularized towards a base policy's foothold p_nom:

        min_p   sum_k w_k ||P_k (h_k - p)||^2  +  w_reg ||p - p_nom||^2,
        w_k = fz_k / sum fz

    -> 2x2 linear system  (sum w_k S'P_kS + w_reg I) u = sum w_k S'P_k (h_k - z0 e_z) + w_reg p_nom_xy,
       S = [e_x e_y], p = (u, z0).

    Needs the plan, so it is one MPC step behind unless
    ControllerCfg.placement_iters > 0 (solve -> place -> solve ...).
    Falls back to the base policy when no plan / no stance in the horizon.

    k, base     : base policy ("raibert" | "lipm" | "openloop") and its gain
    w_reg       : weight of the base foothold (0 = pure alignment)
    min_fz      : ignore plan rows with fz below this [N]
    """
    uses_plan = True

    def __init__(self, k: float = 0.03, base: str = "raibert", w_reg: float = 0.1,
                 min_fz: float = 5.0, max_offset: float = 0.2):
        super().__init__(max_offset)
        self.base = POLICIES[base](**({} if base == "openloop" else {"k": k}))
        self.w_reg, self.min_fz = w_reg, min_fz
        self.name = f"force_aligned({self.base.name},w_reg={w_reg:g})"
        self.last_rows = 0          # plan rows used in the last call (for checks)

    def _raw(self, cfg):
        p_nom = np.array(self.base._raw(cfg), dtype=float)
        z0 = p_terrain(p_nom)[2]
        plan = cfg.plan
        if plan is None or cfg.hip_b is None:
            return p_nom
        rows = plan.next_stance_rows(cfg.leg)
        f = plan.forces[rows, cfg.leg]
        rows, f = rows[f[:, 2] > self.min_fz], f[f[:, 2] > self.min_fz]
        self.last_rows = len(rows)
        if len(rows) == 0:
            return p_nom
        w = f[:, 2] / f[:, 2].sum()
        A = self.w_reg * np.eye(2)
        rhs = self.w_reg * p_nom[:2]
        for wk, fk, k in zip(w, f, rows):
            x = plan.X_pred[k]
            h = x[:3] + rot_z(x[5]) @ cfg.hip_b
            n = fk / np.linalg.norm(fk)
            P = np.eye(3) - np.outer(n, n)
            A += wk * P[:2, :2]
            rhs += wk * (P[:2] @ (h - np.array([0.0, 0.0, z0])))
        p = p_nom.copy()
        p[:2] = np.linalg.solve(A, rhs)
        p[2] = z0
        return p

POLICIES = {
    "openloop": lambda **kw: openloop(**kw),
    "raibert": lambda **kw: Raibert(**kw),
    "lipm": lambda **kw: lipm(**kw),
    "force_aligned": lambda **kw: ForceAligned(**kw),
}

