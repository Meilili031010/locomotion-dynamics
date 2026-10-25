import numpy as np
from swing_traj import compute_foot_placement, p_terrain
from structs import FootPlacementCfg
from structs import SwingTrajectoryCfg   # [FIXED] SwingTrajectoryConfig does not exist (ImportError)

class FootPlacement:
    name = "base"

    def __init__(self, max_stride: float = 0.2):
        self.max_stride = max_stride

    # [FIXED] was `__call`: Python name-mangles it to _FootPlacement__call, so
    # policy(cfg) raised "object is not callable".
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
        p[:2] += self.k * (cfg.v_base[:2] - cfg.v_cmd[:2])   # [FIXED] v_cmd is (3,) [vx, vy, wz]: (2,) - (3,) broadcast error
        return p

# [REVIEW NOTE] This is the Cheetah 3 heuristic (symmetry + velocity feedback +
# yaw + centrifugal terms), not a LIPM / capture-point law. With k=None the
# feedback gain becomes sqrt(h/g), which is the LIPM capture-point gain.
class lipm(FootPlacement):
    def __init__(self, k: float = 0.03, max_offset: float = 0.2):
        super().__init__(max_offset)
        self.swing_cfg = SwingTrajectoryCfg(k_fb=k)   # [FIXED] class name
        self.name = f"lipm_k{k}"                     # [CHANGED] gain in the name

    def _raw(self, cfg):
        return compute_foot_placement(
            cfg, self.swing_cfg
        )

# ---- [ADDED] begin ----
# Name -> constructor, so run/analysis scripts (and worker processes) can build
# a policy from plain data: POLICIES["raibert"](k=0.1)
POLICIES = {
    "openloop": lambda **kw: openloop(**kw),
    "raibert": lambda **kw: Raibert(**kw),
    "lipm": lambda **kw: lipm(**kw),
}
# ---- [ADDED] end ----
