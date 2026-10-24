import numpy as np
from swing_traj import compute_foot_placement, p_terrain
from structs import FootPlacementCfg
from structs import SwingTrajectoryConfig

class FootPlacement:
    name = "base"

    def __init__(self, max_stride: float = 0.2):
        self.max_stride = max_stride

    def __call(self, cfg: FootPlacementCfg) -> np.ndarray:
        p = np.array(self._raw(cfg))
        hip_td = _hip_at_touchdown(cfg)
        d = p[:2] - hip_td[:2]
        stide_plan = np.linalg.norm(d)
        if stide_plan > self.offset:
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
        self.name = "Raibert"

    def _raw(self, cfg):
        p = _hip_at_touchdown(cfg)
        p[:2] += cfg.v_base[:2] * cfg.t_stance / 2.
        p[:2] += self.k * (cfg.v_base[:2] - cfg.v_cmd)
        return p

class lipm(FootPlacement):
    def __init__(self, k: float = 0.03, max_offset: float = 0.2):
        super().__init__(max_offset)
        self.swing_cfg = SwingTrajectoryConfig(k_fb=k)
        self.name = "lipm"

    def _raw(self, cfg):
        return compute_foot_placement(
            cfg, self.swing_cfg
        )