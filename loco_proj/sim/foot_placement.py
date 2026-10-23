"""
[NEW FILE] Foot placement policies.

Every policy maps the same PlacementInput to a world-frame foothold (3,).
The controller only calls `policy(inp)`, so a new approach for the Poincare
study is a new class here; nothing else changes.

Policies:
    NominalHip  : hip at touchdown + feed-forward (T_st/2) v_cmd. No velocity
                  feedback -> open-loop baseline.
    Raibert     : hip at touchdown + (T_st/2) v + k (v - v_cmd).
    Cheetah3    : Raibert + centrifugal + yaw-rate terms (Di Carlo 2018),
                  built on swing_traj.compute_foot_placement.
"""
from dataclasses import dataclass, field

import numpy as np

from swing_traj import SwingTrajectoryConfig, compute_foot_placement, p_terrain


@dataclass
class PlacementInput:
    leg: int
    p_hip: np.ndarray        # (3,) shoulder point now, world
    r_hip: np.ndarray        # (3,) shoulder offset from base, world (yaw-rotated)
    v_base: np.ndarray       # (3,) base linear velocity, world
    w_base: np.ndarray       # (3,) base angular velocity, world
    v_cmd: np.ndarray        # (3,) [vx, vy, wz] command, world frame
    height: float            # base height
    yaw: float
    t_stance: float
    t_swing: float
    t_swing_remaining: float


class FootPlacementPolicy:
    """Base class: subclasses implement `_raw(inp)`; reach clipping is shared."""
    name = "base"

    def __init__(self, max_offset: float = 0.2):
        # max horizontal distance between foothold and hip-at-touchdown [m]
        self.max_offset = max_offset

    def __call__(self, inp: PlacementInput) -> np.ndarray:
        p = np.array(self._raw(inp), dtype=float)
        hip_td = _hip_at_touchdown(inp)
        d = p[:2] - hip_td[:2]
        n = np.linalg.norm(d)
        if n > self.max_offset:
            p[:2] = hip_td[:2] + d * (self.max_offset / n)
        p[2] = p_terrain(p)[2]
        return p

    def _raw(self, inp: PlacementInput) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


def _hip_at_touchdown(inp: PlacementInput) -> np.ndarray:
    return inp.p_hip + np.array([inp.v_cmd[0], inp.v_cmd[1], 0.0]) * inp.t_swing_remaining


class NominalHip(FootPlacementPolicy):
    """Open-loop: no dependence on the measured velocity."""
    name = "nominal"

    def _raw(self, inp):
        p = _hip_at_touchdown(inp)
        p[:2] += inp.v_cmd[:2] * inp.t_stance / 2.0
        return p


class Raibert(FootPlacementPolicy):
    """Raibert heuristic with velocity feedback gain k [s]."""

    def __init__(self, k: float = 0.03, max_offset: float = 0.2):
        super().__init__(max_offset)
        self.k = k
        self.name = f"raibert_k{k:g}"

    def _raw(self, inp):
        p = _hip_at_touchdown(inp)
        p[:2] += inp.v_base[:2] * inp.t_stance / 2.0
        p[:2] += self.k * (inp.v_base[:2] - inp.v_cmd[:2])
        return p


class Cheetah3(FootPlacementPolicy):
    """
    Cheetah 3 heuristic. k_fb=None -> sqrt(h/g) (capture-point gain, the
    swing_traj.py default); the MIT code uses k = 0.03.
    """

    def __init__(self, k_fb: float = 0.03, use_centrifugal: bool = True,
                 max_offset: float = 0.2):
        super().__init__(max_offset)
        self.cfg = SwingTrajectoryConfig(k_fb=k_fb)
        self.use_centrifugal = use_centrifugal
        self.name = f"cheetah3_k{'sqrt(h/g)' if k_fb is None else f'{k_fb:g}'}"

    def _raw(self, inp):
        return compute_foot_placement(
            p_hip=inp.p_hip, v_base=inp.v_base, v_cmd=inp.v_cmd, height=inp.height,
            r_hip=inp.r_hip, t_stance=inp.t_stance, heading=inp.yaw, cfg=self.cfg,
            t_swing_remaining=inp.t_swing_remaining, use_centrifugal=self.use_centrifugal)


POLICIES = {
    "nominal": lambda **kw: NominalHip(**kw),
    "raibert": lambda **kw: Raibert(**kw),
    "cheetah3": lambda **kw: Cheetah3(**kw),
}
