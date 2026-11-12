"""
[NEW FILE] Hook for gait optimization (method still to be chosen, e.g. from
Ross Hatton's geometric gait optimization work).

What is here:
    GAIT_PARAMS            parameter vector x <-> Experiment fields
    apply_params(exp, x)   write x into a copy of an Experiment
    objective(x, exp)      simulate, return (cost, metrics)  -- the black box
    feasible(x, exp)       checks that the gait fits the MPC grid

What is NOT here: the optimizer. Plug yours in where marked below. A Hatton-
style method would replace objective() by a model-based cost (displacement
and cost metric over a closed curve in shape space), and use the simulator
only to check the result.

Run (prints the cost of a few hand-picked gaits):
    python Gait_opt.py
"""
import copy
from dataclasses import dataclass

import numpy as np

from Cli import Experiment
from Metrics import evaluate


@dataclass
class Param:
    name: str
    lo: float
    hi: float
    get: callable
    set: callable


def _set_offset(i):
    def f(e, v):
        o = list(e.ctrl.gait_cfg.offsets)
        o[i] = v
        e.ctrl.gait_cfg.offsets = tuple(o)
    return f


# Parameter vector. FL lift-off phase is fixed at 0 (only relative phases matter).
GAIT_PARAMS = [
    Param("period", 0.3, 0.8, lambda e: e.ctrl.gait_cfg.period,
          lambda e, v: setattr(e.ctrl.gait_cfg, "period", v)),
    Param("duty", 0.4, 0.8, lambda e: e.ctrl.gait_cfg.duty,
          lambda e, v: setattr(e.ctrl.gait_cfg, "duty", v)),
    Param("offset_FR", 0.0, 1.0, lambda e: e.ctrl.gait_cfg.offsets[1], _set_offset(1)),
    Param("offset_RL", 0.0, 1.0, lambda e: e.ctrl.gait_cfg.offsets[2], _set_offset(2)),
    Param("offset_RR", 0.0, 1.0, lambda e: e.ctrl.gait_cfg.offsets[3], _set_offset(3)),
    Param("step_height", 0.04, 0.12, lambda e: e.ctrl.swing_cfg.step_height,
          lambda e, v: setattr(e.ctrl.swing_cfg, "step_height", v)),
]
NAMES = [p.name for p in GAIT_PARAMS]
TROT = np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.08])     # = default table (FL, RR together)


def apply_params(exp: Experiment, x) -> Experiment:
    """Copy of exp with the gait parameters x. The gait switches to the duty/offsets form."""
    e = copy.deepcopy(exp)
    g = e.ctrl.gait_cfg
    if g.offsets is None:
        g.duty, g.offsets = 0.5, (0.0, 0.0, 0.0, 0.0)
    for p, v in zip(GAIT_PARAMS, x):
        p.set(e, float(v))
    g.__post_init__()
    # keep the MPC horizon about one gait cycle
    e.ctrl.mpc_cfg.h = max(10, int(round(g.period / e.ctrl.mpc_cfg.dt)))
    return e


def feasible(x, exp: Experiment) -> tuple[bool, str]:
    """
    Contact switches must land on MPC steps: every lift-off / touchdown phase
    times the period must be a multiple of mpc dt.
    """
    e = apply_params(exp, x)
    g, dt = e.ctrl.gait_cfg, e.ctrl.mpc_cfg.dt
    for p, v in zip(GAIT_PARAMS, x):
        if not p.lo <= v <= p.hi:
            return False, f"{p.name}={v} outside [{p.lo}, {p.hi}]"
    ev = np.concatenate([g.offsets, np.mod(np.array(g.offsets) + 1 - g.duty, 1.0)]) * g.period
    if np.any(np.abs(ev / dt - np.round(ev / dt)) > 1e-6):
        return False, f"gait events {np.round(ev, 4)} s are not multiples of mpc dt {dt}"
    return True, ""


def objective(x, exp: Experiment, duration: float = 6.0,
              w_speed: float = 5.0, fall_cost: float = 100.0) -> tuple[float, dict]:
    """
    Black-box cost of the gait x:
        CoT_abs + w_speed * |speed - v_cmd| / v_cmd       (fell: fall_cost)
    Change the cost here (e.g. CoT_pos, tau^2, stability margins) to match
    the question you study.
    """
    ok, why = feasible(x, exp)
    if not ok:
        return fall_cost, {"infeasible": why}
    m = evaluate(apply_params(exp, x), duration)
    if m["fallen"] or m.get("T", 0.0) == 0.0:
        return fall_cost, m
    v_err = abs(m["speed"] - m["v_cmd"]) / max(m["v_cmd"], 1e-3)
    return m["CoT_abs"] + w_speed * v_err, m


# --------------------------------------------------------------------------
# YOUR OPTIMIZER GOES HERE, e.g.
#   x_best = my_optimizer(lambda x: objective(x, exp)[0], x0=TROT, bounds=...)
# objective() is expensive (one simulation, ~seconds) and noisy across
# gait families, so prefer few-evaluation methods (grid over 1-2 params,
# Bayesian optimization, CMA-ES with a small population) or a model-based
# method that only uses the simulator to verify.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    exp = Experiment()
    # x = [period, duty, offset_FR, offset_RL, offset_RR, step_height], FL lifts off at 0
    candidates = {
        "trot":  TROT,                                          # FL+RR, FR+RL
        "pace":  np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.08]),     # FL+RL, FR+RR
        "bound": np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.08]),     # FL+FR, RL+RR
        "walk":  np.array([0.8, 0.75, 0.5, 0.75, 0.25, 0.08]),  # one leg at a time
    }
    for name, x in candidates.items():
        cost, m = objective(x, exp)
        print(f"{name:12s} cost={cost:7.3f}  " +
              ("fell" if m.get("fallen") else
               m.get("infeasible") or f"speed={m['speed']:.3f} CoT_abs={m['CoT_abs']:.3f}"))
