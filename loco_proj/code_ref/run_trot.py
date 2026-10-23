"""
[NEW FILE] Trot demo.

    cd loco_proj/sim
    ../.venv/bin/python run_trot.py                    # headless, prints summary
    ../.venv/bin/python run_trot.py --viewer           # MuJoCo viewer
    ../.venv/bin/python run_trot.py --vx 0.8 --wz 0.5 --policy raibert
"""
import argparse
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: BLAS threads only add overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from controller import ControllerConfig, LocomotionController
from foot_placement import Cheetah3, NominalHip, Raibert
from sim_env import make_model, run


def make_policy(name: str, k: float | None):
    if name == "nominal":
        return NominalHip()
    if name == "raibert":
        return Raibert(k=0.03 if k is None else k)
    return Cheetah3(k_fb=0.03 if k is None else k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--time", type=float, default=8.0)
    ap.add_argument("--policy", default="cheetah3", choices=["cheetah3", "raibert", "nominal"])
    ap.add_argument("--k", type=float, default=None, help="foot placement feedback gain")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--save", default="results/trot_log.npz")
    a = ap.parse_args()

    model, data = make_model()
    cfg = ControllerConfig()
    ctrl = LocomotionController(model, data, cfg, make_policy(a.policy, a.k))
    target = np.array([a.vx, a.vy, a.wz])

    def v_cmd(t):
        # stand still, then ramp the command up over 1 s
        s = np.clip((t - cfg.t_stand) / 1.0, 0.0, 1.0)
        return s * target

    log, fallen = run(model, data, ctrl, a.time, v_cmd, viewer=a.viewer)
    L = log.arrays()
    tail = L["t"] > L["t"][-1] - 2.0
    print(f"policy={ctrl.placement.name}  fallen={fallen}  t_end={L['t'][-1]:.2f}s")
    print(f"mean vel (last 2 s) world = {L['qvel'][tail, :3].mean(0).round(3)}")
    print(f"mean height (last 2 s)    = {L['qpos'][tail, 2].mean():.3f}")
    print(f"final base position       = {L['qpos'][-1, :3].round(3)}")
    if a.save:
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        log.save(a.save)
        print("log saved to", a.save)


if __name__ == "__main__":
    main()
