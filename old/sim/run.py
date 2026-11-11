"""
[NEW FILE] Run one trot simulation.

    cd loco_proj/sim
    ../.venv/bin/python run.py --vx 0.5                          # headless summary
    ../.venv/bin/python run.py --vx 0.5 --viewer
    ../.venv/bin/python run.py --vx 0.5 --stiffness 2e4 --damping 300 --friction 0.8
    ../.venv/bin/python run.py --vx 0.5 --solref 0.01 1 --dt 0.0005 --integrator implicitfast
    ../.venv/bin/python run.py --vx 0.5 --set mpc.h=30 mpc.period_constraint=hard
    ../.venv/bin/python run.py --vx 0.5 --push 3.0 0.1 0 60 0     # 60 N sideways for 0.1 s at t=3
    ../.venv/bin/python run.py --vx 0.5 --record-ref results/ref_v05.npz   # save the limit cycle
    ../.venv/bin/python run.py --vx 0.5 --periodic-ref results/ref_v05.npz --set mpc.h=30 mpc.period_constraint=soft
    ../.venv/bin/python run.py --config my_exp.json --save results/log.npz --plot results/run.png

All configuration flags: see experiment.py (--set works for every config field).
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: BLAS threads only add overhead
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import Counter

import numpy as np

from experiment import add_cli, from_cli
from ref_gen import PeriodicReference
from sim_env import run, describe, ForcePush


def heading_velocities(L: dict) -> np.ndarray:
    """(N, 3) [vx, vy, wz] in the heading frame, from the logged centroidal MPC state."""
    X = L["x_mpc"]
    yaw = X[:, 5]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * X[:, 6] + s * X[:, 7], -s * X[:, 6] + c * X[:, 7], X[:, 11]], -1)


def summary(L: dict, fallen: bool, tail_s: float = 2.0) -> str:
    tail = L["t"] > L["t"][-1] - tail_s
    vh = heading_velocities(L)[tail].mean(0)
    cmd = L["v_cmd"][tail].mean(0)
    st = Counter(map(str, L["mpc_status"][tail]))
    return (f"fallen={fallen}  t_end={L['t'][-1]:.2f}s\n"
            f"last {tail_s:.0f}s  [vx, vy, wz] = {np.round(vh, 3)}  (cmd {np.round(cmd, 3)})\n"
            f"height = {L['x_mpc'][tail, 2].mean():.3f}   max|roll|,|pitch| = "
            f"{np.abs(L['x_mpc'][tail, 3]).max():.3f}, {np.abs(L['x_mpc'][tail, 4]).max():.3f}\n"
            f"MPC status: {dict(st)}   period residual = {L['period_residual'][tail].mean():.2e}")


def plot(L: dict, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dd"
    blue, orange = "#2a78d6", "#eb6834"
    t = L["t"]; vh = heading_velocities(L); X = L["x_mpc"]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    panels = [(axs[0, 0], "forward speed vx [m/s]", vh[:, 0], L["v_cmd"][:, 0]),
              (axs[0, 1], "lateral speed vy [m/s]", vh[:, 1], L["v_cmd"][:, 1]),
              (axs[1, 0], "yaw rate wz [rad/s]", vh[:, 2], L["v_cmd"][:, 2]),
              (axs[1, 1], "base height z [m]", X[:, 2], None)]
    for ax, name, y, c in panels:
        ax.plot(t, y, color=blue, lw=1.5, label="measured")
        if c is not None:
            ax.plot(t, c, color=orange, lw=1.5, ls="--", label="command")
            ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.set_title(name, color=ink, fontsize=10, loc="left")
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    for ax in axs[1]:
        ax.set_xlabel("time [s]", color=ink)
    fig.suptitle(title, color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli(ap)
    r = ap.add_argument_group("run")
    r.add_argument("--time", type=float, default=8.0)
    r.add_argument("--viewer", action="store_true")
    r.add_argument("--push", type=float, nargs=5, action="append", default=[],
                   metavar=("T0", "DUR", "FX", "FY", "FZ"), help="external base force (world), repeatable")
    r.add_argument("--save", default=None, help="log .npz")
    r.add_argument("--plot", default=None, help="summary figure .png")
    r.add_argument("--record-ref", default=None, help="save a PeriodicReference from the last cycles")
    r.add_argument("--ref-cycles", type=int, default=4)
    r.add_argument("--dump-config", default=None, help="write the full experiment as JSON")
    a = ap.parse_args()

    exp = from_cli(a)
    model, data, ctrl = exp.build()
    print(describe(model, exp.sim))
    print(f"policy={ctrl.foot_placement.name}  v_cmd={exp.v_cmd}  MPC: h={ctrl.cfg.mpc_cfg.h} "
          f"mu={ctrl.cfg.mpc_cfg.mu:.3f} period_constraint={ctrl.cfg.mpc_cfg.period_constraint}"
          f"({ctrl.cfg.mpc_cfg.period_pairs}) periodic_ref={ctrl.cfg.mpc_cfg.periodic_ref}")
    if a.dump_config:
        exp.save_json(a.dump_config)

    pushes = [ForcePush(p[0], p[1], tuple(p[2:5])) for p in a.push]
    log, fallen = run(model, data, ctrl, a.time, exp.command(), pushes=pushes,
                      viewer=a.viewer, log_every=2 if a.record_ref else 10)
    L = log.arrays()
    print(summary(L, fallen))

    for path in (a.save, a.plot, a.record_ref):
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if a.save:
        log.save(a.save, experiment=str(exp.to_dict()))
        print("log saved:", a.save)
    if a.plot:
        plot(L, a.plot, f"{exp.label}  v_cmd={exp.v_cmd}")
        print("plot saved:", a.plot)
    if a.record_ref:
        if fallen:
            print("not recording a reference: the robot fell")
        else:
            pref = PeriodicReference.from_cycles(L["phase"], L["x_mpc"], ctrl.cfg.gait_cfg.period,
                                                 n_cycles=a.ref_cycles, v_cmd=exp.v_cmd)
            pref.save(a.record_ref)
            print(f"periodic reference saved: {a.record_ref}  (cycle mean [vx,vy,wz] = "
                  f"{np.round(pref.mean_cmd(), 3)})")


if __name__ == "__main__":
    main()
