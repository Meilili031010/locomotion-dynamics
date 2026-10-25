"""
[NEW FILE] Sample-based region of attraction (ROA) of the trot limit cycle.

Perturbations are applied on the Poincare section (poincare_aug.SectionSpec),
at the limit-cycle state, in any subset of
    vx vy vz wx wy wz   heading-frame base velocity push [m/s, rad/s]
    dphi                gait clock shift [cycles]
Each sample is simulated for --cycles crossings and classified:
    converged       final section state within --tol of the fixed point
                    (each coordinate scaled by TOL_SCALE)
    not converged   still standing but not back on the orbit
    fell
Samplers (--method):
    grid    n x n grid over two coordinates           -> ROA slice
    random  Sobol samples in the box [-bound, bound]  -> success fraction, pairwise views
    radial  random directions (scaled by the bounds), bisection on the push size
            -> ROA radius per direction; min = largest "ball" inside the ROA estimate

    ../.venv/bin/python roa.py --vx 0.5 --method grid --dims vx vy --bounds 0.8 0.8 --n 15
    ../.venv/bin/python roa.py --vx 0.5 --method grid --dims vy dphi --bounds 0.8 0.25
    ../.venv/bin/python roa.py --vx 0.5 --method random --dims vx vy wz dphi --bounds 0.8 0.8 2 0.2 --n 256
    ../.venv/bin/python roa.py --vx 0.5 --method radial --dims vx vy wz --bounds 1.5 1.5 4 --n 48
    ../.venv/bin/python roa.py --vx 0.5 --policy openloop --method grid --dims vx vy --bounds 0.8 0.8
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiment import Experiment, add_cli, from_cli
from poincare_aug import SectionSpec, PERTURB_NAMES, find_limit_cycle, simulate_sections

# |s - s*| / TOL_SCALE < tol for every coordinate -> converged
TOL_SCALE = {"z": 0.01, "roll": 0.02, "pitch": 0.02, "vx": 0.05, "vy": 0.05, "vz": 0.05,
             "wx": 0.2, "wy": 0.2, "wz": 0.1, "dphi": 0.01,
             "FL_x": 0.02, "FL_y": 0.02, "RR_x": 0.02, "RR_y": 0.02}
OUTCOMES = ["converged", "not converged", "fell"]


def perturbation_vector(dims: list[str], values: np.ndarray) -> np.ndarray:
    p = np.zeros(7)
    for d, v in zip(dims, values):
        p[PERTURB_NAMES.index(d)] = v
    return p


def evaluate(args) -> dict:
    """One sample: push at the section, simulate, classify."""
    exp, spec, snap, s_star, push, n_cycles, tol = args
    S, t, fallen, _ = simulate_sections(exp, spec, n_cycles + 1, snap, push[:6], push[6])
    if fallen or len(S) < n_cycles + 1:
        return dict(outcome=2, err=np.inf, cycles=-1)
    scale = np.array([TOL_SCALE[n] for n in spec.names()])
    err = np.abs(S - s_star) / scale            # (cycles+1, d)
    e_max = err.max(axis=1)
    ok = e_max < tol
    # first crossing after which it stays inside the tolerance
    k_conv = -1
    if ok[-1]:
        bad = np.where(~ok)[0]
        k_conv = int(bad[-1] + 1) if len(bad) else 0
    return dict(outcome=0 if ok[-1] else 1, err=float(e_max[-1]), cycles=k_conv)


def _radial(args) -> float:
    """Bisection on the push size along one direction -> largest converging radius."""
    exp, spec, snap, s_star, direction, r_max, n_bisect, n_cycles, tol = args
    lo, hi = 0.0, r_max
    if evaluate((exp, spec, snap, s_star, direction * hi, n_cycles, tol))["outcome"] == 0:
        return hi                                  # whole range converges
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if evaluate((exp, spec, snap, s_star, direction * mid, n_cycles, tol))["outcome"] == 0:
            lo = mid
        else:
            hi = mid
    return lo


def run_roa(exp: Experiment, spec: SectionSpec, method: str, dims: list[str], bounds: np.ndarray,
            n: int, n_cycles: int = 10, tol: float = 1.0, n_bisect: int = 6, seed: int = 0,
            workers=None) -> dict:
    lc = find_limit_cycle(exp, spec)
    snap, s_star = lc["snap"], lc["s_star"]
    bounds = np.asarray(bounds, dtype=float)
    out = dict(method=method, dims=dims, bounds=bounds, s_star=s_star, names=spec.names())

    if method in ("grid", "random"):
        if method == "grid":
            if len(dims) != 2:
                raise ValueError("grid needs exactly 2 dims")
            g = [np.linspace(-b, b, n) for b in bounds]
            P = np.array([[a, b] for b in g[1] for a in g[0]])
        else:
            from scipy.stats import qmc
            P = (qmc.Sobol(len(dims), seed=seed).random(n) * 2 - 1) * bounds
        jobs = [(exp, spec, snap, s_star, perturbation_vector(dims, p), n_cycles, tol) for p in P]
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(evaluate, jobs, chunksize=2))
        out.update(points=P, outcome=np.array([r["outcome"] for r in res]),
                   err=np.array([r["err"] for r in res]),
                   cycles=np.array([r["cycles"] for r in res]))
    elif method == "radial":
        rng = np.random.default_rng(seed)
        D = rng.normal(size=(n, len(dims)))
        D /= np.linalg.norm(D, axis=1, keepdims=True)          # unit directions in scaled space
        jobs = [(exp, spec, snap, s_star, perturbation_vector(dims, d * bounds), 1.0,
                 n_bisect, n_cycles, tol) for d in D]
        with ProcessPoolExecutor(workers) as ex:
            r = np.array(list(ex.map(_radial, jobs)))
        out.update(directions=D, radius=r, boundary=D * r[:, None] * bounds)
    else:
        raise ValueError(method)
    return out


def summarize(res: dict) -> str:
    dims = res["dims"]
    if res["method"] == "radial":
        r = res["radius"]
        k = int(np.argmin(r))
        worst = dict(zip(dims, np.round(res["boundary"][k], 3)))
        return (f"radial ROA estimate over {dims} (radius in units of the bounds, max 1):\n"
                f"  min {r.min():.3f}  median {np.median(r):.3f}  max {r.max():.3f}\n"
                f"  weakest direction (push at the boundary): {worst}")
    o = res["outcome"]
    frac = [np.mean(o == i) for i in range(3)]
    c = res["cycles"][o == 0]
    return (f"{res['method']} over {dims}, {len(o)} samples: "
            + ", ".join(f"{name} {100 * f:.1f}%" for name, f in zip(OUTCOMES, frac))
            + (f"\n  cycles to converge: median {np.median(c):.0f}, max {c.max()}" if len(c) else ""))


def plot_roa(res: dict, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dd"
    colors = ["#2a78d6", "#eda100", "#eb6834"]     # categorical slots: blue, yellow, orange
    markers = ["o", "s", "X"]
    dims = res["dims"]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    if res["method"] == "radial":
        if len(dims) == 2:
            B = res["boundary"]
            order = np.argsort(np.arctan2(B[:, 1] / res["bounds"][1], B[:, 0] / res["bounds"][0]))
            B = B[order]
            ax.fill(B[:, 0], B[:, 1], color=colors[0], alpha=0.15, lw=0)
            ax.plot(np.r_[B[:, 0], B[:1, 0]], np.r_[B[:, 1], B[:1, 1]], color=colors[0], lw=2,
                    marker="o", ms=4, label="ROA boundary (bisection)")
            ax.set_xlabel(dims[0], color=ink); ax.set_ylabel(dims[1], color=ink)
        else:
            r = np.sort(res["radius"])
            ax.plot(np.arange(len(r)), r, color=colors[0], lw=2, marker="o", ms=4)
            ax.set_xlabel("direction (sorted)", color=ink)
            ax.set_ylabel("ROA radius (fraction of bounds)", color=ink)
    else:
        P = res["points"]
        for i, name in enumerate(OUTCOMES):
            m = res["outcome"] == i
            if m.any():
                ax.scatter(P[m, 0], P[m, 1], s=34, color=colors[i], marker=markers[i],
                           edgecolor="white", linewidth=0.8, label=f"{name} ({m.sum()})")
        ax.set_xlabel(dims[0], color=ink); ax.set_ylabel(dims[1], color=ink)
        if len(dims) > 2:
            ax.set_title(f"projection on ({dims[0]}, {dims[1]})", color=muted, fontsize=9)
    ax.scatter([0], [0], marker="+", s=120, color=ink, zorder=5, label="limit cycle")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(grid)
    ax.tick_params(colors=muted, labelsize=8)
    fig.suptitle(title, color=ink, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli(ap)
    r = ap.add_argument_group("region of attraction")
    r.add_argument("--method", choices=["grid", "random", "radial"], default="grid")
    r.add_argument("--dims", nargs="+", default=["vx", "vy"], choices=PERTURB_NAMES)
    r.add_argument("--bounds", type=float, nargs="+", default=None,
                   help="max push per dim (default 0.8 m/s, 2 rad/s, 0.25 cycles)")
    r.add_argument("--n", type=int, default=15, help="grid: points per axis; random: samples; radial: directions")
    r.add_argument("--cycles", type=int, default=10)
    r.add_argument("--tol", type=float, default=1.0)
    r.add_argument("--bisect", type=int, default=6)
    r.add_argument("--section", choices=["touchdown", "clock"], default="touchdown")
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=None, help="output prefix (default results/roa_<method>_<dims>)")
    a = ap.parse_args()

    default_b = {"vx": 0.8, "vy": 0.8, "vz": 0.5, "wx": 2.0, "wy": 2.0, "wz": 2.0, "dphi": 0.25}
    bounds = a.bounds or [default_b[d] for d in a.dims]
    if len(bounds) != len(a.dims):
        ap.error("--bounds needs one value per --dims entry")
    exp = from_cli(a)
    spec = SectionSpec(section=a.section)
    res = run_roa(exp, spec, a.method, a.dims, bounds, a.n, a.cycles, a.tol, a.bisect,
                  a.seed, a.workers)
    print(f"{exp.label}  v_cmd={exp.v_cmd}  section={spec.section}")
    print(summarize(res))
    out = a.out or f"results/roa_{a.method}_{'_'.join(a.dims)}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out + ".npz", **{k: np.asarray(v) for k, v in res.items()},
             experiment=str(exp.to_dict()), section=str(spec))
    plot_roa(res, out + ".png", f"ROA  {exp.label}  v_cmd={exp.v_cmd}  ({a.cycles} cycles)")
    print("saved", out + ".npz", out + ".png")


if __name__ == "__main__":
    main()
