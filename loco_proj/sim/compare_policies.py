"""
[NEW FILE] Compare foot placement policies with the Poincare return map.

    cd loco_proj/sim
    ../.venv/bin/python compare_policies.py                     # vx = 0.5 m/s
    ../.venv/bin/python compare_policies.py --vcmd 0.6 0 0.5    # trot while turning

For each policy: fixed point s*, return-map eigenvalues (least squares),
and recovery from one fixed lateral push. Writes
    results/poincare_<vcmd>.npz   (all numbers)
    results/poincare_<vcmd>.png   (eigenvalues | push recovery)
Add a policy by adding an ExperimentSpec to POLICY_SET.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

import numpy as np

from poincare import ExperimentSpec, VEL_IDX, analyze, simulate_sections, find_limit_cycle

# (spec kwargs) - the label is generated from policy + policy_kw
POLICY_SET = [
    dict(policy="nominal"),                       # no velocity feedback in placement
    dict(policy="raibert", policy_kw={"k": 0.03}),  # MIT software gain
    dict(policy="raibert", policy_kw={"k": 0.17}),  # ~ sqrt(h/g): capture-point gain
    dict(policy="cheetah3", policy_kw={"k_fb": 0.03}),  # + centrifugal + yaw terms
]
PUSH = np.array([0.0, 0.15, 0.0, 0.0, 0.0, 0.0])   # lateral push [m/s] at the section
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]  # categorical slots 1-3, 5 (dataviz palette)


def push_recovery(spec, n=10):
    s_star, snap, _, _ = find_limit_cycle(spec)
    S, fallen, _ = simulate_sections(spec, n, snap, PUSH)
    err = np.linalg.norm((S - s_star)[:, VEL_IDX], axis=1)
    return err, fallen


def plot(results, recov, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dd"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(th), np.sin(th), color=muted, lw=1, ls="--", zorder=1)
    ax1.text(0.71, 0.75, "|λ| = 1", color=muted, fontsize=9)
    for r, c in zip(results, COLORS):
        ev = r.eigvals
        ax1.scatter(ev.real, ev.imag, s=46, color=c, edgecolor="white", linewidth=1.5,
                    label=f"{r.spec.label}  ρ={r.spectral_radius:.2f}", zorder=3)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.15, 1.15); ax1.set_ylim(-1.15, 1.15)
    ax1.set_xlabel("Re λ", color=ink); ax1.set_ylabel("Im λ", color=ink)
    ax1.set_title("Return-map eigenvalues", color=ink, fontsize=11, loc="left")
    ax1.legend(fontsize=8, frameon=False, loc="lower left")

    for (lab, err, fallen), c in zip(recov, COLORS):
        k = np.arange(len(err))
        ax2.semilogy(k, err, color=c, lw=2, marker="o", ms=4, label=lab + (" (fell)" if fallen else ""))
    ax2.set_xlabel("gait cycle after push", color=ink)
    ax2.set_ylabel("‖velocity coords − s*‖  (log)", color=ink)
    ax2.set_title(f"Recovery from lateral push Δvy = {PUSH[1]} m/s", color=ink, fontsize=11, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    fig.suptitle(title, color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    ap.add_argument("--extended", action="store_true", help="add foothold coordinates to the section")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tag = "_".join(f"{v:g}" for v in a.vcmd)
    results, recov = [], []
    for kw in POLICY_SET:
        if kw["policy"] == "cheetah3" and a.vcmd[2] == 0.0:
            print("note: cheetah3 == raibert(k=0.03) when wz_cmd = 0 (its extra terms are yaw terms); skipped")
            continue
        spec = ExperimentSpec(v_cmd=tuple(a.vcmd), extended=a.extended, **kw)
        try:
            r = analyze(spec, n_trials=a.trials)
        except RuntimeError as e:
            print("SKIP:", e)
            continue
        print(r.summary(), flush=True)
        err, fallen = push_recovery(spec)
        results.append(r)
        recov.append((spec.label, err, fallen))

    print("\n policy                      rho    s*: vx     vy     wz")
    for r in results:
        s = r.s_star
        print(f" {r.spec.label:26s} {r.spectral_radius:.3f}  {s[3]:+.3f} {s[4]:+.3f} {s[8]:+.3f}")

    np.savez(os.path.join(a.out, f"poincare_{tag}.npz"),
             labels=[r.spec.label for r in results], names=results[0].spec.names,
             s_star=[r.s_star for r in results], A=[r.A for r in results],
             eigvals=[r.eigvals for r in results], J_fd=[r.J_fd for r in results],
             recovery=[e for _, e, _ in recov])
    png = os.path.join(a.out, f"poincare_{tag}.png")
    plot(results, recov, png, f"Go2 trot, v_cmd = {a.vcmd}  (section: FL/RR touchdown)")
    print("saved", png)


if __name__ == "__main__":
    main()
