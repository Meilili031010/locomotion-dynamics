"""
[NEW FILE] Poincare analysis on the time-augmented state.

Why augment: the controller is clock-driven (contact schedule = f(t)), so the
closed loop x' = f(x, t) is non-autonomous. With the gait phase
phi = (t - t_stand - clock_offset) / T  (mod 1, a point on the circle S^1)
added to the state, x_bar = (x, phi) is autonomous:
    x' = f(x, phi),   phi' = 1 / T.
A section is a surface in (x, phi) space; the return map P: Sigma -> Sigma.

Sections (--section):
  clock      Sigma = {phi = 0 mod 1}            stroboscopic map. phi is the
             same at every crossing, so delta_phi is constant (dropped from the
             fit) and the return time is exactly T.
  touchdown  Sigma = {FL foot makes ground contact after >= min_air airborne}
             state-based. The clock phase at the event, delta_phi (wrapped to
             [-0.5, 0.5)), is a genuine coordinate: late/early touchdown, and
             the return time tau_k = t_{k+1} - t_k varies.

Section coordinates (heading frame, so x, y, yaw drop out):
  s_bar = [z, roll, pitch, vx, vy, vz, wx, wy, wz, (FL/RR foot xy), delta_phi]
  velocities from the centroidal state (CoM velocity, angular momentum) or the
  base link (--coords base).

Perturbations at the section: base velocity pushes (6) and a clock phase shift
(QuadrupedController.shift_phase) -> 7 directly perturbable coordinates.

Outputs: fixed point s_bar*, return time tau*, DP from least squares over
pushed trajectories (eigenvalues, spectral radius), central-difference
columns for the perturbable coordinates, d tau / d s_bar.

    ../.venv/bin/python poincare_aug.py --vx 0.5
    ../.venv/bin/python poincare_aug.py --vx 0.5 --section clock
    ../.venv/bin/python poincare_aug.py --vx 0.5 --compare raibert:k=0.03 raibert:k=0.17 openloop
    ../.venv/bin/python poincare_aug.py --vx 0.5 --set mpc.h=30 mpc.period_constraint=hard
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import mujoco
import numpy as np

from experiment import Experiment, add_cli, from_cli, _parse_value
from sim_env import run, snapshot, restore, push_velocity
from structs import rot_z

BASE_NAMES = ["z", "roll", "pitch", "vx", "vy", "vz", "wx", "wy", "wz"]
FOOT_NAMES = ["FL_x", "FL_y", "RR_x", "RR_y"]
PERTURB_NAMES = ["vx", "vy", "vz", "wx", "wy", "wz", "dphi"]
DEFAULT_SCALE = np.array([0.05, 0.05, 0.03, 0.15, 0.15, 0.15, 0.03])   # random push size per coordinate


@dataclass
class SectionSpec:
    section: str = "touchdown"      # "touchdown" | "clock"
    leg: str = "FL"                 # touchdown leg
    min_air: float = 0.1            # [s] airborne before a contact counts as touchdown
    coords: str = "centroidal"      # "centroidal" | "base"
    feet: bool = False              # add FL/RR foot xy (relative to base, heading frame)
    n_warmup: int = 14              # cycles before the fixed point is measured
    n_avg: int = 6

    def names(self) -> list[str]:
        return BASE_NAMES + (FOOT_NAMES if self.feet else []) + ["dphi"]

    @property
    def dim(self) -> int:
        return len(self.names())

    def perturb_index(self) -> list[int]:
        """Positions of PERTURB_NAMES inside the section vector."""
        n = self.names()
        return [n.index(p) for p in PERTURB_NAMES]


# --------------------------------------------------------------------------
# Section coordinates + crossing detection
# --------------------------------------------------------------------------
def section_coords(model, data, ctrl, spec: SectionSpec) -> np.ndarray:
    mujoco.mj_forward(model, data)          # kinematics of the current state (no state change)
    st = ctrl.st
    if spec.coords == "centroidal":
        x = st.mpc_state_centroidal(ctrl.params["I_body"], ctrl.params["com_b"])
    else:
        x = st.mpc_state()
    Rz = rot_z(x[5])
    v_h, w_h = Rz.T @ x[6:9], Rz.T @ x[9:12]
    out = [x[2], x[3], x[4], *v_h, *w_h]
    if spec.feet:
        rel = (st.foot_positions[[0, 3]] - data.qpos[:3]) @ Rz
        out += list(rel[:, :2].reshape(-1))
    phi = ctrl.get_phase()
    out.append(phi - np.floor(phi + 0.5))   # wrapped to [-0.5, 0.5)
    return np.array(out)


class SectionRecorder:
    """run() callback: detects crossings, stores s_bar, time and (optionally) snapshots."""

    def __init__(self, model, spec: SectionSpec, keep_snapshots=False, stop_after=None):
        self.spec = spec
        self.keep = keep_snapshots
        self.stop_after = stop_after
        self.S, self.t, self.snaps = [], [], []
        self.foot = model.geom(spec.leg).id
        self.floor = model.geom("floor").id
        self.air_time = 0.0
        self.last_cycle = None

    def _foot_contact(self, data) -> bool:
        if data.ncon == 0:
            return False
        g = data.contact.geom[:data.ncon]
        return bool(np.any(((g[:, 0] == self.foot) & (g[:, 1] == self.floor)) |
                           ((g[:, 1] == self.foot) & (g[:, 0] == self.floor))))

    def prime(self, model, data, ctrl):
        """Call when starting from a snapshot taken at a crossing (no immediate re-trigger)."""
        self.air_time = 0.0
        self.last_cycle = ctrl.get_cycle_index()

    def crossed(self, model, data, ctrl) -> bool:
        if ctrl.get_cycle_index() < 0:           # still standing
            return False
        if self.spec.section == "clock":
            c = ctrl.get_cycle_index()
            hit = self.last_cycle is not None and c != self.last_cycle
            self.last_cycle = c
            return hit
        on = self._foot_contact(data)
        hit = on and self.air_time >= self.spec.min_air
        self.air_time = 0.0 if on else self.air_time + model.opt.timestep
        return hit

    def record(self, model, data, ctrl):
        self.S.append(section_coords(model, data, ctrl, self.spec))
        self.t.append(data.time)
        if self.keep:
            self.snaps.append(snapshot(model, data, ctrl))

    def __call__(self, model, data, ctrl):
        if self.crossed(model, data, ctrl):
            self.record(model, data, ctrl)
        return self.stop_after is not None and len(self.S) >= self.stop_after


# --------------------------------------------------------------------------
# Simulation (top-level functions so they run in worker processes)
# --------------------------------------------------------------------------
def simulate_sections(exp: Experiment, spec: SectionSpec, n_sections: int, snap=None,
                      dv=None, dphi: float = 0.0, keep_snapshots=False):
    """
    From rest (snap=None) or from a snapshot taken at a crossing, apply a push
    (dv: heading-frame base velocity change (6,), dphi: clock shift [cycles]),
    then record n_sections crossings. From a snapshot, S[0] is the (pushed)
    state at that crossing. Returns (S, t, fallen, recorder).
    """
    model, data, ctrl = exp.build()
    rec = SectionRecorder(model, spec, keep_snapshots, stop_after=n_sections)
    period = ctrl.cfg.gait_cfg.period
    if snap is None:
        duration = ctrl.cfg.t_stand + (n_sections + 2) * period
    else:
        restore(model, data, ctrl, snap)
        if dv is not None:
            push_velocity(model, data, dv)
        if dphi:
            ctrl.shift_phase(dphi)
        rec.prime(model, data, ctrl)
        rec.record(model, data, ctrl)
        duration = (n_sections + 1) * period
    _, fallen = run(model, data, ctrl, duration, exp.command(), log_every=None, callbacks=[rec])
    return np.array(rec.S), np.array(rec.t), fallen, rec


def find_limit_cycle(exp: Experiment, spec: SectionSpec):
    n = spec.n_warmup + spec.n_avg
    S, t, fallen, rec = simulate_sections(exp, spec, n, keep_snapshots=True)
    if fallen or len(S) < n:
        raise RuntimeError(f"{exp.label}: fell or no steady gait ({len(S)}/{n} crossings)")
    tail = S[-spec.n_avg:]
    s_star = tail.mean(axis=0)
    tau = np.diff(t)[-spec.n_avg + 1:]
    return dict(s_star=s_star, tau_star=float(tau.mean()), snap=rec.snaps[-1], history=S,
                t=t, spread=np.abs(tail - s_star).max(axis=0))


def _pushed_traj(args):
    exp, spec, snap, push, n = args
    S, t, fallen, _ = simulate_sections(exp, spec, n + 1, snap, push[:6], push[6])
    return None if fallen or len(S) < n + 1 else (S, t)


def jacobian_fd(exp, spec, snap, delta=None, workers=None):
    """Central differences on the 7 perturbable coordinates -> J (d, 7)."""
    delta = DEFAULT_SCALE * 0.6 if delta is None else np.asarray(delta)
    jobs = []
    for j in range(7):
        for sgn in (1, -1):
            p = np.zeros(7); p[j] = sgn * delta[j]
            jobs.append((exp, spec, snap, p, 1))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_pushed_traj, jobs))
    pidx = spec.perturb_index()
    J = np.full((spec.dim, 7), np.nan)
    for j in range(7):
        rp, rm = res[2 * j], res[2 * j + 1]
        if rp is None or rm is None:
            continue
        (Sp, _), (Sm, _) = rp, rm
        J[:, j] = (Sp[1] - Sm[1]) / (Sp[0, pidx[j]] - Sm[0, pidx[j]])
    return J


def jacobian_lstsq(exp, spec, snap, s_star, tau_star, n_trials=32, n_steps=5, scale=None,
                   seed=0, skip_first=True, workers=None):
    """
    Fit ds_{k+1} = A ds_k and dtau_k = c . ds_k over randomly pushed trajectories.
    skip_first: drop the pair starting at the push itself (legs inconsistent with
    the pushed base velocity there). Coordinates that never vary (e.g. dphi on
    the clock section) are left out of the fit: `active` marks the ones used.
    """
    rng = np.random.default_rng(seed)
    scale = DEFAULT_SCALE if scale is None else np.asarray(scale)
    jobs = [(exp, spec, snap, rng.uniform(-1, 1, 7) * scale, n_steps) for _ in range(n_trials)]
    with ProcessPoolExecutor(workers) as ex:
        trajs = [r for r in ex.map(_pushed_traj, jobs) if r is not None]
    k0 = 1 if skip_first else 0
    X = np.vstack([S[k0:-1] for S, _ in trajs]) - s_star
    Y = np.vstack([S[k0 + 1:] for S, _ in trajs]) - s_star
    tau = np.concatenate([np.diff(t)[k0:] for _, t in trajs]) - tau_star
    active = X.std(axis=0) > 1e-9
    Xa, Ya = X[:, active], Y[:, active]
    At, *_ = np.linalg.lstsq(Xa, Ya, rcond=None)
    A = At.T
    resid = float(np.linalg.norm(Ya - Xa @ At) / max(np.linalg.norm(Ya), 1e-12))
    c, *_ = np.linalg.lstsq(Xa, tau, rcond=None)
    return dict(A=A, active=active, resid=resid, dtau_ds=c, n_pairs=len(X), n_trajs=len(trajs))


@dataclass
class AugPoincareResult:
    exp: Experiment
    spec: SectionSpec
    s_star: np.ndarray
    tau_star: float
    spread: np.ndarray
    history: np.ndarray
    J_fd: np.ndarray
    A: np.ndarray
    active: np.ndarray
    fit_residual: float
    dtau_ds: np.ndarray
    eigvals: np.ndarray

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(self.eigvals)))

    def summary(self) -> str:
        n = self.spec.names()
        act = [x for x, a in zip(n, self.active) if a]
        pidx = self.spec.perturb_index()
        return "\n".join([
            f"== {self.exp.label}  v_cmd={self.exp.v_cmd}  section={self.spec.section}",
            "  s_bar*: " + ", ".join(f"{k}={v:+.4f}" for k, v in zip(n, self.s_star)),
            f"  return time tau* = {self.tau_star:.4f} s   orbit spread = {self.spread.max():.1e}",
            f"  fit on {act}",
            f"  lstsq residual = {self.fit_residual:.3f}",
            "  |eig| = " + " ".join(f"{a:.3f}" for a in np.sort(np.abs(self.eigvals))[::-1]),
            f"  spectral radius = {self.spectral_radius:.3f} "
            f"({'stable' if self.spectral_radius < 1 else 'NOT stable'})",
            "  dP_ii/ds_i (fd): " + ", ".join(
                f"{PERTURB_NAMES[j]}={self.J_fd[pidx[j], j]:+.3f}" for j in range(7)),
            "  dtau/ds: " + ", ".join(f"{k}={v:+.3f}" for k, v in zip(act, self.dtau_ds)),
        ])


def analyze(exp: Experiment, spec: SectionSpec, n_trials=32, n_steps=5, workers=None):
    lc = find_limit_cycle(exp, spec)
    J = jacobian_fd(exp, spec, lc["snap"], workers=workers)
    fit = jacobian_lstsq(exp, spec, lc["snap"], lc["s_star"], lc["tau_star"],
                         n_trials=n_trials, n_steps=n_steps, workers=workers)
    return AugPoincareResult(exp, spec, lc["s_star"], lc["tau_star"], lc["spread"], lc["history"],
                             J, fit["A"], fit["active"], fit["resid"], fit["dtau_ds"],
                             np.linalg.eigvals(fit["A"]))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_policy(text: str) -> tuple[str, dict]:
    """'raibert:k=0.1' -> ('raibert', {'k': 0.1})"""
    name, _, rest = text.partition(":")
    kw = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        kw[k] = _parse_value(v)
    return name, kw


def plot_results(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid = "#2b2b29", "#6b6a63", "#e4e3dd"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(th), np.sin(th), color=muted, lw=1, ls="--")
    for r, c in zip(results, colors):
        ev = r.eigvals
        ax1.scatter(ev.real, ev.imag, s=46, color=c, edgecolor="white", linewidth=1.5,
                    label=f"{r.exp.label}  rho={r.spectral_radius:.2f}", zorder=3)
        dphi = r.history[:, -1]
        ax2.plot(np.arange(len(dphi)), dphi * r.exp.ctrl.gait_cfg.period * 1e3, color=c, lw=2,
                 marker="o", ms=3, label=r.exp.label)
    ax1.set_aspect("equal"); ax1.set_xlim(-1.15, 1.15); ax1.set_ylim(-1.15, 1.15)
    ax1.set_xlabel("Re λ", color=ink); ax1.set_ylabel("Im λ", color=ink)
    ax1.set_title(f"Return-map eigenvalues ({results[0].spec.section} section)", color=ink,
                  fontsize=11, loc="left")
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax2.set_xlabel("crossing index (from rest)", color=ink)
    ax2.set_ylabel("touchdown time vs clock [ms]", color=ink)
    ax2.set_title("Clock phase at the section", color=ink, fontsize=11, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    for ax in (ax1, ax2):
        ax.grid(color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli(ap)
    p = ap.add_argument_group("poincare")
    p.add_argument("--section", choices=["touchdown", "clock"], default="touchdown")
    p.add_argument("--coords", choices=["centroidal", "base"], default="centroidal")
    p.add_argument("--feet", action="store_true", help="add FL/RR foot xy to the section state")
    p.add_argument("--compare", nargs="+", default=None, metavar="POLICY[:k=v,..]",
                   help="analyze several policies, e.g. raibert:k=0.03 raibert:k=0.17 openloop")
    p.add_argument("--trials", type=int, default=32)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default="results/poincare_aug")
    a = ap.parse_args()

    base = from_cli(a)
    spec = SectionSpec(section=a.section, coords=a.coords, feet=a.feet)
    exps = [base]
    if a.compare:
        exps = []
        for text in a.compare:
            e = copy.deepcopy(base)
            e.policy, e.policy_kw = parse_policy(text)
            exps.append(e)

    results = []
    for e in exps:
        try:
            r = analyze(e, spec, n_trials=a.trials, n_steps=a.steps, workers=a.workers)
        except RuntimeError as err:
            print("SKIP:", err)
            continue
        print(r.summary(), flush=True)
        results.append(r)
    if not results:
        return
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out + ".npz", names=spec.names(), labels=[r.exp.label for r in results],
             s_star=[r.s_star for r in results], tau_star=[r.tau_star for r in results],
             A=np.array([r.A for r in results], dtype=object), active=[r.active for r in results],
             eigvals=np.array([r.eigvals for r in results], dtype=object),
             J_fd=[r.J_fd for r in results], dtau_ds=np.array([r.dtau_ds for r in results], dtype=object),
             history=np.array([r.history for r in results], dtype=object),
             experiment=str(base.to_dict()), section=str(spec))
    plot_results(results, a.out + ".png")
    print("saved", a.out + ".npz", a.out + ".png")


if __name__ == "__main__":
    main()
