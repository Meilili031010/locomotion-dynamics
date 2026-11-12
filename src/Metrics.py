"""
[NEW FILE] Energy / performance measures for comparing foot placement
policies and gaits.

    meter = EnergyMeter(model, ctrl, t_start=2.0)
    log, fallen = run(model, data, ctrl, T, cmd, callbacks=[meter])
    print(meter.result())

Or in one call:  evaluate(exp, duration) -> dict   (used by Gait_opt.py)
"""
import numpy as np

from Def import G, JOINTS


class EnergyMeter:
    """
    Callback for Sim.run (called after every sim step). Integrates from
    t_start on (skip standing + command ramp):
        E_abs   sum_j |tau_j qd_j| dt     mechanical energy, braking counted as cost
        E_pos   sum_j max(tau_j qd_j, 0) dt   positive work only
        tau_sq  sum_j tau_j^2 dt          Joule-heating proxy (motor-independent)
    Cost of transport  CoT = E / (m g d),  d = horizontal distance of the base.
    Also mean |tau| per joint type (hip, thigh, calf) over STANCE legs: the
    force-aligned policy should mainly lower the hip/thigh values.
    """

    def __init__(self, model, ctrl, t_start: float = 0.0):
        self.ctrl = ctrl
        self.dt = model.opt.timestep
        self.mass = float(model.body_subtreemass[ctrl.st.base_idx])
        self.t_start = t_start
        self.E_abs = self.E_pos = self.tau_sq = 0.0
        self.T = 0.0
        self.p0 = None
        self.p1 = None
        self.stance_tau = np.zeros(3)
        self.stance_n = 0
        self.align_sum = 0.0        # fz-weighted angle between planned force and hip-foot line
        self.align_w = 0.0

    def __call__(self, model, data, ctrl) -> bool:
        if data.time < self.t_start:
            return False
        if self.p0 is None:
            self.p0 = data.qpos[:2].copy()
        st = ctrl.st
        tau, qd = st.jnt_tau, st.jnt_vel           # applied torque, (4, 3)
        p = tau * qd
        self.E_abs += np.abs(p).sum() * self.dt
        self.E_pos += np.clip(p, 0.0, None).sum() * self.dt
        self.tau_sq += (tau ** 2).sum() * self.dt
        self.T += self.dt
        c = ctrl.info_leg["contacts"]
        if c.any():
            self.stance_tau += np.abs(tau[c]).sum(axis=0)
            self.stance_n += int(c.sum())
            hip = data.qpos[:3] + ctrl.hip_point_b @ st.R_wb.T     # (4, 3) hip points, world
            leg = hip - st.foot_pos
            f = ctrl.info_leg["forces"]
            for i in np.flatnonzero(c):
                fn = np.linalg.norm(f[i])
                if f[i, 2] > 1.0:
                    cosang = leg[i] @ f[i] / (np.linalg.norm(leg[i]) * fn)
                    self.align_sum += f[i, 2] * np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                    self.align_w += f[i, 2]
        self.p1 = data.qpos[:2].copy()
        return False

    def result(self) -> dict:
        if self.p0 is None or self.T == 0.0:
            return {"T": 0.0}
        d = float(np.linalg.norm(self.p1 - self.p0))
        mgd = self.mass * -G * max(d, 1e-6)
        out = dict(T=self.T, distance=d, speed=d / self.T,
                   E_abs=self.E_abs, E_pos=self.E_pos, power_abs=self.E_abs / self.T,
                   CoT_abs=self.E_abs / mgd, CoT_pos=self.E_pos / mgd,
                   tau_sq_rate=self.tau_sq / self.T)
        if self.align_w:
            out["stance_align_deg"] = self.align_sum / self.align_w
        if self.stance_n:
            for j, name in enumerate(JOINTS):
                out[f"stance_|tau|_{name}"] = self.stance_tau[j] / self.stance_n
        return out


def format_result(r: dict) -> str:
    if r.get("T", 0.0) == 0.0:
        return "energy: nothing measured (t_start after the end?)"
    s = (f"energy over {r['T']:.1f}s: speed={r['speed']:.3f} m/s  CoT_abs={r['CoT_abs']:.3f}  "
         f"CoT_pos={r['CoT_pos']:.3f}  power={r['power_abs']:.1f} W  tau^2 rate={r['tau_sq_rate']:.0f}")
    if "stance_|tau|_hip" in r:
        s += ("\nstance mean |tau| [N m]: " +
              "  ".join(f"{j}={r[f'stance_|tau|_{j}']:.2f}" for j in JOINTS))
    if "stance_align_deg" in r:
        s += f"   force-leg angle={r['stance_align_deg']:.2f} deg (fz-weighted)"
    return s


def evaluate(exp, duration: float = 6.0, t_measure: float | None = None,
             stop_on_fall: bool = True) -> dict:
    """
    Build and run one Experiment headless, return the metrics plus
    fallen, mean commanded speed and the MPC failure ratio.
    t_measure: start of the energy window (default: after standing + ramp + 1 s).
    """
    from Sim import run
    model, data, ctrl = exp.build()
    if t_measure is None:
        t_measure = exp.ctrl.t_stand + exp.t_ramp + 1.0
    meter = EnergyMeter(model, ctrl, t_measure)
    log, fallen = run(model, data, ctrl, duration, exp.command(), log_every=10,
                      callbacks=[meter], realtime=False, stop_on_fall=stop_on_fall)
    L = log.arrays()
    r = meter.result()
    tail = L["t"] >= t_measure
    status = L["mpc_status"][tail] if tail.any() else np.array([])
    r.update(fallen=bool(fallen), t_end=float(L["t"][-1]) if len(L["t"]) else 0.0,
             v_cmd=float(np.linalg.norm(exp.v_cmd[:2])),
             mpc_fail_ratio=float(np.mean(["solved" not in str(s) for s in status])) if len(status) else 0.0)
    return r
