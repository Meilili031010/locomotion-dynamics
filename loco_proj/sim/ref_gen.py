import numpy as np
from structs import MPC_Cfg, IDX_G, IDX_P, IDX_RPY, IDX_V, IDX_W, GRAVITY

def ref_gen(x0: np.ndarray, v_cmd_b: np.ndarray, cfg: MPC_Cfg) -> np.ndarray:
    """
    x0: (13,)
    v_cmd_b: [vxd_b, vyd_b, wzd_b] (3,) 
    Xs: (13, h+1)
    """
    dt, h = cfg.dt, cfg.h
    vx_b, vy_b, wz = v_cmd_b

    k = np.arange(h)
    yaw = x0[IDX_RPY][2] + wz * dt * k
    c, s = np.cos(yaw), np.sin(yaw)
    vx_w = c * vx_b - s * vy_b
    vy_w = s * vx_b + c * vy_b

    Xs = np.zeros((13, h + 1))
    Xs[:, 0] = x0
    Xs[IDX_G, 0] = GRAVITY  # [ADDED] x0 from a sensor/zeros may not carry the gravity state

    Xs[0, 1:] = x0[0] + np.cumsum(vx_w) * dt
    Xs[1, 1:] = x0[1] + np.cumsum(vy_w) * dt
    Xs[2, 1:] = cfg.zd

    Xs[3, 1:] = 0.0
    Xs[4, 1:] = 0.0
    Xs[5, 1:] = yaw + wz * dt

    # [REVIEW NOTE] column k+1 holds yaw_{k+1} but a velocity rotated by yaw_k
    # (one-step lag when turning). Harmless at MPC rates; ref_gen_periodic
    # uses yaw_{k+1} for both.
    Xs[6, 1:] = vx_w
    Xs[7, 1:] = vy_w
    Xs[8, 1:] = 0.0
    Xs[9, 1:] = 0.0
    Xs[10, 1:] = 0.0
    Xs[11, 1:] = wz

    Xs[IDX_G, 1:] = GRAVITY
    return Xs



# ---- [ADDED] begin: periodic reference trajectory ----
from dataclasses import dataclass, field
from structs import rot_z


@dataclass
class PeriodicReference:
    """
    Reference over one gait cycle, indexed by the gait phase phi in [0, 1).

    Stored in the heading (yaw) frame so it is independent of where the robot
    is and which way it faces:
        z, roll, pitch      (M,)
        v_h                 (M, 3) linear velocity, heading frame
        w_h                 (M, 3) angular velocity, heading frame (w_h[:, 2] = yaw rate)
    x, y and yaw are not stored; ref_gen_periodic() integrates them.

    Build it with
        PeriodicReference.constant(v_cmd_b, zd, period)   same as ref_gen()
        PeriodicReference.from_cycles(phase, X, period)    from a recorded steady
                                                           cycle (the robot's own
                                                           limit cycle)
    """
    phase: np.ndarray
    z: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    v_h: np.ndarray
    w_h: np.ndarray
    period: float
    v_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def sample(self, phi: np.ndarray) -> dict:
        """Linear interpolation, periodic in phi (phi may be unwrapped)."""
        ph = np.mod(phi, 1.0)
        g = self.phase
        f = lambda y: np.interp(ph, g, y, period=1.0)
        return dict(z=f(self.z), roll=f(self.roll), pitch=f(self.pitch),
                    v_h=np.stack([f(self.v_h[:, i]) for i in range(3)], axis=-1),
                    w_h=np.stack([f(self.w_h[:, i]) for i in range(3)], axis=-1))

    def mean_cmd(self) -> np.ndarray:
        """[vx, vy, wz] averaged over the cycle (heading frame)."""
        return np.array([self.v_h[:, 0].mean(), self.v_h[:, 1].mean(), self.w_h[:, 2].mean()])

    @classmethod
    def constant(cls, v_cmd_b, zd: float, period: float, n: int = 50):
        v = np.asarray(v_cmd_b, dtype=float)
        ph = np.arange(n) / n
        v_h = np.zeros((n, 3)); v_h[:, 0], v_h[:, 1] = v[0], v[1]
        w_h = np.zeros((n, 3)); w_h[:, 2] = v[2]
        z = np.full(n, zd)
        return cls(ph, z, np.zeros(n), np.zeros(n), v_h, w_h, period, v.copy())

    @classmethod
    def from_cycles(cls, phase: np.ndarray, X: np.ndarray, period: float,
                    n_bins: int = 50, n_cycles: int | None = None, v_cmd=None):
        """
        phase : (N,) unwrapped gait phase of each sample (cycles)
        X     : (N, 13) MPC states [p, rpy, v, w_world, g] at those samples
        Uses the last n_cycles complete cycles (all if None), averages each
        quantity in n_bins phase bins.
        """
        phase = np.asarray(phase); X = np.asarray(X)
        if n_cycles is not None:
            last = np.floor(phase.max())
            keep = phase >= last - n_cycles
            keep &= phase < last
            phase, X = phase[keep], X[keep]
        yaw = X[:, 5]
        c, s = np.cos(yaw), np.sin(yaw)
        def to_h(vw):   # world -> heading frame
            return np.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1], vw[:, 2]], -1)
        v_h, w_h = to_h(X[:, 6:9]), to_h(X[:, 9:12])
        b = np.minimum((np.mod(phase, 1.0) * n_bins).astype(int), n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        if cnt.sum() == 0:
            raise ValueError("no samples in the selected cycles")
        ph = (np.arange(n_bins) + 0.5) / n_bins
        full = cnt > 0
        def avg(y):    # bin mean; empty bins filled by periodic interpolation
            m = np.bincount(b, weights=y, minlength=n_bins)[full] / cnt[full]
            return np.interp(ph, ph[full], m, period=1.0)
        return cls(ph, avg(X[:, 2]), avg(X[:, 3]), avg(X[:, 4]),
                   np.stack([avg(v_h[:, i]) for i in range(3)], -1),
                   np.stack([avg(w_h[:, i]) for i in range(3)], -1),
                   period, np.zeros(3) if v_cmd is None else np.asarray(v_cmd, float))

    def save(self, path: str):
        np.savez(path, **{k: np.asarray(v) for k, v in self.__dict__.items()})

    @classmethod
    def load(cls, path: str):
        d = np.load(path)
        return cls(**{k: (float(d[k]) if k == "period" else d[k]) for k in d.files})


def ref_gen_periodic(x0: np.ndarray, phase0: float, v_cmd_b: np.ndarray, cfg: MPC_Cfg,
                     pref: PeriodicReference, match_cmd: bool = True,
                     terms: tuple | None = None) -> np.ndarray:
    """
    Same output as ref_gen() (13, h+1), but z, roll, pitch, v, w follow the
    periodic reference at the predicted phase phi_k = phase0 + k dt / T.
    x, y, yaw are integrated from the reference velocities starting at x0.
    match_cmd: shift v_h / yaw rate so their cycle mean equals v_cmd_b, so a
    reference recorded at one speed can be reused at another.
    terms: reference terms taken from the table (default cfg.periodic_ref_terms);
    the rest use the constant ref_gen values.
    """
    dt, h = cfg.dt, cfg.h
    terms = cfg.periodic_ref_terms if terms is None else terms
    phi = phase0 + dt * np.arange(h + 1) / pref.period
    r = pref.sample(phi)
    v_cmd_b = np.asarray(v_cmd_b, float)
    if "z" not in terms:
        r["z"] = np.full(h + 1, cfg.zd)
    for name in ("roll", "pitch"):
        if name not in terms:
            r[name] = np.zeros(h + 1)
    if "v" not in terms:
        r["v_h"] = np.zeros((h + 1, 3)); r["v_h"][:, :2] = v_cmd_b[:2]
    if "w" not in terms:
        r["w_h"] = np.zeros((h + 1, 3)); r["w_h"][:, 2] = v_cmd_b[2]
    v_h, w_h = r["v_h"].copy(), r["w_h"].copy()
    if match_cmd:
        d = v_cmd_b - pref.mean_cmd()
        if "v" not in terms:
            d[:2] = 0.0
        if "w" not in terms:
            d[2] = 0.0
        v_h[:, 0] += d[0]; v_h[:, 1] += d[1]; w_h[:, 2] += d[2]

    Xs = np.zeros((13, h + 1))
    Xs[:, 0] = x0
    Xs[IDX_G, 0] = GRAVITY
    yaw = np.empty(h + 1); yaw[0] = x0[5]
    p = np.empty((h + 1, 2)); p[0] = x0[:2]
    for k in range(h):
        R = rot_z(yaw[k])
        p[k + 1] = p[k] + (R[:2, :2] @ v_h[k, :2]) * dt
        yaw[k + 1] = yaw[k] + w_h[k, 2] * dt
    for k in range(1, h + 1):
        R = rot_z(yaw[k])
        Xs[0:2, k] = p[k]
        Xs[2, k] = r["z"][k]
        Xs[3, k], Xs[4, k], Xs[5, k] = r["roll"][k], r["pitch"][k], yaw[k]
        Xs[6:9, k] = R @ v_h[k]
        Xs[9:12, k] = R @ w_h[k]
    Xs[IDX_G, 1:] = GRAVITY
    return Xs
# ---- [ADDED] end ----


if __name__ == "__main__":   # [CHANGED] moved below the added code
    x0 = np.zeros((13,))
    vcmd_b = np.array([1,0,0])
    cfg = MPC_Cfg()
    print(ref_gen(x0, vcmd_b, cfg))
