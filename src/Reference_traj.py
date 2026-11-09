import numpy as np
from Def import G, IDX_RPY, IDX_G
from Def import MPC_Cfg, PeriodicReference
from Utils import rot_z
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
    Xs[IDX_G, 0] = G  # [ADDED] x0 from a sensor/zeros may not carry the gravity state

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

    Xs[IDX_G, 1:] = G
    return Xs

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
    Xs[IDX_G, 0] = G
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
    Xs[IDX_G, 1:] = G
    return Xs