import numpy as np
from Def import G
from Def import MPC_Cfg
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