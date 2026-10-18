import numpy as np
from dataclasses import dataclass, field

IDX_P = slice(0, 3)      # position
IDX_RPY = slice(3, 6)    # [roll, pitch, heading]
IDX_V = slice(6, 9)      # trans vel
IDX_W = slice(9, 12)     # angular vel
IDX_G = 12               # Gravity acceleration
GRAVITY = -9.81


@dataclass
class MPC_Config:
    dt: float = 1e-3
    h: int = 30
    mu: float = 0.5
    fz_max: float = 45.43 + 23.7
    fz_min: float = 0.0
    Q_diag: np.ndarray = field(default_factory=lambda: np.ones(13))
    R_diag: np.ndarray = field(default_factory=lambda: np.ones(12))
    zd: float = 0.27


def ref_gen(x0: np.ndarray, v_cmd_b: np.ndarray, cfg: MPC_Config) -> np.ndarray:
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

    Xs[0, 1:] = x0[0] + np.cumsum(vx_w) * dt
    Xs[1, 1:] = x0[1] + np.cumsum(vy_w) * dt
    Xs[2, 1:] = cfg.zd

    Xs[3, 1:] = 0.0
    Xs[4, 1:] = 0.0
    Xs[5, 1:] = yaw + wz * dt

    Xs[6, 1:] = vx_w
    Xs[7, 1:] = vy_w
    Xs[8, 1:] = 0.0
    Xs[9, 1:] = 0.0
    Xs[10, 1:] = 0.0
    Xs[11, 1:] = wz

    Xs[IDX_G, 1:] = GRAVITY
    return Xs

if __name__ == "__main__":
    x0 = np.zeros((13,))
    vcmd_b = np.array([1,0,0])
    cfg = MPC_Config()
    print(ref_gen(x0, vcmd_b, cfg))
