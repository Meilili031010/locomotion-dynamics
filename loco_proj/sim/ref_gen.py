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
    # [CHANGED] dt 1e-3 -> 0.025, h 30 -> 20.
    # Cheetah 3 uses MPC steps of ~25-40 ms and a horizon of about one gait
    # cycle. 1 ms x 30 only looked 30 ms ahead (not even one footstep).
    # 0.025 s divides the 0.25 s trot half-period, so contact switches line up
    # with MPC steps. Horizon = 20 x 0.025 = 0.5 s = one trot cycle.
    dt: float = 0.025
    h: int = 20
    # [CHANGED] 0.5 -> 0.28. MuJoCo's Go2 feet have friction 0.4 on an
    # elliptic (round) cone. The MPC uses a friction *pyramid*, whose corners
    # allow |f_xy| = sqrt(2) * mu * fz; the pyramid inside a 0.4 cone has
    # mu = 0.4 / sqrt(2) ~ 0.28. With 0.5 the MPC asked for forces that slip.
    mu: float = 0.28
    # [CHANGED] fz_max was 45.43 + 23.7 (sum of joint torque limits, N*m, not a
    # force). Go2 weighs ~149 N, so a trot pair carries ~75 N/leg at rest;
    # 160 N leaves room for dynamic loading.
    fz_max: float = 160.0
    fz_min: float = 0.0
    # [CHANGED] was all ones. Started from the MIT convexMPC weights, then tuned
    # in sim for Go2 (state layout [p(3), rpy(3), v(3), w(3), g]). Gravity
    # entry must have zero weight.
    Q_diag: np.ndarray = field(default_factory=lambda: np.array(
        [1.0, 1.0, 50.0,      # p
         10.0, 10.0, 10.0,    # roll, pitch, yaw
         10.0, 10.0, 1.0,     # v
         0.1, 0.1, 1.0,       # w
         0.0]))               # g
    R_diag: np.ndarray = field(default_factory=lambda: np.full(12, 1e-5))  # [CHANGED] alpha ~ 1e-5 (Cheetah: 4e-5)
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
    Xs[IDX_G, 0] = GRAVITY  # [ADDED] x0 from a sensor/zeros may not carry the gravity state

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
