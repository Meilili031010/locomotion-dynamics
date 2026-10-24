from dataclasses import dataclass, field
import numpy as np

IDX_P = slice(0, 3)      # position
IDX_RPY = slice(3, 6)    # [roll, pitch, heading]
IDX_V = slice(6, 9)      # trans vel
IDX_W = slice(9, 12)     # angular vel
IDX_G = 12               # Gravity acceleration
GRAVITY = -9.81
NX = 13
NU = 12
LEGS = ["FL", "FR", "RL", "RR"]

# [ADDED] Small offset used when converting time -> phase. Times built from an
# integer step counter (k * dt) can land a hair below a sub-phase boundary
# (e.g. 0.49999999 instead of 0.5) and pick the wrong table row.
_PHASE_EPS = 1e-9


@dataclass
class FootPlacementCfg:
    leg: int
    p_hip: np.ndarray        # (3,) shoulder point now, world
    r_hip: np.ndarray        # (3,) shoulder offset from base, world (yaw-rotated)
    v_base: np.ndarray       # (3,) base linear velocity, world
    w_base: np.ndarray       # (3,) base angular velocity, world
    v_cmd: np.ndarray        # (3,) [vx, vy, wz] command, world frame
    height: float            # base height
    yaw: float
    t_stance: float
    t_swing: float
    t_swing_remaining: float

@dataclass
class MPC_Cfg:
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

@dataclass
class LegControllerCfg:
    kp_swing: np.ndarray = field(default_factory = 
            lambda: 400. * np.ones((1,3)))
    kd_swing: np.ndarray = field(default_factory = 
            lambda: 10. * np.ones((1,3)))
    # swing_mb_g: bool = True
    # swing_mb_acc: bool = True
    # stance_mb_g: bool = True
    # jnt_fric: bool = True
    jnt_fric_coeff: float = 0.05

@dataclass
class SwingTrajectoryCfg:
    step_height: float = 0.08
    t_swing: float = 0.25
    touchdown_velocity: float = 0.
    gravity: float = 9.81
    k_fb: float = None

@dataclass
class MPCSol:
    forces: np.ndarray        # (h, 4, 3) planned net forces on swing legs
    X_pred: np.ndarray        # (h, 13) predicted states
    status: str
    solve_time: float

@dataclass
class GaitConfig:
    period: float = 0.5
    """
        table encoding:
        0 denotes stance leg; values for nonzero element denote terminal time for swing leg
    """
    table: np.ndarray = field(default_factory=lambda: np.array([
        [0, 0.5, 0.5, 0],   # FR、RL swing
        [1, 0, 0, 1],   # FL、RR swing
    ], dtype=float))

    def __post_init__(self):
        self.table = np.asarray(self.table, dtype=float)
        assert self.period > 0
        assert self.table.ndim == 2 and self.table.shape[1] == 4
        assert np.all(self.table >= 0)

def skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def continuous_dynamics(yaw: float, r_foot: np.ndarray, mass: float,
                        I_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ac (13, 13), Bc (13, 12) of the single-rigid-body model.
    r_feet: (4, 3) foot position minus CoM, world frame.
    """
    Rz = rot_z(yaw)
    I_w_inv = np.linalg.inv(Rz @ I_body @ Rz.T)

    Ac = np.zeros((NX, NX))
    Ac[0:3, 6:9] = np.eye(3)
    Ac[3:6, 9:12] = Rz.T
    Ac[8, 12] = 1.0                     # v_z_dot += g  (g state holds -9.81)

    Bc = np.zeros((NX, NU))
    for i in range(4):
        Bc[6:9, 3 * i:3 * i + 3] = np.eye(3) / mass
        Bc[9:12, 3 * i:3 * i + 3] = I_w_inv @ skew(r_foot[i])
    return Ac, Bc


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    # Euler Integration
    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc
    Mdt = M * dt
    E = np.eye(NX + NU) + Mdt + 0.5 * Mdt @ Mdt
    return E[:NX, :NX], E[:NX, NX:]
