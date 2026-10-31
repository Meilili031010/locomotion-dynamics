from dataclasses import field, dataclass
import numpy as np

NX = 13
NU = 12
LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = ["hip", "thigh", "calf"]
G = -9.81
IDX_KEYS = [f"{leg}_{jnt}_joint" for leg in LEGS for jnt in JOINTS]
_PHASE_EPS = 1e-9

@dataclass
class GaitCfg:
    period: float = 0.5
    """
        table encoding:
        0 denotes stance leg; values for nonzero element denote terminal time for swing leg
    """
    table: np.ndarray = field(default_factory=lambda: np.array([
        [0, 0.5, 0.5, 0],   # FR, RL swing
        [1, 0, 0, 1],   # FL, RR swing
    ], dtype=float))

    def __post_init__(self):
        self.table = np.asarray(self.table, dtype=float)
        assert self.period > 0
        assert self.table.ndim == 2 and self.table.shape[1] == 4
        assert np.all(self.table >= 0)

@dataclass
class SwingTrajectoryCfg:
    step_height: float = 0.08
    t_swing: float = 0.25
    touchdown_velocity: float = 0.
    k_fb: float = None

@dataclass(slots=True)
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
    # [ADDED] compute_foot_placement() read footCfg.use_centrifugal, which did
    # not exist (AttributeError with slots=True).
    use_centrifugal: bool = True

@dataclass
class MPC_Cfg:
    # [CHANGED] dt 1e-3 -> 0.025, h 30 -> 20.
    # Cheetah 3 uses MPC steps of ~25-40 ms and a horizon of about one gait
    # cycle. 1 ms x 30 only looked 30 ms ahead (not even one footstep).
    # 0.025 s divides the 0.25 s trot half-period, so contact switches line up
    # with MPC steps. Horizon = 20 x 0.025 = 0.5 s = one trot cycle.
    dt: float = 0.025
    h: int = 20
    mu: float = 0.28
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

    period_constraint: str = "none"
    # Which step pairs (k, k+N) are constrained:
    #   "last"    : only k = h - N  -> the last period of the plan is a closed
    #               orbit (periodic terminal constraint). Default.
    #   "section" : one pair at a gait-phase boundary (phi_k integer = the
    #               Poincare section, FL/RR touchdown), first one with k >= k0
    #               that fits in the horizon; else falls back to "last".
    #   "all"     : every k = k0 .. h-N. Over-determined: each step actuates only
    #               the 6 velocity components, so it is usually infeasible as
    #               "hard" and drags the plan back towards the measured x_0.
    period_pairs: str = "last"
    period_k0: int = 1
    period_weight: float = 10.0
    period_idx: tuple = PERIOD_IDX
    # Use the periodic reference given to the controller (if any) instead of
    # the constant-velocity ref_gen().
    periodic_ref: bool = False
    # Which reference terms follow the periodic table; the others are the
    # constant ref_gen values (rpy = 0, w = [0, 0, wz_cmd], z = zd).
    # Tracking roll/pitch/w recorded from the robot's own cycle makes the MPC
    # re-create body motion caused by the swing legs (invisible to its model):
    # pitch swing doubled in tests. ("z", "v") avoids that.
    periodic_ref_terms: tuple = ("z", "roll", "pitch", "v", "w")
    # ---- [ADDED] end ----