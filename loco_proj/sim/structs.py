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
JOINTS = ["hip", "thigh", "calf"]
INFO_KEYS = ["step", "prev_contacts", "lift_off", "foothold", "forces", "mpc_result",
             "clock_offset", "mpc_tick"]   # [CHANGED] + clock_offset, mpc_tick (see QuadrupedController)

# ---- [ADDED] begin ----
# State components used by the MPC period constraint (everything except g).
# PERIOD_SHIFT_IDX: components that are not periodic but shift by the
# reference displacement over one period (x, y, yaw).
PERIOD_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
PERIOD_SHIFT_IDX = (0, 1, 5)
# ---- [ADDED] end ----

# [ADDED] Small offset used when converting time -> phase. Times built from an
# integer step counter (k * dt) can land a hair below a sub-phase boundary
# (e.g. 0.49999999 instead of 0.5) and pick the wrong table row.
_PHASE_EPS = 1e-9


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

    # ---- [ADDED] begin: phase variable / period constraint / periodic reference ----
    # The MPC carries the gait phase phi_k = phi_0 + k dt / T along the horizon
    # (MPCSol.phase). It is exogenous (does not depend on the forces), so it is
    # not a QP decision variable; it is used to
    #   * pair steps one period apart for the period constraint, and
    #   * look up a periodic reference X_ref(phi) (ref_gen.PeriodicReference).
    #
    # Period constraint:  x_{k+N} = x_k (+ reference shift),  N = T / dt,
    # for k = period_k0 .. h - N.  Needs h >= N + period_k0 (e.g. h = 30 for
    # T = 0.5 s, dt = 0.025 s). x, y, yaw use the reference displacement over
    # one period; velocity/angular velocity xy parts are rotated by the
    # reference yaw change (periodic in the heading frame).
    #   "none" : off
    #   "soft" : add period_weight * ||x_{k+N} - x_k - shift||^2 to the cost
    #   "hard" : equality constraints; falls back to "soft" if infeasible
    # period_k0 = 0 includes x_0 (the measured state): the plan must return to
    # the current state -> perturbations are kept, not rejected. Use >= 1.
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

@dataclass
class LegControllerCfg:
    # [FIXED] shape (1, 3) -> (3,). With (1, 3), kp * (p_d - p) is (1, 3) and
    # Ji.T @ f fails (matmul of (3, 3) with (1, 3)).
    kp_swing: np.ndarray = field(default_factory =
            lambda: 400. * np.ones(3))
    kd_swing: np.ndarray = field(default_factory =
            lambda: 10. * np.ones(3))
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
    forces: np.ndarray        # (h, 4, 3) planned ground reaction forces (zero on swing legs)  [FIXED] comment
    X_pred: np.ndarray        # (h, 13) predicted states
    status: str
    solve_time: float
    # [ADDED] gait phase of x_0 .. x_h (unwrapped, in cycles) and the
    # period-constraint residual max|x_{k+N} - x_k - shift| of the plan.
    phase: np.ndarray = None
    period_residual: float = 0.0
    # --- [ADDED] --- 
    def copy_into(self, dst: "MPCSol"):
        np.copyto(dst.forces, self.forces)
        np.copyto(dst.X_pred, self.X_pred)
        dst.status = self.status
        dst.solve_time = self.solve_time
        if self.phase is not None:                       # [ADDED]
            if dst.phase is None or dst.phase.shape != self.phase.shape:
                dst.phase = self.phase.copy()
            else:
                np.copyto(dst.phase, self.phase)
        dst.period_residual = self.period_residual       # [ADDED]

    @classmethod
    def zeros(cls, h: int) -> "MPCSol":
        """[ADDED] empty solution; the controller needs one before the first solve."""
        return cls(np.zeros((h, 4, 3)), np.zeros((h, NX)), "init", 0.0, np.zeros(h + 1), 0.0)

    @classmethod
    def empty_like(cls, template: "MPCSol") -> "MPCSol":
        return cls(
            forces=np.empty_like(template.forces),
            X_pred=np.empty_like(template.X_pred),
            status=template.status,
            solve_time=template.solve_time,
            phase=None if template.phase is None else np.empty_like(template.phase),   # [ADDED]
            period_residual=template.period_residual,                                 # [ADDED]
        )
    # --- [ADDED] end --- 

@dataclass
class GaitCfg:
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

@dataclass
class ControllerCfg:
    mpc_cfg: MPC_Cfg = field(default_factory=MPC_Cfg)
    # [FIXED] field(default=GaitCfg) made the default the *class* itself, not an
    # instance (cfg.gait_cfg.period would read the class attribute, and
    # GaitScheduler(cfg.gait_cfg).table would not be converted). default_factory.
    gait_cfg: GaitCfg = field(default_factory=GaitCfg)
    swing_cfg: SwingTrajectoryCfg = field(default_factory=SwingTrajectoryCfg)
    leg_cfg: LegControllerCfg = field(default_factory=LegControllerCfg)
    t_stand: float = 0.5
    update_foothold_in_swing: bool = True
    step_per_mpc: int | None = None
    centroidal_state: bool = True


# ---- [ADDED] begin: simulation settings (applied by sim_env.make_model) ----
@dataclass
class SimCfg:
    """
    MuJoCo settings for one run. None = keep the value from the xml.

    Contact softness (floor <-> feet), MuJoCo solref, two ways:
        contact_solref = (timeconst, dampratio)   "standard" mode, e.g. (0.02, 1)
        contact_stiffness / contact_damping       "direct" mode, solref = (-k, -b)
    The direct mode wins if contact_stiffness is set. Smaller timeconst /
    larger stiffness = stiffer ground. timeconst should stay >= 2 * timestep.
    contact_solimp = (dmin, dmax, width, midpoint, power).
    """
    xml: str = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/scene.xml"
    timestep: float = 0.001
    integrator: str | None = None          # "euler" | "implicitfast" | "implicit" | "rk4"
    obstacles: bool = False                # keep the bars/stairs of the stock scene
    gravity: float = -9.81
    # contact
    contact_solref: tuple | None = None
    contact_stiffness: float | None = None
    contact_damping: float | None = None
    contact_solimp: tuple | None = None
    friction: float | None = None          # sliding friction, floor and feet
    condim: int | None = None              # 1, 3, 4, 6 for the feet
    cone: str | None = None                # "pyramidal" | "elliptic"
    impratio: float | None = None
    noslip_iterations: int | None = None
    solver_iterations: int | None = None
    # robot
    mass_scale: float = 1.0                # scales every body mass + inertia
    torque_scale: float = 1.0              # scales actuator ctrlrange
    # initial state
    init_height_offset: float = 0.0        # extra base height at reset [m]
# ---- [ADDED] end ----


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
    # [FIXED comment] not Euler: exact zero-order hold expm([[A, B], [0, 0]] dt).
    # M^3 = 0 for this model, so I + M dt + (M dt)^2 / 2 is the full series.
    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc
    Mdt = M * dt
    E = np.eye(NX + NU) + Mdt + 0.5 * Mdt @ Mdt
    return E[:NX, :NX], E[:NX, NX:]
