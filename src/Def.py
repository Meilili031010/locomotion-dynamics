from dataclasses import field, dataclass
import numpy as np

NX = 13
NU = 12

# IDX for 
IDX_P = slice(0, 3)      # position
IDX_RPY = slice(3, 6)    # [roll, pitch, heading]
IDX_V = slice(6, 9)      # trans vel
IDX_W = slice(9, 12)     # angular vel
IDX_RPY = slice(3, 6)
IDX_G = 12


LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = ["hip", "thigh", "calf"]
G = -9.81
IDX_KEYS = [f"{leg}_{jnt}_joint" for leg in LEGS for jnt in JOINTS]
_PHASE_EPS = 1e-9
PERIOD_IDX = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
PERIOD_SHIFT_IDX = (0, 1, 5) # idx reduced

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

# ---- [ADDED] end ----



