"""
GAIT PLANNING LIBRARY
=====================

Reference for writing
    sim/gait.py        gait clock, contact schedule, swing timing
    sim/ref_gen.py     body reference trajectory for the MPC
    sim/foot_place.py  where to put the foot (policies)
    sim/swing_traj.py  how the foot travels there (Bezier swing)

Only numpy; every example runs on its own:
    python gait_lib.py           # all
    python gait_lib.py B J K     # chosen sections

Sections
    A  Gait clock and phase          phase = t / T mod 1, float pitfalls
    B  Contact table + lookup        np.searchsorted, sub-phases
    C  Durations                     stance / swing time per leg, time since lift-off
    D  Events                        touchdown / lift-off from boolean arrays
    E  Contact horizon               (h, 4) table for the MPC, vectorized
    F  Standing start, clock offset  gait time vs sim time
    G  Frames                        body/heading command -> world
    H  Constant-velocity reference   integrate x, y, yaw over the horizon
    I  Periodic reference            phase-indexed table, np.interp(period=1), binning
    J  Foot placement                Raibert, Cheetah 3 terms, capture point, clamping
    K  Swing trajectory              cubic Bezier + derivatives, z profile, min-jerk
    L  Policy pattern                callable classes + name registry
"""
import sys

import numpy as np

LEGS = ["FL", "FR", "RL", "RR"]


def _title(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ==========================================================================
# A. GAIT CLOCK AND PHASE
# ==========================================================================
"""
period T (trot: 0.5 s).   phase phi = (t / T) mod 1  in [0, 1)
cycle index  = floor(t / T)          unwrapped phase = t / T  (keeps counting cycles)
    The unwrapped phase is what the Poincare analysis and the periodic
    reference use; the wrapped one selects the contact pattern.

Pitfall (float): t built as n * dt can be 0.49999999999 instead of 0.5,
so floor / searchsorted picks the PREVIOUS sub-phase for one sim step.
Fix: add a tiny epsilon before wrapping:  (t / T + 1e-9) % 1.0
(structs._PHASE_EPS). Use the same epsilon everywhere you convert time -> phase.
"""
PHASE_EPS = 1e-9


def example_A():
    _title("A. phase and the epsilon pitfall")
    T, dt = 0.5, 0.001
    t = 0.0
    for _ in range(1250):                      # accumulate t += dt, like a sim clock
        t += dt
    print(f"after 1250 steps t = {t!r}  (should be 1.25 -> phase 0.5, second half)")
    print("phase without eps:", (t / T) % 1.0, " -> sub-phase", int(2 * ((t / T) % 1.0)), "(wrong)")
    print("phase with eps   :", (t / T + PHASE_EPS) % 1.0, " -> sub-phase",
          int(2 * ((t / T + PHASE_EPS) % 1.0)))
    print("(even better: keep an integer step counter n and use t = n * dt)")


# ==========================================================================
# B. CONTACT TABLE + LOOKUP
# ==========================================================================
"""
One way to encode a gait (sim/ GaitCfg.table): rows = sub-phases, cols = legs
[FL, FR, RL, RR]; 0 = stance, nonzero = swing and the value is the phase at
which that sub-phase ENDS.
    trot:  [[0, .5, .5, 0],      phase 0.0-0.5: FR, RL swing
            [1,  0,  0, 1]]      phase 0.5-1.0: FL, RR swing
    ends   = table.max(axis=1)            [0.5, 1.0]
    swing  = table > 0                    (n_rows, 4) bool
    starts = [0, ends[:-1]]
Other encodings work too (list of (start, end) swing windows per leg; a
"duty factor + offset" per leg). Pick one that makes YOUR lookups easy.

Find the sub-phase of a phase value:
    np.searchsorted(ends, phi, side="right")
    side="right": phi exactly on a boundary (0.5) belongs to the NEXT row.
contacts = ~swing[row]     (True = stance)

Validate a table in the constructor: ends strictly increasing, last end == 1,
every leg has at least one stance row (asserts in gait.GaitScheduler).
"""
TROT = np.array([[0, 0.5, 0.5, 0], [1, 0, 0, 1]], dtype=float)
# a 4-beat walk for experiments: one leg swings at a time
WALK = np.array([[0.25, 0, 0, 0], [0, 0, 0, 0.5], [0, 0.75, 0, 0], [0, 0, 1.0, 0]], dtype=float)


def table_rows(table):
    ends = table.max(axis=1)
    swing = table > 0
    starts = np.concatenate([[0.0], ends[:-1]])
    return starts, ends, swing


def example_B():
    _title("B. contact table lookup (trot)")
    starts, ends, swing = table_rows(TROT)
    for phi in [0.0, 0.25, 0.4999, 0.5, 0.75]:
        row = int(np.searchsorted(ends, phi, side="right"))
        print(f"phi={phi:<6} row={row} contacts={(~swing[row]).astype(int)}  ({LEGS})")


# ==========================================================================
# C. DURATIONS
# ==========================================================================
"""
Sub-phase lengths:  d = ends - starts        (fractions of T)
stance time of leg i per cycle = T * sum(d[~swing[:, i]])
swing  time of leg i per cycle = T - stance time       (one swing per cycle)
    -> trot T = 0.5: 0.25 s stance, 0.25 s swing; duty factor 0.5

Time since lift-off (needed by the swing trajectory, foot placement):
    walk backwards from the current row while the leg is swinging,
    swing start = start of the first swinging row (careful at the wrap
    1.0 -> 0.0: a swing can span the end of the cycle, e.g. a row ending at
    1.0 followed by row 0, both swinging).
    elapsed = t - swing_start_time;   remaining = t_swing - elapsed
"""


def example_C():
    _title("C. stance / swing durations")
    T = 0.5
    for name, tab in [("trot", TROT), ("walk", WALK)]:
        starts, ends, swing = table_rows(tab)
        d = ends - starts
        st = [T * d[~swing[:, i]].sum() for i in range(4)]
        print(f"{name}: stance {np.round(st, 3)}  swing {np.round(T - np.array(st), 3)}")

    # elapsed swing time, simple case: one swing block per leg, no wrap
    starts, ends, swing = table_rows(TROT)
    t = 0.37
    phi = (t / T + PHASE_EPS) % 1.0
    row = int(np.searchsorted(ends, phi, side="right"))
    cycle_start = np.floor(t / T + PHASE_EPS) * T
    elapsed = np.where(swing[row], t - (cycle_start + starts[row] * T), 0.0)
    print(f"t={t}: swing elapsed per leg {np.round(elapsed, 3)}")


# ==========================================================================
# D. EVENTS
# ==========================================================================
"""
Keep the previous contacts c_prev (4,) bool:
    touchdown = c & ~c_prev          swing -> stance this step
    liftoff   = ~c & c_prev          stance -> swing this step
Boolean arrays index other arrays:
    lift_off_pos[liftoff] = p_foot[liftoff]     record where each swing starts
Pitfall: update c_prev AFTER using it, every step (a [FIXED] bug in quad_ctrl:
c_prev was never updated so the swing always started from the initial foot).
"""


def example_D():
    _title("D. touchdown / lift-off events")
    c_prev = np.array([1, 1, 1, 1], bool)
    lift_off_pos = np.zeros((4, 3))
    p_foot = np.arange(12.0).reshape(4, 3)
    for c in [np.array([1, 0, 0, 1], bool), np.array([1, 0, 0, 1], bool), np.array([0, 1, 1, 0], bool)]:
        td, lo = c & ~c_prev, ~c & c_prev
        lift_off_pos[lo] = p_foot[lo]
        print("contacts", c.astype(int), "touchdown", td.astype(int), "liftoff", lo.astype(int))
        c_prev = c
    print("recorded lift-off rows:\n", lift_off_pos)


# ==========================================================================
# E. CONTACT HORIZON
# ==========================================================================
"""
The MPC needs contacts at t0, t0 + dt_mpc, ..., t0 + (h-1) dt_mpc:
    times = t0 + dt * np.arange(h)
    rows  = np.searchsorted(ends, (times / T + eps) % 1.0, side="right")
    C_h   = ~swing[rows]            (h, 4)  -- fancy indexing, no loop
Pick dt_mpc so it divides the sub-phase length (0.25 / 0.025 = 10) and contact
switches fall exactly on MPC steps.
"""


def contacts_horizon(t0, h, dt, T=0.5, table=TROT):
    _, ends, swing = table_rows(table)
    times = t0 + dt * np.arange(h)
    rows = np.searchsorted(ends, (times / T + PHASE_EPS) % 1.0, side="right")
    return ~swing[rows]


def example_E():
    _title("E. contact horizon (h=20, dt=0.025), shown transposed: rows = legs")
    C = contacts_horizon(0.1, 20, 0.025)
    for i, leg in enumerate(LEGS):
        print(f"{leg}: " + "".join("#" if c else "." for c in C[:, i]))
    print("  # stance, . swing")


# ==========================================================================
# F. STANDING START, CLOCK OFFSET
# ==========================================================================
"""
Stand on four legs for t_stand seconds, then start the gait:
    t_gait = t_sim - t_stand - clock_offset
    t_gait < 0  -> all legs stance (also inside the horizon: rows whose time is
                   still negative are stance, the rest come from the table)
clock_offset shifts the gait clock without touching the robot: a phase
perturbation for the time-augmented Poincare map (quad_ctrl.shift_phase).
Round shifts to whole sim steps, otherwise the contact switches fall between
sim steps.
"""


def example_F():
    _title("F. standing start")
    t_stand, h, dt = 0.5, 12, 0.025
    t_sim = 0.42
    ts = t_sim - t_stand + dt * np.arange(h)
    C = np.ones((h, 4), bool)
    started = ts > -PHASE_EPS
    if started.any():
        k0 = int(np.argmax(started))
        C[k0:] = contacts_horizon(ts[k0], h - k0, dt)
    print("FR over the horizon:", "".join("#" if c else "." for c in C[:, 1]),
          f"(gait starts at row {int(np.argmax(started))})")


# ==========================================================================
# G. FRAMES
# ==========================================================================
"""
Velocity command is given in the HEADING frame (forward / left / yaw rate):
    v_cmd_b = [vx, vy, wz]
World frame (yaw = current heading):
    v_world_xy = Rz(yaw)[:2, :2] @ [vx, vy],    wz unchanged
Heading frame vs body frame: heading = yaw only (ignores roll/pitch), which
is what you want for commands and foot placement on flat ground.
"""


def example_G():
    _title("G. heading command -> world")
    yaw = np.pi / 2
    v_cmd_b = np.array([0.5, 0.0, 0.2])
    v_w = np.array([*(rot_z(yaw)[:2, :2] @ v_cmd_b[:2]), v_cmd_b[2]])
    print("facing +y, forward 0.5 m/s ->", np.round(v_w, 3))


# ==========================================================================
# H. CONSTANT-VELOCITY REFERENCE  (MPC X_ref)
# ==========================================================================
"""
Rows 1..h of X_ref (13 states, layout of mpc_lib section A):
    yaw_k   = yaw_0 + wz * k * dt
    v_k     = Rz(yaw_k) [vx, vy, 0]                world velocity
    p_xy_k  = p_xy_0 + cumsum(v_k) * dt            (np.cumsum along the horizon)
    z = zd (nominal height), roll = pitch = 0, w = [0, 0, wz], g = -9.81
Row 0 = x0 (the measured state).
Start x, y, yaw from the MEASURED state (not the previous reference) so the
reference never drifts away from the robot.
Pitfall: keep one orientation convention. sim/ref_gen returns (13, h+1) and
the MPC asserts (h+1, 13) -> a transpose in quad_ctrl. Choose one and assert.
"""


def constant_reference(x0, v_cmd_b, h, dt, zd):
    vx, vy, wz = v_cmd_b
    k = np.arange(1, h + 1)
    yaw = x0[5] + wz * dt * k
    c, s = np.cos(yaw), np.sin(yaw)
    vxw, vyw = c * vx - s * vy, s * vx + c * vy
    X = np.zeros((h + 1, 13))
    X[0] = x0
    X[1:, 0] = x0[0] + np.cumsum(vxw) * dt
    X[1:, 1] = x0[1] + np.cumsum(vyw) * dt
    X[1:, 2] = zd
    X[1:, 5] = yaw
    X[1:, 6], X[1:, 7] = vxw, vyw
    X[1:, 11] = wz
    X[:, 12] = -9.81
    return X


def example_H():
    _title("H. constant-velocity reference, turning")
    x0 = np.zeros(13); x0[2] = 0.26
    X = constant_reference(x0, [0.5, 0.0, 1.0], h=20, dt=0.025, zd=0.27)
    print("shape", X.shape)
    print("last row  x, y, z, yaw, vx, vy:", np.round(X[-1, [0, 1, 2, 5, 6, 7]], 3))


# ==========================================================================
# I. PERIODIC REFERENCE
# ==========================================================================
"""
A reference that changes along the gait cycle: store quantities on a phase
grid phi_j in [0, 1) and look them up at the predicted phases
    phi_k = phi_0 + k dt / T                      (along the MPC horizon)
Periodic linear interpolation:
    np.interp(phi % 1.0, grid, values, period=1.0)
Building the table from a recorded steady cycle: bin samples by phase and
average:
    b = (phi % 1 * n_bins).astype(int)
    mean = np.bincount(b, weights=y, minlength=n_bins) / np.bincount(b, minlength=n_bins)
Store velocities in the HEADING frame so the table works for any yaw;
integrate x, y, yaw from them like in section H.
Lesson recorded in sim/README: tracking recorded roll/pitch makes the MPC
recreate leg-induced body motion; ("z", "v") terms only worked better.
"""


def example_I():
    _title("I. periodic table: bin a noisy signal, then interpolate")
    rng = np.random.default_rng(1)
    phase = np.sort(rng.uniform(0, 5, 2000))                 # 5 cycles, unwrapped
    z = 0.27 + 0.01 * np.sin(4 * np.pi * phase) + rng.normal(0, 0.002, phase.size)
    n_bins = 20
    b = np.minimum((np.mod(phase, 1.0) * n_bins).astype(int), n_bins - 1)
    z_bin = np.bincount(b, weights=z, minlength=n_bins) / np.bincount(b, minlength=n_bins)
    grid = (np.arange(n_bins) + 0.5) / n_bins
    phi_k = 3.9 + 0.025 * np.arange(5) / 0.5                 # wraps past 4.0
    print("z_ref at phi_k:", np.round(np.interp(phi_k % 1.0, grid, z_bin, period=1.0), 4))
    print("true          :", np.round(0.27 + 0.01 * np.sin(4 * np.pi * phi_k), 4))


# ==========================================================================
# J. FOOT PLACEMENT
# ==========================================================================
"""
Inputs (structs.FootPlacementCfg): shoulder/hip point p_hip (world), base
velocity v, command v_cmd, height h, t_stance, remaining swing time t_rem.

Hip at touchdown (the body keeps moving while the foot is in the air):
    p_hip_td = p_hip + v_cmd_xy * t_rem
Raibert (1984):
    p = p_hip_td + v * t_stance / 2 + k (v - v_cmd)
        symmetry term: foot lands so the stance is centered under the hip
        feedback term: step further in the direction you are too fast
Cheetah 3 (eq. 10 + rotation, centrifugal terms):
    p = p_hip_td + t_stance/2 v + k (v - v_cmd)
        + t_stance/2 (w_cmd x r_hip)                    yaw: hip moves on a circle
        + 1/2 sqrt(h/g) (v x w_cmd)                     centrifugal
Capture point / LIPM:  k = sqrt(h / g)   (~0.17 for h = 0.27)
Then:
    z = terrain height (foot sphere center: + radius 0.022 on flat ground)
    clamp the stride:  d = p_xy - p_hip_td_xy; if |d| > max: d *= max/|d|
Update the target every step during swing (uses the latest velocity) or only
at lift-off; sim/ makes it a flag (update_foothold_in_swing).
"""


def raibert(p_hip, v, v_cmd, t_stance, t_rem, k=0.03, max_stride=0.2, z_ground=0.022):
    p = p_hip.astype(float).copy()
    p[:2] += v_cmd[:2] * t_rem
    hip_td = p.copy()
    p[:2] += v[:2] * t_stance / 2 + k * (v[:2] - v_cmd[:2])
    d = p[:2] - hip_td[:2]
    n = np.linalg.norm(d)
    if n > max_stride:
        p[:2] = hip_td[:2] + d * max_stride / n
    p[2] = z_ground
    return p


def example_J():
    _title("J. foot placement")
    p_hip = np.array([0.19, 0.14, 0.27])
    v_cmd = np.array([0.5, 0.0, 0.0])
    for v in [np.array([0.5, 0, 0]), np.array([0.8, 0, 0]), np.array([0.5, 0.3, 0])]:
        print(f"v={v[:2]}  Raibert k=0.03:", np.round(raibert(p_hip, v, v_cmd, 0.25, 0.25), 3),
              "  k=sqrt(h/g):", np.round(raibert(p_hip, v, v_cmd, 0.25, 0.25, k=np.sqrt(0.27 / 9.81)), 3))
    r_hip, w_cmd = np.array([0.19, 0.14, 0.0]), np.array([0, 0, 1.0])
    print("yaw term t_st/2 (w x r_hip):", np.round(0.125 * np.cross(w_cmd, r_hip), 4))


# ==========================================================================
# K. SWING TRAJECTORY
# ==========================================================================
"""
Cubic Bezier with control points P0..P3, s in [0, 1]:
    B(s)   = (1-s)^3 P0 + 3(1-s)^2 s P1 + 3(1-s) s^2 P2 + s^3 P3
    B'(s)  = 3(1-s)^2 (P1-P0) + 6(1-s)s (P2-P1) + 3 s^2 (P3-P2)
    B''(s) = 6(1-s)(P2 - 2P1 + P0) + 6 s (P3 - 2P2 + P1)
Time scaling s = t / T_swing:   vel = B' / T,   acc = B'' / T^2   (chain rule)
    P1 = P0 and P2 = P3  -> zero velocity at both ends (smooth lift + land)

xy: one Bezier from lift-off point to foothold.
z : two halves (lift to apex, apex to touchdown), each s_sub = 2 s (or 2 s - 1),
    so d/dt gets an extra factor 2 (and 4 for acceleration).
    apex = max(z_lift, z_td) + step_height
    Non-zero touchdown z-velocity v_td:  P2_z = z_td - v_td / (3 * ds_sub/dt)
    (from B'(1) = 3 (P3 - P2)).
Output per leg: (3, 3) [pos; vel; acc]; acc feeds the swing feed-forward
(controller_lib section C).

Alternative: minimum-jerk quintic  p(s) = p0 + (pf-p0)(10 s^3 - 15 s^4 + 6 s^5)
(zero vel AND acc at both ends); numpy.polynomial.Polynomial(coeffs) gives
.deriv() for free. z often uses step_height * sin(pi s) for quick tests.
Always check derivatives numerically (example below).
"""


def bezier(P, s):
    P0, P1, P2, P3 = P
    pos = (1 - s) ** 3 * P0 + 3 * (1 - s) ** 2 * s * P1 + 3 * (1 - s) * s ** 2 * P2 + s ** 3 * P3
    d1 = 3 * (1 - s) ** 2 * (P1 - P0) + 6 * (1 - s) * s * (P2 - P1) + 3 * s ** 2 * (P3 - P2)
    d2 = 6 * (1 - s) * (P2 - 2 * P1 + P0) + 6 * s * (P3 - 2 * P2 + P1)
    return pos, d1, d2


def swing_z(t, T, z0, zf, height):
    """z, z_dot, z_ddot of a two-half Bezier, zero velocity at both ends."""
    s = np.clip(t / T, 0.0, 1.0)
    apex = max(z0, zf) + height
    if s <= 0.5:
        P, ss = (z0, z0, apex, apex), 2 * s
    else:
        P, ss = (apex, apex, zf, zf), 2 * s - 1
    z, d1, d2 = bezier(np.array(P, float), ss)
    k = 2.0 / T                      # d(ss)/dt
    return z, d1 * k, d2 * k ** 2


def example_K():
    _title("K. swing trajectory")
    T, h = 0.25, 0.08
    for t in np.linspace(0, T, 6):
        z, dz, ddz = swing_z(t, T, 0.022, 0.022, h)
        print(f"t={t:.3f} z={z:.4f} dz={dz:+.3f} ddz={ddz:+8.2f}")
    t, e = 0.1, 1e-6
    num = (swing_z(t + e, T, 0.022, 0.022, h)[0] - swing_z(t - e, T, 0.022, 0.022, h)[0]) / (2 * e)
    print("derivative check dz:", round(swing_z(t, T, 0.022, 0.022, h)[1], 5), "vs", round(num, 5))
    p0, pf = np.array([0.0, 0.0]), np.array([0.15, 0.02])
    xy, _, _ = bezier(np.array([p0, p0, pf, pf]), 0.5)
    s = 0.5
    print("xy at half swing, Bezier:", xy, "  min-jerk:", p0 + (pf - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5))


# ==========================================================================
# L. POLICY PATTERN
# ==========================================================================
"""
Several foot placement laws, one interface -> compare them in experiments.
    class FootPlacement:
        def __call__(self, cfg) -> np.ndarray:      common post-processing
            p = self._raw(cfg)                      (clamp stride, terrain z)
            ...
        def _raw(self, cfg): raise NotImplementedError   subclasses implement
    policy = Raibert(k=0.1);   p = policy(cfg)        instance is callable
Registry: name -> constructor, so command lines / worker processes build a
policy from plain strings:
    POLICIES = {"raibert": lambda **kw: Raibert(**kw), ...}
    POLICIES["raibert"](k=0.17)
Pitfall: a method named __call (two underscores, no trailing) is
name-mangled and does NOT make the object callable (a [FIXED] bug).
"""


class _Base:
    def __call__(self, v, v_cmd):
        return float(np.clip(self._raw(v, v_cmd), -0.2, 0.2))

    def _raw(self, v, v_cmd):
        raise NotImplementedError


class _OpenLoop(_Base):
    def _raw(self, v, v_cmd):
        return 0.125 * v_cmd


class _Raibert(_Base):
    def __init__(self, k=0.03):
        self.k = k

    def _raw(self, v, v_cmd):
        return 0.125 * v + self.k * (v - v_cmd)


_POLICIES = {"openloop": lambda **kw: _OpenLoop(**kw), "raibert": lambda **kw: _Raibert(**kw)}


def example_L():
    _title("L. policy registry (1-D stride offsets)")
    for name, kw in [("openloop", {}), ("raibert", {"k": 0.03}), ("raibert", {"k": 0.17})]:
        print(f"{name:9s}{kw}: offset at v=0.8, cmd=0.5 ->", round(_POLICIES[name](**kw)(0.8, 0.5), 4))


EXAMPLES = dict(A=example_A, B=example_B, C=example_C, D=example_D, E=example_E,
                F=example_F, G=example_G, H=example_H, I=example_I, J=example_J,
                K=example_K, L=example_L)

if __name__ == "__main__":
    for key in (sys.argv[1:] or EXAMPLES):
        EXAMPLES[key.upper()]()
