"""
CONTROLLER LIBRARY  (leg control + wiring the controller together)
==================================================================

Reference for writing
    sim/leg_controller.py   stance: forces -> torques; swing: foot tracking
    sim/quad_ctrl.py        one control step: gait -> footholds -> MPC -> legs

The examples use ONE Go2-like leg (abduction, hip, knee; thigh/calf 0.213 m)
on a fixed base (LEG_XML), so the leg Jacobian is 3x3 exactly like a Go2 leg.

    python controller_lib.py         # all examples
    python controller_lib.py C       # swing tracking only

Sections
    A  Stance: ground force -> joint torque     tau = -J^T f   (sign check)
    B  Gravity / Coriolis compensation          qfrc_bias
    C  Swing: Cartesian PD + feed-forward       J^T (Kp e + Kd e_dot) + M J^-1 a
    D  Joint friction compensation              damping, Coulomb (tanh)
    E  Limits and actuator order                np.clip, ctrl[act_idx]
    F  Multi-rate timing                        MPC every N sim steps, hold forces
    G  One control step, in order               what step() does and needs
    H  Controller state                         preallocation, copy vs view, snapshots
    I  Config dataclasses                       default_factory pitfall
    J  Debug checklist
"""
import sys
from dataclasses import dataclass, field

import mujoco
import numpy as np

LEG_XML = """
<mujoco model="one_leg">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 0.5">
      <geom type="box" size="0.1 0.05 0.03" mass="5" contype="0" conaffinity="0"/>
      <body name="hip" pos="0 0 0">
        <joint name="abd" type="hinge" axis="1 0 0" range="-0.8 0.8" damping="0.05" frictionloss="0.1"/>
        <geom type="cylinder" fromto="0 -0.03 0 0 0.03 0" size="0.04" mass="0.7" contype="0" conaffinity="0"/>
        <body name="thigh" pos="0 0.09 0">
          <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 3.4" damping="0.05" frictionloss="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.213" size="0.02" mass="1.0" contype="0" conaffinity="0"/>
          <body name="calf" pos="0 0 -0.213">
            <joint name="knee" type="hinge" axis="0 1 0" range="-2.7 -0.8" damping="0.05" frictionloss="0.1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.213" size="0.012" mass="0.2" contype="0" conaffinity="0"/>
            <geom name="foot" type="sphere" pos="0 0 -0.213" size="0.022" mass="0.04" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="knee" ctrlrange="-45 45"/>
    <motor joint="abd"  ctrlrange="-23.7 23.7"/>
    <motor joint="hip"  ctrlrange="-23.7 23.7"/>
  </actuator>
  <keyframe><key qpos="0 0.8 -1.6"/></keyframe>
</mujoco>
"""
# note: actuators listed knee, abd, hip on purpose -> section E


class Leg:
    """Small helper with the same quantities state.State gives for one leg."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(LEG_XML)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jids = [m.joint(n).id for n in ("abd", "hip", "knee")]
        self.dofs = m.jnt_dofadr[jids]
        self.qadr = m.jnt_qposadr[jids]
        act_of_joint = {m.actuator_trnid[a, 0]: a for a in range(m.nu)}
        self.act_idx = np.array([act_of_joint[j] for j in jids])
        self.tau_limit = m.actuator_ctrlrange[self.act_idx, 1]
        self.foot = m.geom("foot").id
        self.M = np.zeros((m.nv, m.nv))
        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)

    def J(self):
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, self.foot)
        return jacp[:, self.dofs]

    def p_foot(self):
        return self.data.geom_xpos[self.foot].copy()


def _title(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


# ==========================================================================
# A. STANCE: GROUND FORCE -> JOINT TORQUE
# ==========================================================================
"""
Virtual work: a force f acting ON the foot produces joint torques J^T f.
The MPC force f_i is the force the GROUND exerts on the robot. To get it,
the leg must push the ground with -f_i:
    tau_i = -J_i^T f_i            J_i = jacp[:, leg_dofs]  (3, 3), world frame
(+ bias compensation, section B, + friction compensation, section D)
J and f must be in the same frame (both world here).

Check your sign: apply the force with mujoco.mj_applyFT; the generalized
force it creates (qfrc_applied) must equal J^T f.
"""


def example_A():
    _title("A. J^T f check with mj_applyFT")
    leg = Leg()
    f = np.array([10.0, -5.0, 60.0])
    qfrc = np.zeros(leg.model.nv)
    mujoco.mj_applyFT(leg.model, leg.data, f, np.zeros(3), leg.p_foot(),
                      leg.model.geom_bodyid[leg.foot], qfrc)
    print("J^T f       :", np.round(leg.J().T @ f, 4))
    print("mj_applyFT  :", np.round(qfrc[leg.dofs], 4))
    print("stance tau = -J^T f_ground:", np.round(-leg.J().T @ f, 3))


# ==========================================================================
# B. GRAVITY / CORIOLIS COMPENSATION
# ==========================================================================
"""
data.qfrc_bias = C(q, qd) qd + g(q)  (valid after mj_forward / mj_step1)
Adding tau += qfrc_bias[leg_dofs] cancels the leg's own weight and velocity
terms, so -J^T f really delivers f (the MPC model has massless legs).
Swing legs need it too (otherwise the PD has to hold the leg up -> sag).

Pitfall: on the floating Go2, qfrc_bias of the leg dofs also contains
coupling with the base motion; it is still the right term to add.
"""


def run_leg(leg, ctrl_fn, T=1.0):
    m, d = leg.model, leg.data
    err = []
    for n in range(int(T / m.opt.timestep)):
        mujoco.mj_step1(m, d)
        tau, e = ctrl_fn(leg, d.time)
        d.ctrl[leg.act_idx] = np.clip(tau, -leg.tau_limit, leg.tau_limit)
        mujoco.mj_step2(m, d)
        err.append(e)
    return np.asarray(err)


def example_B():
    _title("B. gravity compensation only: does the leg stay put?")
    for use_bias in (False, True):
        leg = Leg()
        p0 = leg.p_foot()
        q_start = leg.data.qpos[leg.qadr].copy()

        def ctrl(leg, t):
            tau = leg.data.qfrc_bias[leg.dofs].copy() if use_bias else np.zeros(3)
            return tau, np.linalg.norm(leg.p_foot() - p0)

        e = run_leg(leg, ctrl, 0.5)
        print(f"bias comp={use_bias!s:5}: foot drift after 0.5 s = {e[-1] * 100:6.2f} cm,"
              f" joint change {np.round(leg.data.qpos[leg.qadr] - q_start, 3)}")
    print("(at rest friction does not matter; once the leg moves it does, section D)")


# ==========================================================================
# C. SWING: CARTESIAN PD + FEED-FORWARD
# ==========================================================================
"""
Desired foot [p_d; v_d; a_d] from the swing trajectory (gait_lib section K).
    v_foot = J @ qd_leg                       (or jacp @ data.qvel)
    f_pd   = Kp (p_d - p) + Kd (v_d - v)      virtual spring-damper at the foot
    tau    = J^T f_pd                         (+ bias)
Feed-forward (joint-space acceleration that produces a_d):
    a = J qdd + Jdot qd   ->   qdd_d ~= J^-1 a_d     (Jdot qd neglected)
    tau_ff = M_leg qdd_d = M_leg @ np.linalg.solve(J, a_d)
    M_leg  = M[np.ix_(dofs, dofs)],  M from mujoco.mj_fullM(model, data, M)
Pitfall: np.linalg.solve(J, a) fails near singular J (straight knee); keep the
knee bent, or use a damped pseudo-inverse  J.T @ inv(J J^T + lam I).
Gains on Go2 in sim/: Kp = 400, Kd = 10 (per axis, N/m and N s/m).
"""


def circle_ref(t, center, r=0.05, w=2 * np.pi * 2):
    c, s = np.cos(w * t), np.sin(w * t)
    p = center + r * np.array([c, 0.0, s])
    v = r * w * np.array([-s, 0.0, c])
    a = -r * w * w * np.array([c, 0.0, s])
    return p, v, a


def example_C():
    _title("C. swing tracking of a 2 Hz circle, r = 5 cm")
    kp, kd = 400.0, 10.0
    for ff in (False, True):
        leg = Leg()
        center = leg.p_foot() - np.array([0.05, 0, 0])

        def ctrl(leg, t):
            d = leg.data
            J = leg.J()
            p, v = leg.p_foot(), J @ d.qvel[leg.dofs]
            p_d, v_d, a_d = circle_ref(t, center)
            tau = J.T @ (kp * (p_d - p) + kd * (v_d - v)) + d.qfrc_bias[leg.dofs]
            if ff:
                mujoco.mj_fullM(leg.model, d, leg.M)
                M_leg = leg.M[np.ix_(leg.dofs, leg.dofs)]
                tau += M_leg @ np.linalg.solve(J, a_d)
            return tau, np.linalg.norm(p_d - p)

        e = run_leg(leg, ctrl, 1.0)
        print(f"feed-forward={ff!s:5}: RMS error (last 0.5 s) = {1000 * np.sqrt(np.mean(e[500:] ** 2)):.2f} mm")


# ==========================================================================
# D. JOINT FRICTION COMPENSATION
# ==========================================================================
"""
MuJoCo joint losses (from the xml):
    model.dof_damping[d]       viscous:  tau_loss = -b qd
    model.dof_frictionloss[d]  Coulomb:  |tau_loss| <= F,  opposes motion
Compensation added to the command:
    tau += b * qd + F * tanh(qd / eps)          eps ~ 0.05 rad/s
tanh instead of sign(qd): smooth near qd = 0, no chattering.
In sim/ this mattered most for STANCE legs: uncompensated friction ate ~4 N
of the MPC's horizontal force per leg.
"""


def example_D():
    _title("D. friction compensation in swing tracking")
    kp, kd, eps = 400.0, 10.0, 0.05
    for comp in (False, True):
        leg = Leg()
        b = leg.model.dof_damping[leg.dofs]
        F = leg.model.dof_frictionloss[leg.dofs]
        center = leg.p_foot() - np.array([0.05, 0, 0])

        def ctrl(leg, t):
            d = leg.data
            J = leg.J()
            qd = d.qvel[leg.dofs]
            p_d, v_d, a_d = circle_ref(t, center)
            mujoco.mj_fullM(leg.model, d, leg.M)
            tau = (J.T @ (kp * (p_d - leg.p_foot()) + kd * (v_d - J @ qd)) + d.qfrc_bias[leg.dofs]
                   + leg.M[np.ix_(leg.dofs, leg.dofs)] @ np.linalg.solve(J, a_d))
            if comp:
                tau += b * qd + F * np.tanh(qd / eps)
            return tau, np.linalg.norm(p_d - leg.p_foot())

        e = run_leg(leg, ctrl, 1.0)
        print(f"friction comp={comp!s:5}: RMS error = {1000 * np.sqrt(np.mean(e[500:] ** 2)):.2f} mm")


# ==========================================================================
# E. LIMITS AND ACTUATOR ORDER
# ==========================================================================
"""
Build once:   act_idx[leg, joint] = actuator index   (via actuator_trnid)
              tau_limit = model.actuator_ctrlrange[act_idx, 1]
Every step:   tau = np.clip(tau, -tau_limit, tau_limit)
              data.ctrl[act_idx.reshape(-1)] = tau.reshape(-1)
Pitfall: data.ctrl[:] = tau.reshape(-1) assumes the xml order equals yours.
Go2 xml order is FR, FL, RR, RL, a common [LEGS] order is FL, FR, RL, RR.
Clipping silently: log how often torques saturate (a sign the MPC asks too much).
"""


def example_E():
    _title("E. actuator order")
    leg = Leg()
    tau = np.array([1.0, 2.0, 3.0])                    # [abd, hip, knee]
    leg.data.ctrl[:] = tau
    print("WRONG ctrl[:] = tau      -> knee motor gets", leg.data.ctrl[leg.act_idx[2]])
    leg.data.ctrl[leg.act_idx] = tau
    print("RIGHT ctrl[act_idx] = tau -> knee motor gets", leg.data.ctrl[leg.act_idx[2]],
          " (act_idx =", leg.act_idx, ")")
    print("clip to limits:", np.clip(np.array([50.0, -30, 10]), -leg.tau_limit, leg.tau_limit))


# ==========================================================================
# F. MULTI-RATE TIMING
# ==========================================================================
"""
sim dt = 1 ms, MPC dt = 25 ms -> solve every N = 25 steps, leg control every step.
    N = round(dt_mpc / dt_sim); assert N * dt_sim == dt_mpc (within 1e-9)
Trigger on the GAIT clock, not a raw counter:
    tick = floor(t_gait / dt_mpc + 1e-6)
    if tick != last_tick: solve; last_tick = tick
Then a gait phase shift moves the MPC sampling with the contact schedule.
Between solves: keep forces[0] of the last solution (zero-order hold).
Pitfall: `t % N == 0` with a FLOAT time t is (almost) never true; use integers
or the tick idea above ([FIXED] bug in quad_ctrl).
"""


def example_F():
    _title("F. MPC trigger")
    dt, dt_mpc, t_stand = 0.001, 0.025, 0.5
    last, solves = None, []
    for n in range(800):
        t_gait = n * dt - t_stand
        tick = int(np.floor(t_gait / dt_mpc + 1e-6))
        if tick != last:
            last = tick
            solves.append(n)
    print("solve at sim steps:", solves[:6], "...  spacing", np.unique(np.diff(solves)))
    print("float modulo check `t % 0.025 == 0` true for",
          sum(((n * dt) % 0.025) == 0 for n in range(800)), "of 32 expected solves")


# ==========================================================================
# G. ONE CONTROL STEP, IN ORDER   (what QuadrupedController.step does)
# ==========================================================================
"""
called between mj_step1 and mj_step2, input v_cmd_b = [vx, vy, wz]

 1. read state            base pose, v, w, foot positions, Jacobians     (sim_interface_lib F, G)
 2. command to world      v_cmd_w = Rz(yaw) v_cmd_b                       (gait_lib G)
 3. contacts now          t_gait = t - t_stand - offset -> table          (gait_lib B, F)
 4. events                liftoff = ~c & c_prev -> store lift-off point   (gait_lib D)
 5. swing timing          elapsed, remaining swing time                   (gait_lib C)
 6. footholds             stance leg: foothold = foot position
                          swing leg : policy(FootPlacementCfg)             (gait_lib J, L)
 7. MPC (if new tick)     x0 (base or centroidal state), X_ref,           (gait_lib H/I, E)
                          contacts horizon, predicted r_feet,
                          solve -> keep forces[0]                          (mpc_lib)
 8. swing references      Bezier from lift-off point to foothold          (gait_lib K)
 9. leg torques           stance: -J^T f + bias + friction                (this file A-D)
                          swing : J^T PD + M J^-1 a + bias + friction
10. clip + write ctrl                                                     (E)
11. log                   contacts, forces, footholds, tau, phase, status (H)
12. n_step += 1

Things that need memory between steps (= controller state, section H):
    c_prev, lift-off points, footholds, held forces, last MPC solution,
    last MPC tick, clock offset, step counter.
"""


# ==========================================================================
# H. CONTROLLER STATE: PREALLOCATION, COPY VS VIEW, SNAPSHOTS
# ==========================================================================
"""
a = b            same object (a view): changing b changes a
a = b.copy()     new array
np.copyto(dst, src)    write INTO an existing array (dst keeps its identity)
x = sol.forces[0]      a VIEW into sol.forces -> changes when sol is reused

Snapshot for Poincare / ROA: everything from section G "needs memory"
    def get_state(self, d): np.copyto(d["foothold"], self.foothold); d["step"] = self.n_step; ...
    def set_state(self, d): np.copyto(self.foothold, d["foothold"]); ...
Pair it with the simulator snapshot (sim_interface_lib I).
Pitfall: an attribute named like a method (self.step = 0 next to def step)
hides the method -> "'int' object is not callable" ([FIXED] bug).
"""


def example_H():
    _title("H. copy vs view")
    forces = np.zeros((20, 4, 3))
    held_view = forces[0]
    held_copy = forces[0].copy()
    forces[0, 0, 2] = 75.0          # next MPC result written into the same buffer
    print("view sees the new value:", held_view[0, 2], "  copy keeps the old one:", held_copy[0, 2])
    snap = {"foothold": np.empty((4, 3))}
    foothold = np.ones((4, 3))
    np.copyto(snap["foothold"], foothold)
    foothold[:] = 0
    print("snapshot after the controller moved on:", snap["foothold"][0])


# ==========================================================================
# I. CONFIG DATACLASSES
# ==========================================================================
"""
@dataclass
class LegCfg:
    kp: np.ndarray = field(default_factory=lambda: 400.0 * np.ones(3))   # mutable -> factory
    mu: float = 0.28                                                       # immutable -> plain default
Pitfalls (all [FIXED] in sim/):
    field(default=GaitCfg)        default is the CLASS, not an instance -> default_factory=GaitCfg
    kp shape (1, 3) vs (3,)       broadcasting surprises later
    @dataclass(slots=True)        reading an undeclared attribute raises AttributeError (good!)
dataclasses.asdict(cfg) -> dict -> json for saving experiment settings.
"""


@dataclass
class _GaitCfg:
    period: float = 0.5


@dataclass
class _CtrlCfg:
    gait: _GaitCfg = field(default_factory=_GaitCfg)
    kp: np.ndarray = field(default_factory=lambda: 400.0 * np.ones(3))


def example_I():
    _title("I. dataclass defaults")
    a, b = _CtrlCfg(), _CtrlCfg()
    a.kp[0] = 1.0
    print("independent defaults:", b.kp[0] == 400.0, " gait is an instance:", isinstance(a.gait, _GaitCfg))


# ==========================================================================
# J. DEBUG CHECKLIST
# ==========================================================================
"""
* Stand first (t_stand, all stance, MPC on): base should hold zd with forces
  ~ m g / 4 each. If not, fix A/B/E before touching the gait.
* Then trot in place (v_cmd = 0), then forward.
* Log and plot: base z, roll, pitch, v vs command; MPC status per solve;
  planned vs actual contact; fz per leg (zero in swing?); torque saturation.
* Compare X_pred of the MPC with what happened next (model mismatch).
* Assert shapes at every interface: (4, 3) forces, (4, 3, 3) swing refs,
  (h, 4) contacts, (h+1, 13) X_ref, (4, 6, nv) Jacobians.
* One change at a time; keep a config dump with every result (json).
"""


EXAMPLES = dict(A=example_A, B=example_B, C=example_C, D=example_D, E=example_E,
                F=example_F, H=example_H, I=example_I)

if __name__ == "__main__":
    for key in (sys.argv[1:] or EXAMPLES):
        EXAMPLES[key.upper()]()
