"""
SIMULATOR INTERFACE LIBRARY  (MuJoCo python bindings)
=====================================================

Reference for writing   sim/sim_env.py   and   sim/state.py   yourself.

Every entry has:  what it is  ->  shapes / units  ->  pitfalls  ->  where the
sim/ code needs it.  The examples run on a small one-legged toy robot
(TOY_XML below), so you can try everything without the Go2 model:

    python sim_interface_lib.py          # runs every example in order
    python sim_interface_lib.py G        # only section G

Sections
    A  Build a model             MjModel / MjSpec / MjData
    B  Find things by name       ids, qpos/qvel addresses, actuators
    C  Simulator options         timestep, gravity, integrator, cone ...
    D  Contact + robot params    solref, solimp, friction, mass, torque limits
    E  The raw state             qpos / qvel layout of a floating base, ctrl
    F  Computed quantities       positions, rotations, CoM, bias forces
    G  Jacobians, mass matrix    mj_jacGeom, mj_fullM
    H  Stepping the simulation   mj_step vs mj_step1 / mj_step2, reset
    I  Snapshots                 mj_getState / mj_setState
    J  Perturbations             xfrc_applied, velocity kicks
    K  Rotations                 quaternion <-> rpy, heading yaw
    L  Viewer + logging          launch_passive, np.savez
"""
import sys

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# Toy robot: floating box base + one 2-joint leg (hip, knee) + sphere foot.
# Same structure as one Go2 leg: free joint, hinge joints, motors, a foot geom.
# --------------------------------------------------------------------------
TOY_XML = """
<mujoco model="toy_leg">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" friction="0.8"/>
    <body name="base" pos="0 0 0.45">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.15 0.08 0.04" mass="3"/>
      <body name="thigh" pos="0 0 -0.04">
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02" mass="0.5"/>
        <body name="calf" pos="0 0 -0.2">
          <joint name="knee" type="hinge" axis="0 1 0" range="-2.5 0" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.015" mass="0.2"/>
          <geom name="foot" type="sphere" pos="0 0 -0.2" size="0.022" mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_motor"  joint="hip"  ctrlrange="-20 20"/>
    <motor name="knee_motor" joint="knee" ctrlrange="-20 20"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.45  1 0 0 0  0.6 -1.2"/>
  </keyframe>
</mujoco>
"""


def toy():
    """(model, data) of the toy robot, at the home keyframe, kinematics computed."""
    model = mujoco.MjModel.from_xml_string(TOY_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def _title(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


# ==========================================================================
# A. BUILD A MODEL
# ==========================================================================
"""
mujoco.MjModel.from_xml_path(path)      -> MjModel   (constant: geometry, masses, options)
mujoco.MjModel.from_xml_string(xml)     -> MjModel
mujoco.MjData(model)                    -> MjData    (changing: state, ctrl, results)

    MjModel = "the robot description", MjData = "the robot right now".
    Many MjData can share one MjModel (e.g. a scratch MjData to compute things
    at the keyframe without touching the running simulation -> robot_params()).

mujoco.MjSpec.from_file(path)  /  MjSpec.from_string(xml)   -> MjSpec
    Editable model BEFORE compiling. Use it to add/remove elements.
        spec.worldbody.geoms            list of geoms attached to the world
        spec.delete(element)            remove a geom / body
        spec.compile()                  -> MjModel
    Where: sim_env.make_model() removes the stairs/bars of the stock scene.

    Pitfall: joint `range` in the xml is in DEGREES unless the model has
    <compiler angle="radian"/> (the Go2 xml sets it). Check model.jnt_range.
    Pitfall: numeric params (friction, solref, timestep ...) can simply be
    edited on the compiled MjModel (section C/D). MjSpec is only needed for
    structural changes (adding/removing elements).
"""


def example_A():
    _title("A. build a model")
    spec = mujoco.MjSpec.from_string(TOY_XML)
    print("world geoms before:", [g.name for g in spec.worldbody.geoms])
    # delete every world geom except the floor (what make_model does)
    for g in [g for g in spec.worldbody.geoms if g.name != "floor"]:
        spec.delete(g)
    model = spec.compile()
    data = mujoco.MjData(model)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
    print("  nq = 7 (free joint) + 2 hinges,  nv = 6 (free joint) + 2 hinges")


# ==========================================================================
# B. FIND THINGS BY NAME
# ==========================================================================
"""
Named accessors (return a view object with .id and the element's fields):
    model.body("base").id          model.geom("foot").id
    model.joint("hip").id          model.actuator("hip_motor").id
    model.joint("hip").bodyid      body the joint belongs to (array of 1)
    data.body("base").xpos         same accessor on data -> computed values

Index arrays (all indexed by element id):
    model.jnt_qposadr[jid]   first index of that joint in qpos
    model.jnt_dofadr[jid]    first index of that joint in qvel / qfrc_* / Jacobian columns
    model.actuator_trnid[aid, 0]   joint id driven by actuator aid
    model.body_parentid[bid]       parent body

    Pitfall 1: qpos and qvel indices DIFFER for a floating base (quaternion
    has 4 numbers, angular velocity 3). Always use jnt_qposadr for qpos and
    jnt_dofadr for qvel.
    Pitfall 2: actuator order in the xml is not necessarily your leg order.
    Go2's xml lists actuators FR, FL, RR, RL. Build a map joint -> actuator
    with actuator_trnid instead of assuming (state.State.act_idx).

Where: state.State.__init__ builds qadr / vadr / act_idx / leg_dofs once.
"""


def example_B():
    _title("B. names -> ids -> addresses")
    model, data = toy()
    names = ["hip", "knee"]
    jids = [model.joint(n).id for n in names]
    qadr = model.jnt_qposadr[jids]
    vadr = model.jnt_dofadr[jids]
    print("joint ids", jids, " qpos addr", qadr, " qvel addr", vadr)
    print("joint angles from qpos:", data.qpos[qadr])

    # joint id -> actuator id, then reorder actuators to MY joint order
    act_of_joint = {model.actuator_trnid[a, 0]: a for a in range(model.nu)}
    act_idx = np.array([act_of_joint[j] for j in jids])
    print("actuator index for [hip, knee]:", act_idx)
    print("torque limits:", model.actuator_ctrlrange[act_idx])


# ==========================================================================
# C. SIMULATOR OPTIONS   (model.opt, edit in place, before stepping)
# ==========================================================================
"""
model.opt.timestep            [s]
model.opt.gravity[:] = [0, 0, -9.81]        (assign into the array, keep [:])
model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER | mjINT_IMPLICITFAST | mjINT_IMPLICIT | mjINT_RK4
model.opt.cone       = mujoco.mjtCone.mjCONE_PYRAMIDAL  | mjCONE_ELLIPTIC
model.opt.impratio            ratio of frictional to normal impedance (>1 = less slip, elliptic cone)
model.opt.iterations          constraint solver iterations
model.opt.noslip_iterations   extra no-slip post-processing (0 = off)

Enum <-> name:   mujoco.mjtIntegrator(model.opt.integrator).name

Where: sim_env.apply_sim_cfg(), driven by structs.SimCfg (None = keep xml value).
"""


def example_C():
    _title("C. options")
    model, _ = toy()
    model.opt.timestep = 0.002
    model.opt.gravity[:] = [0.0, 0.0, -9.81]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    print("integrator:", mujoco.mjtIntegrator(model.opt.integrator).name,
          " cone:", mujoco.mjtCone(model.opt.cone).name, " dt:", model.opt.timestep)


# ==========================================================================
# D. CONTACT + ROBOT PARAMETERS   (per geom / body / actuator arrays)
# ==========================================================================
"""
model.geom_solref[gid] = (timeconst, dampratio)   "standard": spring-damper with that
                                                  time constant, keep timeconst >= 2*dt
                       = (-stiffness, -damping)   "direct": both NEGATIVE numbers
model.geom_solimp[gid] = (dmin, dmax, width, midpoint, power)   how hard vs penetration
model.geom_friction[gid] = (sliding, torsional, rolling)        [:, 0] is the usual mu
model.geom_condim[gid]   1 frictionless, 3 sliding friction, 4 +torsional, 6 +rolling
model.geom_priority[gid] higher priority geom's params win in a contact pair;
                         equal priority -> friction = max of the two geoms

model.body_mass[bid], model.body_inertia[bid] (3, principal moments)
model.body_subtreemass[bid]       mass of the body + all children (base -> whole robot)
model.actuator_ctrlrange[aid]     (lo, hi) torque limits for motors
model.dof_damping[dof], model.dof_frictionloss[dof]   joint viscous / Coulomb friction

Fancy indexing writes many geoms at once:  model.geom_solref[[g1, g2, g3]] = (0.02, 1)

Pitfall: friction pyramid vs cone. MuJoCo default cone is pyramidal, the
Go2 sim may use elliptic. A friction PYRAMID |fx|,|fy| <= mu fz inside a CONE
of mu_c has mu = mu_c / sqrt(2)   (that is why MPC_Cfg.mu = 0.28 for 0.4).

Where: sim_env.apply_sim_cfg(), sim_env.foot_friction(), leg_controller (dof_damping).
"""


def example_D():
    _title("D. contact / robot params")
    model, _ = toy()
    gids = [model.geom("floor").id, model.geom("foot").id]
    model.geom_solref[gids] = [0.02, 1.0]            # standard mode
    model.geom_friction[gids, 0] = 0.6
    print("solref:", model.geom_solref[gids].tolist())
    print("friction:", model.geom_friction[gids, 0])
    print("whole robot mass (subtree of base):",
          model.body_subtreemass[model.body("base").id])
    dofs = model.jnt_dofadr[[model.joint("hip").id, model.joint("knee").id]]
    print("joint damping:", model.dof_damping[dofs], "frictionloss:", model.dof_frictionloss[dofs])


# ==========================================================================
# E. THE RAW STATE
# ==========================================================================
"""
data.time
data.qpos (nq,)   free joint first:  [x, y, z,  qw, qx, qy, qz,  joints...]
data.qvel (nv,)   free joint first:  [vx, vy, vz (WORLD frame),
                                      wx, wy, wz (BODY frame!),  joint rates...]
data.ctrl (nu,)   actuator commands (torques for <motor>), in ACTUATOR order
data.act, data.qacc_warmstart ...  (internal; handled by snapshots, section I)

Pitfall: angular velocity in qvel is in the BODY frame. World frame:
    w_world = R_wb @ qvel[3:6]       with R_wb = data.xmat[base].reshape(3, 3)
Pitfall: quaternion in qpos is scalar-FIRST (w, x, y, z); scipy is scalar-last.

Where: state.State.base_lin_vel_world / base_ang_vel_world / joint_positions.
"""


def example_E():
    _title("E. raw state")
    model, data = toy()
    data.qvel[3:6] = [0.0, 0.0, 1.0]           # spin about body z
    data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]   # roll 90 deg
    mujoco.mj_forward(model, data)
    R_wb = data.xmat[model.body("base").id].reshape(3, 3)
    print("body-frame w:", data.qvel[3:6], "-> world-frame w:", np.round(R_wb @ data.qvel[3:6], 3))
    print("  (rolled 90 deg about x, so body z points along world -y)")


# ==========================================================================
# F. COMPUTED QUANTITIES   (valid only after mj_forward / mj_step1)
# ==========================================================================
"""
mujoco.mj_forward(model, data)   compute everything from qpos/qvel (no time step)

Per body (index by body id):
    data.xpos[b]  (3,)    body frame origin, world
    data.xmat[b]  (9,)    body rotation, row-major -> .reshape(3, 3) = R_world_body
    data.xquat[b] (4,)
    data.xipos[b], data.ximat[b]   position / rotation of the body's INERTIAL frame (its CoM)
    data.subtree_com[b]   CoM of body + children, world  (base -> whole robot CoM)
Per geom:
    data.geom_xpos[g] (3,), data.geom_xmat[g] (9,)   -> foot sphere centers
Per dof (index by dofadr):
    data.qfrc_bias       (nv,)  gravity + Coriolis/centrifugal  = C(q,qd) qd + g(q)
    data.qfrc_actuator   (nv,)  joint torque produced by the actuators
    data.qfrc_constraint (nv,)  from contacts / limits

mujoco.mj_subtreeVel(model, data)   NOT done by mj_forward; fills
    data.subtree_linvel[b]   CoM velocity of the subtree
    data.subtree_angmom[b]   angular momentum about the subtree CoM
    -> centroidal MPC state (state.State.mpc_state_centroidal)

Contacts:  data.ncon, data.contact[i].geom1 / geom2 / pos / frame
           mujoco.mj_contactForce(model, data, i, f6)   force in the contact frame

Pitfall: these arrays are VIEWS into MjData. Keep a value with .copy(),
otherwise it changes at the next step (state.State returns .copy()).
Pitfall: after setting qpos by hand call mj_forward before reading xpos etc.

Where: state.py (all properties), state.robot_params() (composite inertia).
"""


def example_F():
    _title("F. computed quantities")
    model, data = toy()
    b = model.body("base").id
    print("base pos:", data.xpos[b], " foot pos:", np.round(data.geom_xpos[model.geom("foot").id], 3))
    print("whole-robot CoM:", np.round(data.subtree_com[b], 4))
    vadr = model.jnt_dofadr[[model.joint("hip").id, model.joint("knee").id]]
    print("gravity torque on [hip, knee] (qfrc_bias):", np.round(data.qfrc_bias[vadr], 3))

    # composite inertia of all bodies about the total CoM (parallel axis theorem),
    # the idea used in state.robot_params()
    com = data.subtree_com[b]
    I_w = np.zeros((3, 3))
    for k in range(1, model.nbody):
        Rk = data.ximat[k].reshape(3, 3)
        I_w += Rk @ np.diag(model.body_inertia[k]) @ Rk.T
        r = data.xipos[k] - com
        I_w += model.body_mass[k] * (r @ r * np.eye(3) - np.outer(r, r))
    print("composite inertia about CoM (world):\n", np.round(I_w, 5))

    data.qvel[0] = 0.5
    mujoco.mj_forward(model, data)
    mujoco.mj_subtreeVel(model, data)
    print("CoM velocity:", np.round(data.subtree_linvel[b], 3),
          " angular momentum:", np.round(data.subtree_angmom[b], 4))


# ==========================================================================
# G. JACOBIANS AND MASS MATRIX
# ==========================================================================
"""
mujoco.mj_jacGeom(model, data, jacp, jacr, geom_id)   point = geom center
mujoco.mj_jacBody(model, data, jacp, jacr, body_id)   point = body origin
mujoco.mj_jac(model, data, jacp, jacr, point, body_id) any world point on body
    jacp (3, nv): linear velocity  v_point = jacp @ qvel   (world frame)
    jacr (3, nv): angular velocity of the body             (world frame)
    You allocate jacp/jacr (np.zeros((3, nv))); MuJoCo fills them.

Leg block: columns of that leg's 3 dofs -> J_leg = jacp[:, leg_dofs]  (3, 3)
    joint torque from a foot force   tau = J_leg.T @ f

mujoco.mj_fullM(model, data, M)   dense mass matrix M (nv, nv)
    (MuJoCo stores M sparsely; mj_fullM unpacks it. Older versions:
     mj_fullM(model, M, data.qM))
    Leg block:  M[np.ix_(dofs, dofs)]          (np.ix_ = rows AND cols)

Pitfall: J[i][:, dofs] on a (6, nv) Jacobian keeps the 3 angular rows too ->
(6, 3). Take [:3] first (one of the [FIXED] bugs in leg_controller.py).

Where: state.State.foot_jacobians, leg_controller.LegController.torques.
"""


def example_G():
    _title("G. Jacobians / mass matrix")
    model, data = toy()
    nv = model.nv
    jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
    g = model.geom("foot").id
    mujoco.mj_jacGeom(model, data, jacp, jacr, g)
    dofs = model.jnt_dofadr[[model.joint("hip").id, model.joint("knee").id]]
    J_leg = jacp[:, dofs]
    print("foot Jacobian, leg columns (3x2):\n", np.round(J_leg, 3))

    # check: v_foot = J qvel vs finite difference
    data.qvel[dofs] = [0.3, -0.5]
    v_J = jacp @ data.qvel
    p0 = data.geom_xpos[g].copy()
    eps = 1e-6
    data.qpos[model.jnt_qposadr[[model.joint("hip").id, model.joint("knee").id]]] += eps * data.qvel[dofs]
    mujoco.mj_forward(model, data)
    print("v from J:", np.round(v_J, 4), "  finite diff:", np.round((data.geom_xpos[g] - p0) / eps, 4))

    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, M)
    print("leg mass matrix block:\n", np.round(M[np.ix_(dofs, dofs)], 5))


# ==========================================================================
# H. STEPPING THE SIMULATION
# ==========================================================================
"""
mujoco.mj_step(model, data)      one full step: forward + integrate.
                                 ctrl must be set BEFORE the call.
mujoco.mj_step1(model, data)     first half: position/velocity dependent
                                 quantities (kinematics, Jacobians, bias).
   ... read state, compute ctrl ...
mujoco.mj_step2(model, data)     second half: uses ctrl, computes accelerations,
                                 integrates.
    Why split: with mj_step, xpos/Jacobians you read belong to the state
    BEFORE the last integration (one step old). With step1/step2 the
    controller sees kinematics consistent with the current qpos/qvel.
    (RK4 integrator does not support the split; use Euler/implicit.)

mujoco.mj_resetData(model, data)                 everything to model defaults
mujoco.mj_resetDataKeyframe(model, data, key)    load <keyframe> number key
Loop pattern used in sim_env.run():
    for n in range(n_steps):
        apply pushes -> mj_step1 -> ctrl.step(cmd) (writes data.ctrl) -> mj_step2
        log, check fall, viewer sync

Where: sim_env.run(), sim_env.reset_robot().
"""


def example_H():
    _title("H. stepping (drop the toy robot with joint PD)")
    model, data = toy()
    qadr = model.jnt_qposadr[[model.joint("hip").id, model.joint("knee").id]]
    vadr = model.jnt_dofadr[[model.joint("hip").id, model.joint("knee").id]]
    q_des = data.qpos[qadr].copy()
    for n in range(500):
        mujoco.mj_step1(model, data)
        data.ctrl[:] = 40 * (q_des - data.qpos[qadr]) - 1.0 * data.qvel[vadr]
        mujoco.mj_step2(model, data)
    print(f"t={data.time:.3f}  base z={data.qpos[2]:.3f}  contacts={data.ncon}")


# ==========================================================================
# I. SNAPSHOTS
# ==========================================================================
"""
spec = mujoco.mjtState.mjSTATE_INTEGRATION     everything needed to continue exactly
                                               (time, qpos, qvel, act, warmstart, ...)
n = mujoco.mj_stateSize(model, spec)
mujoco.mj_getState(model, data, s, spec)       copy into s (n,)
mujoco.mj_setState(model, data, s, spec)       copy back; then mj_forward

Pitfall: restoring the simulator alone is not enough. Your controller has
internal state too (gait clock, lift-off points, last MPC solution) -> save
it as well (QuadrupedController.get_state / set_state). Poincare and ROA
analyses rely on both.

Where: sim_env.get_sim_state / set_sim_state / snapshot / restore.
"""


def example_I():
    _title("I. snapshots")
    model, data = toy()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    s = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, s, spec)
    for _ in range(200):
        mujoco.mj_step(model, data)
    z_a = data.qpos[2]
    mujoco.mj_setState(model, data, s, spec)
    mujoco.mj_forward(model, data)
    for _ in range(200):
        mujoco.mj_step(model, data)
    print("same result after restore:", np.isclose(z_a, data.qpos[2]), f"(z={z_a:.5f})")


# ==========================================================================
# J. PERTURBATIONS
# ==========================================================================
"""
data.xfrc_applied[b] (6,)  [force(3), torque(3)] world frame, applied at the
                           body's CoM every step until you set it back to 0.
                           -> timed pushes (sim_env.ForcePush)
Velocity kick: add to data.qvel directly, then mj_forward.
    linear part is world frame:     qvel[0:3] += dv_world
    angular part is BODY frame:     qvel[3:6] += R_wb.T @ dw_world
mujoco.mj_applyFT(model, data, force, torque, point, body, data.qfrc_applied)
                           force at an arbitrary point (not CoM)

Where: sim_env.push_velocity, sim_env.ForcePush, poincare_aug / roa perturbations.
"""


def example_J():
    _title("J. perturbations")
    model, data = toy()
    b = model.body("base").id
    data.xfrc_applied[b, :3] = [30.0, 0.0, 0.0]
    for _ in range(50):
        mujoco.mj_step(model, data)
    data.xfrc_applied[b] = 0.0
    print("vx after a 30 N x 0.05 s push:", round(data.qvel[0], 3))


# ==========================================================================
# K. ROTATIONS
# ==========================================================================
"""
Quaternion in qpos: [w, x, y, z]  (scalar first)

scipy:  from scipy.spatial.transform import Rotation as R
    R.from_quat(q[[1, 2, 3, 0]])      scipy wants [x, y, z, w]  (scalar last)
     .as_euler("xyz")                 extrinsic xyz = [roll, pitch, yaw],
                                      R = Rz(yaw) Ry(pitch) Rx(roll)
     .as_matrix()
mujoco: mujoco.mju_quat2Mat(m9, q) / mju_mat2Quat(q, m9) / mju_mulQuat ...

Heading yaw straight from the quaternion (no scipy):
    yaw = atan2(2 (w z + x y), 1 - 2 (y^2 + z^2))
"Up" component (body z axis . world z), cheap fall check:
    up_z = 1 - 2 (x^2 + y^2)

Rz(yaw) maps HEADING-frame vectors (commands, "forward") to world.

Where: state.State.base_pose, sim_env.heading_yaw, sim_env.is_fallen, structs.rot_z.
"""


def example_K():
    _title("K. rotations")
    from scipy.spatial.transform import Rotation as R
    q_wxyz = R.from_euler("xyz", [0.1, -0.2, 0.7]).as_quat()[[3, 0, 1, 2]]
    rpy = R.from_quat(q_wxyz[[1, 2, 3, 0]]).as_euler("xyz")
    w, x, y, z = q_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    m9 = np.zeros(9)
    mujoco.mju_quat2Mat(m9, q_wxyz)
    print("rpy:", np.round(rpy, 3), " yaw formula:", round(yaw, 3),
          " up_z:", round(1 - 2 * (x * x + y * y), 3))
    print("mujoco and scipy matrices equal:",
          np.allclose(m9.reshape(3, 3), R.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()))


# ==========================================================================
# L. VIEWER + LOGGING
# ==========================================================================
"""
from mujoco import viewer
v = viewer.launch_passive(model, data)    window that YOU drive (your loop keeps control)
v.is_running()                            False after the user closes the window
v.sync()                                  push the latest data to the window (every ~10 steps)
v.close()
    Real-time pacing: sleep(dt - elapsed) using time.perf_counter().
    macOS: launch_passive needs the `mjpython` launcher instead of `python`.
viewer.launch(model, data)                blocking, MuJoCo's own loop (no controller)

Logging: append copies into python lists each n steps, then
    np.savez(path, t=np.asarray(t), qpos=np.asarray(qpos), ...)
    d = np.load(path); d["qpos"]; d.files
    Pitfall: append data.qpos.copy(), not data.qpos (all entries would be the same view).

Where: sim_env.run (viewer), sim_env.SimLog.
"""


def example_L():
    _title("L. logging")
    import os
    import tempfile
    model, data = toy()
    log_t, log_z = [], []
    for n in range(300):
        mujoco.mj_step(model, data)
        if n % 50 == 0:
            log_t.append(data.time)
            log_z.append(data.qpos[2].copy())
    path = os.path.join(tempfile.gettempdir(), "toy_log.npz")
    np.savez(path, t=np.asarray(log_t), z=np.asarray(log_z))
    d = np.load(path)
    print("saved keys:", d.files, " z:", np.round(d["z"], 3))


EXAMPLES = dict(A=example_A, B=example_B, C=example_C, D=example_D, E=example_E,
                F=example_F, G=example_G, H=example_H, I=example_I, J=example_J,
                K=example_K, L=example_L)

if __name__ == "__main__":
    for key in (sys.argv[1:] or EXAMPLES):
        EXAMPLES[key.upper()]()
