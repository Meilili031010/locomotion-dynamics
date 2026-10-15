import numpy as np
import mujoco

GO2_XML = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/go2.xml"

spec = mujoco.MjSpec.from_file(GO2_XML)

floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [0.0, 0.0, 0.05]
floor.friction = [1.0, 0.005, 0.0]
floor.solref = [0.02, 1.0]

m = spec.compile()
d = mujoco.MjData(m)
HOME_QPOS = np.array(
    [0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0]
    + [0.0, 0.9, -1.8] * 4
)
d.qpos[:] = HOME_QPOS
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)
OUT_TXT = "go2_model_info.txt"
OUT_NPZ = "go2_nominal.npz"


def name(objtype, i):
    n = mujoco.mj_id2name(m, objtype, i)
    return n if n else f"<unnamed_{i}>"

JNT_TYPE = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
TRN_TYPE = {0: "joint", 1: "jointinparent", 2: "slidercrank", 3: "tendon", 4: "site"}
lines = []
def P(*a):
    lines.append(" ".join(str(x) for x in a))

# ---------- 1. Simulator settings ----------
P("=== Simulator settings ===")
P("timestep     :", m.opt.timestep)
P("integrator   :", m.opt.integrator)
P("gravity      :", m.opt.gravity)
P("cone         :", m.opt.cone, "(0=pyramidal, 1=elliptic)")
P("impratio     :", m.opt.impratio)
P("iterations   :", m.opt.iterations)
P(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} nsite={m.nsite}")

# ---------- 2. Rigid body model parameters ----------
P("\n=== Rigid body model parameters ===")
for b in range(m.nbody):
    parent = name(mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])
    P(f"[{b:2d}] {name(mujoco.mjtObj.mjOBJ_BODY, b):<18s} parent={parent:<16s} "
      f"mass={m.body_mass[b]:.4f} ipos={np.round(m.body_ipos[b],5)} "
      f"iquat={np.round(m.body_iquat[b],5)} diag_inertia={np.round(m.body_inertia[b],6)}")

# ---------- 3. Joints ----------
P("\n=== Joints ===")
for j in range(m.njnt):
    jtype = JNT_TYPE.get(m.jnt_type[j], m.jnt_type[j])
    dadr = m.jnt_dofadr[j]
    nd = {0: 6, 1: 3, 2: 1, 3: 1}[m.jnt_type[j]]
    P(f"[{j:2d}] {name(mujoco.mjtObj.mjOBJ_JOINT, j):<18s} type={jtype:<6s} "
      f"body={name(mujoco.mjtObj.mjOBJ_BODY, m.jnt_bodyid[j]):<16s} "
      f"axis={np.round(m.jnt_axis[j],3)} range={np.round(m.jnt_range[j],3)} "
      f"qposadr={m.jnt_qposadr[j]} dofadr={dadr}")
    for k in range(nd):
        P(f"       dof{dadr+k:2d}: damping={m.dof_damping[dadr+k]:.4f} "
          f"armature={m.dof_armature[dadr+k]:.5f} frictionloss={m.dof_frictionloss[dadr+k]:.4f}")

# ---------- 4. Actuators ----------
P("\n=== Actuators ===")
for a in range(m.nu):
    jid = m.actuator_trnid[a, 0]
    P(f"[{a:2d}] {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a):<14s} "
      f"trn={TRN_TYPE.get(m.actuator_trntype[a], m.actuator_trntype[a])} "
      f"-> {name(mujoco.mjtObj.mjOBJ_JOINT, jid):<18s} "
      f"gear={np.round(m.actuator_gear[a,:1],3)} "
      f"ctrlrange={np.round(m.actuator_ctrlrange[a],2)} "
      f"forcerange={np.round(m.actuator_forcerange[a],2)} "
      f"gaintype={m.actuator_gaintype[a]} biastype={m.actuator_biastype[a]}")

# ---------- 5. Geometries for contact ----------
P("\n=== Geometries for contact ===")
for g in range(m.ngeom):
    bid = m.geom_bodyid[g]
    collide = (m.geom_contype[g] | m.geom_conaffinity[g]) != 0
    P(f"[{g:3d}] {name(mujoco.mjtObj.mjOBJ_GEOM, g):<16s} body={name(mujoco.mjtObj.mjOBJ_BODY, bid):<16s} "
      f"type={m.geom_type[g]} size={np.round(m.geom_size[g],4)} "
      f"collide={collide} contype={m.geom_contype[g]} conaff={m.geom_conaffinity[g]} "
      f"friction={np.round(m.geom_friction[g],3)} "
      f"solref={np.round(m.geom_solref[g],4)} solimp={np.round(m.geom_solimp[g],3)}")

# ---------- 6. Sites ----------
P("\n=== Sites ===")
for s in range(m.nsite):
    P(f"[{s:2d}] {name(mujoco.mjtObj.mjOBJ_SITE, s):<16s} body={name(mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[s])}")

# ---------- 7. Key frame ----------
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.mj_forward(m, d)

M = np.zeros((m.nv, m.nv))
# import struct
# keys = [attr for attr in dir(mujoco) if attr.startswith("mj_")]
# print(keys)
mujoco.mj_fullM(m, d, M)
bias = d.qfrc_bias.copy()                # C(q,qd)qd + g(q)

P("\n=== Dynamics model under keyframe ===")
P("qpos:", np.round(d.qpos, 4))
P("M:", np.round(np.diag(M), 5))
P("g(q):", np.round(bias, 4))
P("Total mass:", np.round(m.body_mass.sum(), 4), "kg")

# ---------- 8. Foot positions ----------
P("\n=== Foot / Calf body positions in the keyframe ===")
for b in range(m.nbody):
    n = name(mujoco.mjtObj.mjOBJ_BODY, b)
    if "foot" in n.lower() or "calf" in n.lower():
        P(f"{n:<18s} xpos={np.round(d.xpos[b],4)}")

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
np.savez(OUT_NPZ, M=M, bias=bias, qpos0=d.qpos.copy(),
         jnt_names=np.array([name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]),
         act_joint_ids=m.actuator_trnid[:, 0],
         ctrlrange=m.actuator_ctrlrange.copy())
