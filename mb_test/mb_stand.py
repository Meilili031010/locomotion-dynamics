import time
import numpy as np
import mujoco
import mujoco.viewer

GO2_XML = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/go2.xml"
SIM_TIME = 5.0
KP, KD = 40.0, 1.0
USE_VIEWER = True
PRINT_EVERY = 0.5
SYNC_EVERY = 10

spec = mujoco.MjSpec.from_file(GO2_XML)
floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [0.0, 0.0, 0.05]
floor.friction = [1.0, 0.005, 0.0]
floor.solref = [0.02, 1.0]
floor.rgba = [0.55, 0.55, 0.55, 1.0]

light = spec.worldbody.add_light()
light.pos = [0.0, 0.0, 3.0]
light.dir = [0.0, 0.0, -1.0]
light.diffuse = [0.8, 0.8, 0.8]
m = spec.compile()
d = mujoco.MjData(m)

HOME_QPOS = np.array([0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0] + [0.0, 0.9, -1.8] * 4)
d.qpos[:] = HOME_QPOS
d.qvel[:] = 0.0
mujoco.mj_forward(m, d)

q_target = d.qpos.copy()
act_qadr = [m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
act_dadr = [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]

FOOT_GEOMS = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
              for n in ("FL", "FR", "RL", "RR")]
F_LEG = m.body_mass.sum() * 9.81 / 4.0
jacp = np.zeros((3, m.nv))

def static_gen_force():
    gen = d.qfrc_bias.copy()
    for gid in FOOT_GEOMS:
        mujoco.mj_jacGeom(m, d, jacp, None, gid)
        gen -= jacp.T @ np.array([0.0, 0.0, F_LEG])
    return gen

def compute_ctrl():
    gen = static_gen_force()
    tau = np.zeros(m.nu)
    for i in range(m.nu):
        qa, da = act_qadr[i], act_dadr[i]
        tau[i] = gen[da] + KP * (q_target[qa] - d.qpos[qa]) - KD * d.qvel[da]
    return tau

log_every = int(round(PRINT_EVERY / m.opt.timestep))
steps = int(SIM_TIME / m.opt.timestep)

def run(viewer=None):
    print(f"Intial base_z = {d.qpos[2]:.4f}, ncon = {d.ncon}")
    print("forward joint torques:", np.round(static_gen_force()[6:], 3))
    try:
        for k in range(steps):
            if viewer is not None and not viewer.is_running():
                break
            t0 = time.time()
            d.ctrl[:] = compute_ctrl()
            mujoco.mj_step(m, d)

            if k % log_every == 0:
                err = np.abs(d.qpos[7:] - q_target[7:]).max()
                print(f"t={d.time:4.2f}s  base_z={d.qpos[2]:.4f}  "
                    f"max_joint_err={err:.4f}  ncon={d.ncon}")

            if viewer is not None and k % SYNC_EVERY == 0:
                viewer.sync()
                dt = SYNC_EVERY * m.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
    finally:
        if viewer is not None:
            viewer.close()

viewer = mujoco.viewer.launch_passive(m, d) if USE_VIEWER else None
run(viewer)


print("\nfinal joint errors:")
err = d.qpos[7:] - q_target[7:]
for j in range(m.njnt - 1):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j + 1)
    print(f"  {name:<18s} {err[j]:+.4f}")