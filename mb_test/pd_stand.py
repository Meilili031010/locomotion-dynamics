import mujoco
import mujoco.viewer
import time
import numpy as np

MODEL_PATH = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/scene.xml"
SIM_TIME = 10.0
KP, KD = 4.0, 1.0

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
    print("Frame 0 as the reference frame")
else:
    mujoco.mj_resetData(m, d)
    print("No keyframe, reset to default pose")

q_target = d.qpos.copy()
d.qpos[:] = q_target
mujoco.mj_forward(m, d)

# actuator - joint qpos/qvel 
act_qadr = []
act_dadr = []
for i in range(m.nu):
    jid = m.actuator_trnid[i, 0]
    act_qadr.append(m.jnt_qposadr[jid])
    act_dadr.append(m.jnt_dofadr[jid])

is_position = all(m.actuator_biastype[i] == 1 for i in range(m.nu))
print("Control mode:", "Position-based actuator (write target angles)" if is_position else "Torque-based (compute PD internally)")

def compute_ctrl():
    if is_position:
        return np.array([q_target[a] for a in act_qadr])
    tau = np.zeros(m.nu)
    for i in range(m.nu):
        q = d.qpos[act_qadr[i]]
        qd = d.qvel[act_dadr[i]]
        tau[i] = KP * (q_target[act_qadr[i]] - q) - KD * qd
    return tau

steps = int(SIM_TIME / m.opt.timestep)

SYNC_EVERY = 10
viewer = mujoco.viewer.launch_passive(m, d)

try:
    k = 0
    while viewer.is_running() and d.time < SIM_TIME:
        t0 = time.time()

        d.ctrl[:] = compute_ctrl()
        mujoco.mj_step(m, d)

        if k % SYNC_EVERY == 0:
            viewer.sync()

        dt = m.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
        k += 1
finally:
    # 1. Explicitly notify the GLFW/EGL backend to close the window
    viewer.close()
    # 2. Allow any pending Wayland/EGL events to flush before Python finalizes
    time.sleep(0.1)

print("\nFinished, base position:", d.qpos[0:3])