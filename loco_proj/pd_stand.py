import mujoco
import mujoco.viewer
import time
import numpy as np

MODEL_PATH = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/scene.xml"
SIM_TIME = 3.0
KP, KD = 40.0, 1.0

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

# 站立姿态：优先使用 keyframe，没有则使用模型默认值
if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
    print("使用 keyframe 0 作为初始姿态")
else:
    mujoco.mj_resetData(m, d)
    print("没有 keyframe，使用默认姿态")

q_target = d.qpos.copy()           # 站立目标 = 初始姿态
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
print("控制方式:", "位置型 actuator（写目标角度）" if is_position else "力矩型（自算 PD）")

def compute_ctrl():
    if is_position:
        # 位置型：ctrl 就是目标关节角
        return np.array([q_target[a] for a in act_qadr])
    # 力矩型：tau = KP*(q_target - q) - KD*qd
    tau = np.zeros(m.nu)
    for i in range(m.nu):
        q = d.qpos[act_qadr[i]]
        qd = d.qvel[act_dadr[i]]
        tau[i] = KP * (q_target[act_qadr[i]] - q) - KD * qd
    return tau

steps = int(SIM_TIME / m.opt.timestep)
# for k in range(steps):
#     d.ctrl[:] = compute_ctrl()
#     mujoco.mj_step(m, d)

#     if k % int(0.5 / m.opt.timestep) == 0:
#         base_h = d.qpos[2]
#         joint_err = np.max(np.abs(d.qpos[7:] - q_target[7:]))
        # print(f"t={d.time:5.2f}s  base_z={base_h:.3f}  max_joint_err={joint_err:.3f}  ncon={d.ncon}")

SYNC_EVERY = 10
with mujoco.viewer.launch_passive(m, d) as viewer:
    k = 0
    while viewer.is_running() and d.time < SIM_TIME:
        t0 = time.time()

        d.ctrl[:] = compute_ctrl()
        mujoco.mj_step(m, d)

        if k % SYNC_EVERY == 0:
            viewer.sync()

        # 按实时速度运行，避免仿真过快
        dt = m.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
        k += 1

print("\n站立结束，最终基座位置:", d.qpos[0:3])