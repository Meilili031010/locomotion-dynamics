import numpy as np
import mujoco

DT = 0.001
USE_RK4 = True
HOME_QPOS = np.array([0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0] + [0.0, 0.9, -1.8] * 4)
GO2_XML = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/go2.xml"


spec = mujoco.MjSpec.from_file(GO2_XML)
m = spec.compile()

m.opt.gravity[:] = 0.0
m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
m.dof_damping[:] = 0.0
m.dof_frictionloss[:] = 0.0
m.opt.timestep = DT
m.opt.integrator = (mujoco.mjtIntegrator.mjINT_RK4 if USE_RK4
                    else mujoco.mjtIntegrator.mjINT_EULER)

d = mujoco.MjData(m)
d.qpos[:] = HOME_QPOS
d.qpos[2] = 0.5
mujoco.mj_forward(m, d)

M = np.zeros((m.nv, m.nv))
def kinetic_energy():
    mujoco.mj_fullM(m, d, M)
    return 0.5 * d.qvel @ M @ d.qvel

rng = np.random.default_rng(0)
freq = rng.uniform(0.5, 3.0, size=(m.nv, 2))
amp = rng.uniform(0.5, 3.0, size=(m.nv, 2))

def excite(t):
    tau = np.zeros(m.nv)
    tau[6:] = (amp[6:, 0] * np.sin(2 * np.pi * freq[6:, 0] * t)
               + amp[6:, 1] * np.sin(2 * np.pi * freq[6:, 1] * t))
    return tau

T0 = kinetic_energy()
W = 0.0
for k in range(int(3.0 / DT)):
    tau = excite(d.time)
    d.qfrc_applied[:] = tau
    v0 = d.qvel.copy()
    mujoco.mj_step(m, d)
    W += tau @ (0.5 * (v0 + d.qvel)) * DT
T1 = kinetic_energy()

dT = T1 - T0
print(f"dt={DT}  RK4={USE_RK4}")
print(f"Delta T = {dT:.6f}, W = {W:.6f}, Residual = {dT - W:.4e}, error ratio = {(dT - W) / dT:.2%}")