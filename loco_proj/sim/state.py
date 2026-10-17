import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

LEGS = ["FL", "FR", "RL", "RR"]
JOINTS = ["hip", "thigh", "calf"]


class State:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv

        self.foot_geoms = np.array([model.geom(n).id for n in LEGS])

        # row = leg, col = hip/thigh/calf
        jids = [model.joint(f"{leg}_{j}_joint").id for leg in LEGS for j in JOINTS]
        self.qadr = model.jnt_qposadr[jids]
        self.vadr = model.jnt_dofadr[jids]

    @property
    def base_pose(self):
        """[x, y, z, roll, pitch, yaw]; R = Rz(yaw) Ry(pitch) Rx(roll)."""
        q = self.data.qpos[3:7]                       # [w, x, y, z]
        rpy = R.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")
        return np.concatenate([self.data.qpos[:3], rpy])

    @property
    def base_lin_vel_world(self):
        return self.data.qvel[:3].copy()

    @property
    def base_ang_vel_world(self):
        Rwb = self.data.xmat[1].reshape(3, 3)         # body 1 = floating base
        return Rwb @ self.data.qvel[3:6]              # qvel[3:6] is body-frame

    @property
    def joint_positions(self):
        return self.data.qpos[self.qadr].reshape(4, 3)

    @property
    def joint_velocities(self):
        return self.data.qvel[self.vadr].reshape(4, 3)

    @property
    def joint_torques(self):
        return self.data.qfrc_actuator[self.vadr].reshape(4, 3)

    @property
    def foot_positions(self):
        return self.data.geom_xpos[self.foot_geoms].copy()

    @property
    def foot_jacobians(self):
        """(4, 6, nv), rows [linear; angular], world frame"""
        J = np.zeros((4, 6, self.nv))
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        for i, g in enumerate(self.foot_geoms):
            mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, g)
            J[i, :3], J[i, 3:] = jacp, jacr
        return J

    @property
    def foot_velocities(self):
        """(4, 3) world-frame velocity"""
        return self.foot_jacobians[:, :3, :] @ self.data.qvel

# Test
GO2_XML = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/go2.xml"
model = mujoco.MjModel.from_xml_path(GO2_XML)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

s = State(model, data)
print(s.base_pose)
print(s.foot_positions)
print(data.qfrc_bias)