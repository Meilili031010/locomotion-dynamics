import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from structs import LEGS, JOINTS


class State:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.nq = model.nq
        self.nv = model.nv

        self.foot_geoms = np.array([model.geom(n).id for n in LEGS])

        # row = leg, col = hip/thigh/calf
        jids = [model.joint(f"{leg}_{j}_joint").id for leg in LEGS for j in JOINTS]
        hids = [model.joint(f"{leg}_hip_joint").id for leg in LEGS]
        self.qadr = model.jnt_qposadr[jids]
        self.vadr = model.jnt_dofadr[jids]

        # hip position in body-fixed frame
        # [FIXED] was data.xanchor[hids]: that is the WORLD-frame joint anchor
        # (includes base height/pose, and is all zeros before mj_forward).
        # Hip bodies are direct children of base_link, so body_pos is already
        # the hip offset in the base frame.
        hip_bodies = [model.joint(f"{leg}_hip_joint").bodyid for leg in LEGS]
        self.hip_b = model.body_pos[np.ravel(hip_bodies)].copy()

        # ---- [ADDED] begin ----
        self.base_body = model.body("base_link").id
        # Actuators in the Go2 xml are ordered FR, FL, RR, RL (not LEGS order).
        # act_idx[leg, joint] = actuator index, so ctrl[act_idx] = tau (4, 3).
        act_of_joint = {model.actuator_trnid[i, 0]: i for i in range(model.nu)}
        self.act_idx = np.array([act_of_joint[j] for j in jids]).reshape(4, 3)
        self.tau_limit = model.actuator_ctrlrange[self.act_idx, 1]   # (4, 3)
        # dof indices of each leg (4, 3), for slicing Jacobians / bias forces
        self.leg_dofs = self.vadr.reshape(4, 3)
        # ---- [ADDED] end ----

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

    # ---- [ADDED] begin ----
    @property
    def R_wb(self):
        """(3, 3) base rotation, body -> world."""
        return self.data.xmat[self.base_body].reshape(3, 3).copy()

    @property
    def com_pos(self):
        """(3,) whole-robot CoM in world frame (MPC moment arms are taken about this)."""
        return self.data.subtree_com[self.base_body].copy()

    @property
    def hip_world(self):
        """(4, 3) hip positions in world frame."""
        return self.data.qpos[:3] + self.hip_b @ self.R_wb.T

    @property
    def leg_bias(self):
        """(4, 3) gravity + Coriolis torques of the leg joints (qfrc_bias)."""
        return self.data.qfrc_bias[self.vadr].reshape(4, 3)

    def mpc_state(self, gravity: float = -9.81) -> np.ndarray:
        """(13,) MPC state [p, rpy, v, w_world, g] from the base link (Cheetah 3)."""
        return np.concatenate([self.base_pose, self.base_lin_vel_world,
                               self.base_ang_vel_world, [gravity]])

    def mpc_state_centroidal(self, I_body: np.ndarray, com_b: np.ndarray,
                             gravity: float = -9.81) -> np.ndarray:
        """
        (13,) MPC state built from whole-robot (centroidal) quantities:
            p = CoM - R com_b      ("virtual base": CoM shifted by the nominal
                                    offset, so zd still means base height)
            v = CoM velocity
            w = (R I_body R^T)^-1 L_com   (L_com: total angular momentum)
        Swinging legs exchange momentum with the base but cannot change v_com
        or L_com, only ground forces can. With Go2's heavy legs (~45% of the
        mass) this matches the MPC model much better than base-link velocities.
        """
        mujoco.mj_subtreeVel(self.model, self.data)
        b = self.base_body
        R = self.R_wb
        pose = self.base_pose
        p = self.data.subtree_com[b] - R @ com_b
        v = self.data.subtree_linvel[b].copy()
        L = self.data.subtree_angmom[b].copy()
        w = np.linalg.solve(R @ I_body @ R.T, L)
        return np.concatenate([p, pose[3:], v, w, [gravity]])

    # ---- [ADDED] ----
def robot_params(model: mujoco.MjModel, key: int = 0) -> dict:
    """
    Single-rigid-body parameters for the MPC, computed at the keyframe pose.

    Cheetah 3 lumps the whole robot into one body. Go2 legs are ~45% of the
    mass, so the composite inertia of all bodies about the total CoM is used
    (not just base_link's).

    Returns dict:
        mass      total mass [kg]
        I_body    (3, 3) composite inertia about the CoM, base frame
        com_b     (3,) CoM offset from base origin, base frame
        foot_b    (4, 3) nominal foot (sphere center) positions, base frame.
                  Used as the "shoulder" point for foot placement: it includes
                  the lateral hip/thigh offset that body_pos of the hip lacks.
    """
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, key)
    mujoco.mj_forward(model, d)
    base = model.body("base_link").id
    Rwb = d.xmat[base].reshape(3, 3)
    p_base = d.xpos[base]
    com = d.subtree_com[base]
    mass = float(model.body_subtreemass[base])

    I_w = np.zeros((3, 3))
    for b in range(1, model.nbody):
        m_b = model.body_mass[b]
        if m_b <= 0:
            continue
        Rb = d.ximat[b].reshape(3, 3)
        I_w += Rb @ np.diag(model.body_inertia[b]) @ Rb.T
        r = d.xipos[b] - com
        I_w += m_b * (r @ r * np.eye(3) - np.outer(r, r))

    foot_ids = [model.geom(n).id for n in LEGS]
    return dict(
        mass=mass,
        I_body=Rwb.T @ I_w @ Rwb,
        com_b=Rwb.T @ (com - p_base),
        foot_b=(d.geom_xpos[foot_ids] - p_base) @ Rwb,
    )
    # ---- [ADDED] end ----

# Test
if __name__ == "__main__":
    GO2_XML = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/go2.xml"
    model = mujoco.MjModel.from_xml_path(GO2_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    s = State(model, data)
    # print(s.base_pose)
    # print(s.foot_positions)
    # print(data.qfrc_bias)
    print(s.hip_b.shape)