import mujoco
import numpy as np
from Def import LEGS, JOINTS, IDX_KEYS, G
from scipy.spatial.transform import Rotation as R
class Interface:
    """
    Read-only view of the Go2 state + torque output, in LEGS order (FL, FR, RL, RR).
    Valid after mj_forward / mj_step1. Arrays are copies unless noted.
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.nu = model.nu      # 12 joint actuator
        self.nv = model.nv      # 1 x 6 DOF vel + 12 joint vel
        self.m = model
        self.d = data
        self.jnt_idx = [self.m.joint(jnt).id for jnt in IDX_KEYS]
        self.foot_geoms_idx = [self.m.geom(leg).id for leg in LEGS]
        self.qadr = model.jnt_qposadr[self.jnt_idx].reshape((4,3))
        self.vadr = model.jnt_dofadr[self.jnt_idx].reshape((4,3))
        self.base_idx = self.m.body("base_link").id
        act_of_joint = {model.actuator_trnid[i, 0]: i for i in range(model.nu)}
        self.act_idx = np.array([act_of_joint[i] for i in self.jnt_idx])   # (12,), LEGS x JOINTS order
        self.tau_limit = model.actuator_ctrlrange[self.act_idx,1].reshape((4,3))
        hip_body = [model.joint(f"{leg}_hip_joint").bodyid for leg in LEGS]
        self.hip_b = model.body_pos[np.ravel(hip_body)].copy()
        self._M = np.zeros((model.nv, model.nv))

    @property
    def base_pose_w(self):
        """(6,) [x, y, z, roll, pitch, yaw]."""
        q = self.d.qpos[:7]
        rpy = R.from_quat(q[[4,5,6,3]]).as_euler("xyz")
        return np.concatenate([q[:3], rpy])

    # [FIXED] was named R_bw, but xmat is the body -> world rotation R_wb
    # (v_world = R_wb @ v_body). Renamed so the math reads correctly.
    @property
    def R_wb(self):
        """(3, 3) base rotation, body -> world."""
        return self.d.xmat[self.base_idx].reshape((3,3)).copy()

    @property
    def base_v_w(self):
        """(6,) [v_world (3), w_world (3)]  -- linear first, same order as the MPC state."""
        # [FIXED] returned [w, v]: QuadrupedController.step() and mpc_state()
        # read it as [v, w], so foot placement used the angular velocity as
        # the base velocity and the base-link MPC state had v and w swapped.
        return np.concatenate([self.d.qvel[:3], self.R_wb @ self.d.qvel[3:6]])

    @property
    def jnt_pos(self):
        """(4, 3) joint angles."""
        return self.d.qpos[self.qadr]

    @property
    def jnt_vel(self):
        """(4, 3) joint velocities."""
        # [FIXED] read qpos[vadr] (joint angles of the wrong joints); qvel is right.
        return self.d.qvel[self.vadr]

    # [FIXED] was `jnt_fric`, but qfrc_actuator is the applied actuator torque,
    # not friction.
    @property
    def jnt_tau(self):
        """(4, 3) torque applied by the motors (after clipping)."""
        return self.d.qfrc_actuator[self.vadr]

    @property
    def foot_pos(self):
        """(4, 3) foot sphere centers, world."""
        return self.d.geom_xpos[self.foot_geoms_idx].copy()   # [FIXED] copy, not a view

    @property
    def foot_jac(self):
        """(4, 6, nv) rows [linear (3); angular (3)], world frame. nv: body(6) + joint(12)"""
        Jac = np.zeros((4,6,self.nv))
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        for i,j in enumerate(self.foot_geoms_idx):
            mujoco.mj_jacGeom(self.m, self.d, jacp,jacr,j)
            # [FIXED] was (jacr, jacp): LegController takes J[i][:3] as the
            # LINEAR Jacobian, so stance torques used the rotational one
            # (the robot fell while standing).
            Jac[i, :3], Jac[i, 3:] = jacp, jacr
        return Jac

    @property
    def foot_vel(self):
        """(4, 3) foot velocities, world."""
        return self.foot_jac[:, :3, :] @ self.d.qvel

    @property
    def com_pos(self):
        """(3,) whole-robot CoM, world."""
        return self.d.subtree_com[self.base_idx].copy()   # [FIXED] self.data -> self.d

    @property
    def hip_w(self):
        """(4, 3) hip joint positions, world."""
        # [FIXED] was base_pose_w (6,) + hip_b @ R -> broadcast error.
        return self.d.qpos[:3] + self.hip_b @ self.R_wb.T

    @property
    def leg_inertia(self):
        """(4, 3, 3) joint-space mass matrix block of each leg."""
        # [FIXED] was `raise NotImplemented` (NotImplemented is not an exception)
        mujoco.mj_fullM(self.m, self.d, self._M)
        return np.stack([self._M[np.ix_(v, v)] for v in self.vadr])

    @property
    def leg_bias(self):
        """(4, 3) gravity + Coriolis"""
        return self.d.qfrc_bias[self.vadr]

    def mpc_state(self, gravity: float = G) -> np.ndarray:
        """(13,) MPC state [p, rpy, v, w_world, g] from the base link (Cheetah 3)."""
        # [FIXED] took no gravity argument, but the controller calls mpc_state(G)
        return np.concatenate([self.base_pose_w, self.base_v_w, [gravity]])

    def mpc_state_centroidal(self, I_body: np.ndarray, com_b: np.ndarray) -> np.ndarray:
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
        mujoco.mj_subtreeVel(self.m, self.d)
        b = self.base_idx
        R = self.R_wb
        pose = self.base_pose_w
        p = self.d.subtree_com[b] - R @ com_b
        v = self.d.subtree_linvel[b].copy()
        L = self.d.subtree_angmom[b].copy()
        w = np.linalg.solve(R @ I_body @ R.T, L)
        return np.concatenate([p, pose[3:], v, w, [G]])

    # TODO (yours): actual contact state / force from MuJoCo, to compare with the plan
    def sim_detect_contact(self):
        raise NotImplementedError   # [FIXED] was `raise NotImplemented`

    def sim_get_contact_force(self):
        raise NotImplementedError

    def sim_ctrl(self, tau: np.ndarray):
        """(4, 3) torques in LEGS order -> clipped -> data.ctrl (actuator order FR, FL, RR, RL)."""
        tau_cmd = np.clip(tau, -self.tau_limit, self.tau_limit)
        self.d.ctrl[self.act_idx] = tau_cmd.ravel()   # [FIXED] was tau_cmd.reshape.ravel()
        return tau_cmd

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


if __name__ == "__main__":
    from pathlib import Path
    file_path = Path(__file__).resolve()
    GO2_XML = file_path.parents[1] / "unitree_ws/src/unitree_mujoco/unitree_robots/go2/go2.xml"
    # print(GO2_XML.is_file())
    m = mujoco.MjModel.from_xml_path(str(GO2_XML))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    go2 = Interface(m,d)
    print(go2.foot_vel)