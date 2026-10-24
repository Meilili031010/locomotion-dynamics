from state import State
import mujoco
import numpy as np
from structs import LegControllerCfg

class LegController:
    def __init__(self, state: State, cfg: LegControllerCfg = None):
        self.state = state
        self.cfg = cfg
        m = state.model
        self.fric = m.dof_damping[state.leg_dofs]
        self.no_fric = m.dof_frictionless[state.leg_dofs]
        self.M_s = np.zeros((m.nv, m.nv))

    def torques(self, contacts: np.ndarray, forces: np.ndarray,
                swing_ref: np.ndarray, J: np.ndarray) -> np.ndarray:
        """
            contacts  (4,) bool   True = stance
            forces    (4, 3)      MPC solved forces (world)
            swing_ref (4, 3, 3)   swing leg [pos; vel; acc] (world)
            J         (4, 6, nv)  foot Jacobians
            returns   (4, 3)      joint torques, 4 x [hip, thigh, calf]
        """
        st, cfg = self.state, self.cfg
        p_foot = st.foot_positions
        v_foot = J[:, :3, :] @ st.data.qvel
        bias = st.leg_bias      # gravity
        qdot = st.joint_velocities
        mujoco.mj_fullM(st.model, st.data, self.M_s)

        tau = np.zeros((4,3))
        for i in range(4):
            Ji = J[i][:, st.leg_dofs[i]]
            if contacts[i]:
                tau[i] += -Ji.T @ forces[i] + bias[i]
            else:
                f = cfg.kp_swing * (swing_ref[i,0] - p_foot[i]) + cfg.kd_swing * (swing_ref[i,1] - v_foot[i])
                tau[i] = Ji.T @ f
                idx = st.leg_dofs[i]
                M_leg = self.M_s[np.ix_(idx, idx)]
                tau[i] += M_leg @ np.linalg.solve(Ji, swing_ref[i, 2]) + bias[i]
                # 
                tau[i] += self.fric[i] * qdot[i] + self.no_fric[i] * np.tanh(qdot[i] / cfg.jnt_fric_coeff)

        return np.clip(tau, -st.tau_limit, st.tau_limit)

    def command(self, tau: np.ndarray):
        self.st.data.ctrl[self.st.act_idx.reshape(-1)] = tau.reshape(-1)
    

                

