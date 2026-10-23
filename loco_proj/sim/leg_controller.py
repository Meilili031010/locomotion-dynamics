"""
[NEW FILE] Leg-level control (Cheetah 3 Sec. II-C / V):
    stance leg : tau = -J^T f_mpc + tau_bias + tau_fric
    swing leg  : tau = J^T [Kp (p_d - p) + Kd (v_d - v)] + M_leg J^-1 a_d
                       + tau_bias + tau_fric
    f        = ground reaction force on the robot (MPC output), world frame
    tau_bias = gravity/Coriolis of the leg links (qfrc_bias rows of the leg).
               On a stance leg, leaving it out makes the real contact force
               differ from the MPC force by J^-T tau_bias (~20 N on Go2).
    tau_fric = joint damping + Coulomb friction of the MuJoCo Go2 model
               (0.1 N*m*s/rad, 0.2 N*m). Uncompensated, it ate ~4 N of
               forward force per stance leg at 0.5 m/s.
    M_leg J^-1 a_d = inertial feed-forward so swing feet do not lag the
               trajectory and land with residual forward speed.
All Cartesian quantities are world frame; J is the 3x3 block of the foot
Jacobian w.r.t. the leg's own joints.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from state import State


@dataclass
class LegControlConfig:
    kp_swing: np.ndarray = field(default_factory=lambda: np.array([400.0, 400.0, 400.0]))
    kd_swing: np.ndarray = field(default_factory=lambda: np.array([12.0, 12.0, 12.0]))
    swing_bias_ff: bool = True       # gravity/Coriolis of the leg links
    stance_bias_ff: bool = True
    swing_accel_ff: bool = True      # M_leg J^-1 a_d
    friction_comp: bool = True       # joint damping + Coulomb friction
    friction_vel_eps: float = 0.05   # [rad/s] smooth sign() for Coulomb term


class LegController:
    def __init__(self, st: State, cfg: LegControlConfig = None):
        self.st = st
        self.cfg = cfg or LegControlConfig()
        m = st.model
        self.damping = m.dof_damping[st.leg_dofs]          # (4, 3)
        self.frictionloss = m.dof_frictionloss[st.leg_dofs]
        self._M = np.zeros((m.nv, m.nv))

    def torques(self, contacts: np.ndarray, forces: np.ndarray,
                swing_ref: np.ndarray, J: np.ndarray = None) -> np.ndarray:
        """
        contacts  (4,) bool   True = stance
        forces    (4, 3)      MPC ground reaction forces (world)
        swing_ref (4, 3, 3)   per leg [pos; vel; acc] (world), used for swing legs
        J         (4, 6, nv)  foot Jacobians (optional, recomputed if None)
        returns   (4, 3)      joint torques, LEGS x [hip, thigh, calf], clipped
        """
        st, cfg = self.st, self.cfg
        if J is None:
            J = st.foot_jacobians
        p_foot = st.foot_positions
        v_foot = J[:, :3, :] @ st.data.qvel
        bias = st.leg_bias
        qd = st.joint_velocities

        if cfg.swing_accel_ff and not contacts.all():
            mujoco.mj_fullM(st.model, st.data, self._M)

        tau = np.zeros((4, 3))
        for i in range(4):
            Ji = J[i][:3][:, st.leg_dofs[i]]         # (3, 3): d p_foot / d q_leg
            if contacts[i]:
                tau[i] = -Ji.T @ forces[i]
                if cfg.stance_bias_ff:
                    tau[i] += bias[i]
            else:
                f = cfg.kp_swing * (swing_ref[i, 0] - p_foot[i]) \
                    + cfg.kd_swing * (swing_ref[i, 1] - v_foot[i])
                tau[i] = Ji.T @ f
                if cfg.swing_accel_ff:
                    d = st.leg_dofs[i]
                    M_leg = self._M[np.ix_(d, d)]
                    tau[i] += M_leg @ np.linalg.solve(Ji, swing_ref[i, 2])
                if cfg.swing_bias_ff:
                    tau[i] += bias[i]
            if cfg.friction_comp:
                tau[i] += self.damping[i] * qd[i] \
                    + self.frictionloss[i] * np.tanh(qd[i] / cfg.friction_vel_eps)
        return np.clip(tau, -st.tau_limit, st.tau_limit)

    def apply(self, tau: np.ndarray):
        """Write (4, 3) torques into data.ctrl using the actuator ordering."""
        self.st.data.ctrl[self.st.act_idx.reshape(-1)] = tau.reshape(-1)
