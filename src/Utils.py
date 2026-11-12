"""
Math helpers shared by the MPC and the reference generator.
[FIXED] This file also held broken copies of the Sim.py functions (make_model,
apply_sim_cfg, reset_robot, ... with undefined names LEGS, _INTEGRATORS,
FOOT_RADIUS). They were never used; the working versions live in Sim.py.
"""
import numpy as np
from Def import NU, NX

def skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def continuous_dynamics(yaw: float, r_foot: np.ndarray, mass: float,
                        I_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ac (13, 13), Bc (13, 12) of the single-rigid-body model.
    r_feet: (4, 3) foot position minus CoM, world frame.
    """
    Rz = rot_z(yaw)
    I_w_inv = np.linalg.inv(Rz @ I_body @ Rz.T)

    Ac = np.zeros((NX, NX))
    Ac[0:3, 6:9] = np.eye(3)
    Ac[3:6, 9:12] = Rz.T
    Ac[8, 12] = 1.0                     # v_z_dot += g  (g state holds -9.81)

    Bc = np.zeros((NX, NU))
    for i in range(4):
        Bc[6:9, 3 * i:3 * i + 3] = np.eye(3) / mass
        Bc[9:12, 3 * i:3 * i + 3] = I_w_inv @ skew(r_foot[i])
    return Ac, Bc


def discretize(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    # [FIXED comment] not Euler: exact zero-order hold expm([[A, B], [0, 0]] dt).
    # M^3 = 0 for this model, so I + M dt + (M dt)^2 / 2 is the full series.
    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ac
    M[:NX, NX:] = Bc
    Mdt = M * dt
    E = np.eye(NX + NU) + Mdt + 0.5 * Mdt @ Mdt
    return E[:NX, :NX], E[:NX, NX:]
