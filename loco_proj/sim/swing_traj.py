from dataclasses import dataclass
import numpy as np

"""
    In MIT Cheetah 3, the velocities for both real and cmd are in world frame
"""

@dataclass
class SwingTrajectoryConfig:
    step_height: float = 0.08
    t_swing: float = 0.25
    touchdown_velocity: float = 0.
    gravity: float = 9.81
    k_fb: float = None


def p_sym(v_base: np.ndarray, t_stance: float, heading: float = 0.0) -> np.ndarray:
    # [FIXED] was |v_xy| * [cos(heading), sin(heading)]: the norm drops the sign
    # and the lateral part of v, so backward/sideways walking still stepped
    # forward. Cheetah 3: p_symmetry = (t_stance / 2) * v  (world frame).
    # `heading` is kept only so old calls still work; it is not needed.
    out = np.zeros(3)
    out[:2] = v_base[:2] * (t_stance / 2.0)
    return out


def p_fb(v_base: np.ndarray, v_cmd: np.ndarray, height: float, k_fb: float = None) -> np.ndarray:
    if k_fb is None:
        k_fb = np.sqrt(max(height, 0.01) / 9.81)
    err = np.zeros(3)
    err[:2] = v_base[:2] - v_cmd[:2]
    return k_fb * err


def p_rot(v_cmd: np.ndarray, r_hip: np.ndarray, t_stance: float) -> np.ndarray:
    omega = np.array([0.0, 0.0, v_cmd[5] if len(v_cmd) > 5 else v_cmd[2]])
    return np.cross(omega, r_hip) * (t_stance / 2.0)


def p_terrain(coords: np.ndarray) -> np.ndarray:
    # 0.022 = Go2 foot sphere radius, i.e. foot-center height on flat ground
    return np.array([0.0, 0.0, 0.022])


# ---- [ADDED] begin ----
def p_centrifugal(v_base: np.ndarray, v_cmd: np.ndarray, height: float,
                  gravity: float = 9.81) -> np.ndarray:
    """Cheetah 3 eq.: p_centrifugal = 1/2 * sqrt(h/g) * (v x w_cmd)."""
    omega = np.array([0.0, 0.0, v_cmd[2]])
    v = np.array([v_base[0], v_base[1], 0.0])
    return 0.5 * np.sqrt(max(height, 0.01) / gravity) * np.cross(v, omega)
# ---- [ADDED] end ----


def compute_foot_placement(
    p_hip: np.ndarray,
    v_base: np.ndarray,
    v_cmd: np.ndarray,
    height: float,
    r_hip: np.ndarray,
    t_stance: float,
    heading: float,
    cfg: SwingTrajectoryConfig,
    t_swing_remaining: float = 0.0,     # [ADDED]
    use_centrifugal: bool = True,       # [ADDED]
) -> np.ndarray:
    """
    [ADDED doc] Cheetah 3 footstep heuristic, all vectors in world frame.
        p_hip   : current hip ("shoulder") position
        v_base  : current base velocity
        v_cmd   : [vx_cmd, vy_cmd, wz_cmd]   (world frame)
        r_hip   : hip offset from base, world frame (for the yaw term)
    The hip is first moved to where it will be at touchdown
    (p_hip + v_cmd * t_swing_remaining), as in the Cheetah software.
    """
    # [ADDED] predict hip position at touchdown
    p_hip_td = p_hip + np.array([v_cmd[0], v_cmd[1], 0.0]) * t_swing_remaining

    p_s = p_sym(v_base, t_stance, heading)
    p_f = p_fb(v_base, v_cmd, height, cfg.k_fb)
    p_r = p_rot(v_cmd, r_hip, t_stance)
    p_c = p_centrifugal(v_base, v_cmd, height, cfg.gravity) if use_centrifugal else 0.0  # [ADDED]

    target_xy = p_hip_td + p_s + p_f + p_r + p_c
    target_xy[2] = 0.0   # [FIXED] p_hip carries the hip height; it was added to the terrain z
    target_z = p_terrain(target_xy)

    return target_xy + target_z


def cubic_bezier(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        s) -> tuple[np.ndarray, np.ndarray]:
    s = np.clip(s, 0.0, 1.0)
    pos = (1 - s)**3 * p0 + 3 * (1 - s)**2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * p3
    dpos_ds = 3 * (1 - s)**2 * (p1 - p0) + 6 * (1 - s) * s * (p2 - p1) + 3 * s**2 * (p3 - p2)
    return pos, dpos_ds


# [ADDED] second derivative, for the swing feed-forward acceleration
def cubic_bezier_dds(p0, p1, p2, p3, s):
    s = np.clip(s, 0.0, 1.0)
    return 6 * (1 - s) * (p2 - 2 * p1 + p0) + 6 * s * (p3 - 2 * p2 + p1)


def foot_trajectory(
    t_phase: float,
    p0: np.ndarray,
    pf: np.ndarray,
    cfg: SwingTrajectoryConfig,
    t_swing: float = None,   # [ADDED] swing time from the gait; falls back to cfg.t_swing
) -> np.ndarray:
    """
    [ADDED doc] Returns (3, 3): rows [pos; vel; acc], world frame.
    cfg.touchdown_velocity is the z velocity at touchdown (negative = downward).
    """
    T = cfg.t_swing if t_swing is None else t_swing   # [ADDED]

    t_clamped = np.clip(t_phase, 0.0, T)
    s = t_clamped / T if T > 0.0 else 1.0
    ds_dt = 1.0 / T if T > 0.0 else 0.0

    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)

    p0_xy = p0[:2]
    p1_xy = p0[:2]
    p2_xy = pf[:2]
    p3_xy = pf[:2]

    pos[:2], dpos_ds_xy = cubic_bezier(p0_xy, p1_xy, p2_xy, p3_xy, s)
    vel[:2] = dpos_ds_xy * ds_dt
    acc[:2] = cubic_bezier_dds(p0_xy, p1_xy, p2_xy, p3_xy, s) * ds_dt**2   # [ADDED]

    z_lift = p0[2]
    z_td = pf[2]
    z_apex = max(z_lift, z_td) + cfg.step_height

    if s <= 0.5:
        s_sub = s * 2.0
        ds_sub_ds = 2.0
        z_val, dz_ds_sub = cubic_bezier(z_lift, z_lift, z_apex, z_apex, s_sub)
        pos[2] = z_val
        vel[2] = dz_ds_sub * ds_sub_ds * ds_dt
        acc[2] = cubic_bezier_dds(z_lift, z_lift, z_apex, z_apex, s_sub) * (ds_sub_ds * ds_dt)**2  # [ADDED]
    else:
        s_sub = (s - 0.5) * 2.0
        ds_sub_ds = 2.0
        p2_z = z_td - (cfg.touchdown_velocity / (3.0 * ds_sub_ds * ds_dt)) if ds_dt > 0.0 else z_td
        z_val, dz_ds_sub = cubic_bezier(z_apex, z_apex, p2_z, z_td, s_sub)
        pos[2] = z_val
        vel[2] = dz_ds_sub * ds_sub_ds * ds_dt
        acc[2] = cubic_bezier_dds(z_apex, z_apex, p2_z, z_td, s_sub) * (ds_sub_ds * ds_dt)**2  # [ADDED]

    return np.vstack([pos, vel, acc])