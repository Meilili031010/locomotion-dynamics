from dataclasses import dataclass
import numpy as np

@dataclass
class SwingTrajectoryConfig:
    step_height: float = 0.08
    t_swing: float = 0.25
    touchdown_velocity: float = 0.
    gravity: float = 9.81
    k_fb: float = None


def p_sym(v_base: np.ndarray, t_stance: float, heading: float) -> np.ndarray:
    v_xy = np.linalg.norm(v_base[:2])
    return np.array([np.cos(heading), np.sin(heading), 0.0]) * v_xy * (t_stance / 2.0)


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
    return np.array([0.0, 0.0, 0.0])


def compute_foot_placement(
    p_hip: np.ndarray,
    v_base: np.ndarray,
    v_cmd: np.ndarray,
    height: float,
    r_hip: np.ndarray,
    t_stance: float,
    heading: float,
    cfg: SwingTrajectoryConfig
) -> np.ndarray:
    p_s = p_sym(v_base, t_stance, heading)
    p_f = p_fb(v_base, v_cmd, height, cfg.k_fb)
    p_r = p_rot(v_cmd, r_hip, t_stance)
    
    target_xy = p_hip + p_s + p_f + p_r
    target_z = p_terrain(target_xy)
    
    return target_xy + target_z


def cubic_bezier(p0, p1, p2, p3, s):
    s = np.clip(s, 0.0, 1.0)
    pos = (1 - s)**3 * p0 + 3 * (1 - s)**2 * s * p1 + 3 * (1 - s) * s**2 * p2 + s**3 * p3
    dpos_ds = 3 * (1 - s)**2 * (p1 - p0) + 6 * (1 - s) * s * (p2 - p1) + 3 * s**2 * (p3 - p2)
    return pos, dpos_ds


def foot_trajectory(
    t_phase: float,
    p0: np.ndarray,
    pf: np.ndarray,
    cfg: SwingTrajectoryConfig
) -> np.ndarray:

    t_clamped = np.clip(t_phase, 0.0, cfg.t_swing)
    s = t_clamped / cfg.t_swing if cfg.t_swing > 0.0 else 1.0
    ds_dt = 1.0 / cfg.t_swing if cfg.t_swing > 0.0 else 0.0

    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)

    p0_xy = p0[:2]
    p1_xy = p0[:2]
    p2_xy = pf[:2]
    p3_xy = pf[:2]

    pos[:2], dpos_ds_xy = cubic_bezier(p0_xy, p1_xy, p2_xy, p3_xy, s)
    vel[:2] = dpos_ds_xy * ds_dt

    z_lift = p0[2]
    z_td = pf[2]
    z_apex = max(z_lift, z_td) + cfg.step_height

    if s <= 0.5:
        s_sub = s * 2.0
        ds_sub_ds = 2.0
        z_val, dz_ds_sub = cubic_bezier(z_lift, z_lift, z_apex, z_apex, s_sub)
        pos[2] = z_val
        vel[2] = dz_ds_sub * ds_sub_ds * ds_dt
    else:
        s_sub = (s - 0.5) * 2.0
        ds_sub_ds = 2.0
        p2_z = z_td - (cfg.touchdown_velocity / (3.0 * ds_sub_ds * ds_dt)) if ds_dt > 0.0 else z_td
        z_val, dz_ds_sub = cubic_bezier(z_apex, z_apex, p2_z, z_td, s_sub)
        pos[2] = z_val
        vel[2] = dz_ds_sub * ds_sub_ds * ds_dt

    return np.vstack([pos, vel, acc])