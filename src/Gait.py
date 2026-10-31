import numpy as np
from Def import GaitCfg,SwingTrajectoryCfg, FootPlacementCfg
from Def import _PHASE_EPS, LEGS, G

#################
# Swing
#################
def p_sym(v_base: np.ndarray, t_stance: float, heading: float = 0.0) -> np.ndarray:
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
    footCfg: FootPlacementCfg,
    swingCfg: SwingTrajectoryCfg
) -> np.ndarray:
    """
        p_hip   : current hip ("shoulder") position
        v_base  : current base velocity
        v_cmd   : [vx_cmd, vy_cmd, wz_cmd]   (world frame)
        r_hip   : hip offset from base, world frame (for the yaw term)
    """
    p_hip_td = footCfg.p_hip + np.array([footCfg.v_cmd[0], footCfg.v_cmd[1], 0.0]) * footCfg.t_swing_remaining

    p_s = p_sym(footCfg.v_base, footCfg.t_stance, footCfg.yaw)
    p_f = p_fb(footCfg.v_base, footCfg.v_cmd, footCfg.height, swingCfg.k_fb)
    p_r = p_rot(footCfg.v_cmd, footCfg.r_hip, footCfg.t_stance)
    p_c = p_centrifugal(footCfg.v_base, footCfg.v_cmd, footCfg.height, G) if footCfg.use_centrifugal else 0.0  # [ADDED]

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
    cfg: SwingTrajectoryCfg,
    t_swing: float = None,   # [ADDED] swing time from the gait; falls back to cfg.t_swing
) -> np.ndarray:
    """
        Returns (3, 3): rows [pos; vel; acc], world frame.
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

class GaitScheduler:
    def __init__(self, cfg: GaitCfg):
        self.cfg = cfg
        self.T = cfg.period
        tab = cfg.table

        if not np.any(tab):
            # Whole period is four-leg stance: one sub-phase ending at 1.0
            self.ends = np.array([1.0])
            self.swing = np.zeros((1, 4), dtype=bool)
        else:
            ends_all = tab.max(axis=1)
            if np.any(ends_all == 0):
                raise ValueError(
                    "A row with no nonzero entries has no end time. "
                    "Use a single all-zero row for full stance, or ends/swing arrays."
                )
            self.ends = ends_all
            self.swing = tab > 0
            ends_b = np.broadcast_to(self.ends[:, None], tab.shape)
            if not np.all(tab[self.swing] == ends_b[self.swing]):
                raise ValueError("Nonzero entries in each row must be the same")  # [FIXED] typo

        self.starts = np.concatenate([[0.0], self.ends[:-1]])

        assert np.all(np.diff(self.ends) > 0), "Sub-phase ends must be strictly increasing"
        assert np.isclose(self.ends[-1], 1.0), "The last sub-phase must end at 1.0"
        assert np.all((~self.swing).any(axis=0)), "Every leg needs at least one stance sub-phase"

        self._prev = None

    def phase(self, t: float) -> float:
        """
            Normalized time
        """
        return (t / self.T + _PHASE_EPS) % 1.0

    def row(self, t: float) -> int:
        """
            Sub-period index
        """
        return int(np.searchsorted(self.ends, self.phase(t), side="right"))

    def contacts(self, t: float) -> np.ndarray:
        """
            swing -> stance
        """
        return ~self.swing[self.row(t)]

    def _swing_block_start(self, t: float, r: int, leg: int) -> float:
        n = len(self.ends)
        cyc = int(np.floor(t / self.T))
        j, wraps = r, 0
        for _ in range(n):
            prev = j - 1
            crossed = prev < 0
            if crossed:
                prev += n
            if not self.swing[prev, leg]:
                break
            j = prev
            wraps += int(crossed)
        return (cyc - wraps + self.starts[j]) * self.T

    def swing_elapsed(self, t: float) -> np.ndarray:
        r = self.row(t)
        out = np.zeros(4)
        for leg in range(4):
            if self.swing[r, leg]:
                out[leg] = t - self._swing_block_start(t, r, leg)
        return out

    def stance_duration(self, leg: int) -> float:
        d = self.ends - self.starts
        return float(d[~self.swing[:, leg]].sum() * self.T)

    def swing_duration(self, leg: int) -> float:
        """Total swing time of one leg per gait cycle [s]."""
        return self.T - self.stance_duration(leg)

    def cycle_index(self, t: float) -> int:
        """Number of completed gait cycles at time t."""
        return int(np.floor(t / self.T + _PHASE_EPS))

    def contacts_horizon(self, t0: float, h: int, dt: float) -> np.ndarray:
        """
            Contact table over the MPC horizon.
            Returns (h, 4) bool, row k = contacts at time t0 + k*dt (True = stance).
        """
        times = t0 + dt * np.arange(h)
        rows = np.searchsorted(self.ends, (times / self.T + _PHASE_EPS) % 1.0, side="right")  # [CHANGED] + eps
        return ~self.swing[rows]

    def update(self, t: float):
        """
            event: cur, touchdown, liftoff
        """
        cur = self.contacts(t)
        if self._prev is None:
            td = np.zeros(4, dtype=bool)
            lo = np.zeros(4, dtype=bool)
        else:
            td = cur & ~self._prev
            lo = ~cur & self._prev
        self._prev = cur
        return cur, td, lo


if __name__ == "__main__":
    cfg = GaitCfg()
    # cfg.table = np.array([[0,0,0,0]])
    gait = GaitScheduler(cfg)
    for t in np.arange(0.0, 0.5, 0.1):
        c, td, lo = gait.update(t)
        print(f"t={t:.2f} row={gait.row(t)} contacts={c.astype(int)} "
              f"swing_elapsed={np.round(gait.swing_elapsed(t), 2)}")
    print("stance duration:", [round(gait.stance_duration(i), 3) for i in range(4)])