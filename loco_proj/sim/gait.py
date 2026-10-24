import numpy as np
from structs import GaitConfig, _PHASE_EPS, LEGS

class GaitScheduler:
    def __init__(self, cfg: GaitConfig):
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
        return (t / self.T + _PHASE_EPS) % 1.0  # [CHANGED] + eps, see _PHASE_EPS

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

    # ---- [ADDED] begin ----
    def swing_duration(self, leg: int) -> float:
        """Total swing time of one leg per gait cycle [s]."""
        return self.T - self.stance_duration(leg)

    def cycle_index(self, t: float) -> int:
        """Number of completed gait cycles at time t (used for Poincare sections)."""
        return int(np.floor(t / self.T + _PHASE_EPS))
    # ---- [ADDED] end ----

    def contacts_horizon(self, t0: float, h: int, dt: float) -> np.ndarray:
        """
            [ADDED doc] Contact table over the MPC horizon.
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
    cfg = GaitConfig()
    # cfg.table = np.array([[0,0,0,0]])
    gait = GaitScheduler(cfg)
    for t in np.arange(0.0, 0.5, 0.1):
        c, td, lo = gait.update(t)
        print(f"t={t:.2f} row={gait.row(t)} contacts={c.astype(int)} "
              f"swing_elapsed={np.round(gait.swing_elapsed(t), 2)}")
    print("stance duration:", [round(gait.stance_duration(i), 3) for i in range(4)])