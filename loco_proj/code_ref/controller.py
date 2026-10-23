"""
[NEW FILE] Cheetah 3 style locomotion controller (one tick = one sim step).

Pipeline each tick:
    gait (contacts, lift-off events)
      -> foot placement policy (swing legs)         foot_placement.py
      -> every mpc.dt: reference + convex MPC        ref_gen.py, mpc_solver.py
      -> swing trajectories                          swing_traj.py
      -> leg torques                                 leg_controller.py

All modules are injected through ControllerConfig / the `placement` argument,
so experiments only swap objects. get_state()/set_state() make the controller
restorable together with the MuJoCo state (needed for Poincare finite
differences).
"""
import copy
from dataclasses import dataclass, field

import mujoco
import numpy as np

from foot_placement import FootPlacementPolicy, PlacementInput, Cheetah3
from gait import GaitConfig, GaitScheduler
from leg_controller import LegControlConfig, LegController
from mpc_solver import ConvexMPC, predict_foot_positions, rot_z
from ref_gen import MPC_Config, ref_gen, GRAVITY
from state import State, robot_params
from swing_traj import SwingTrajectoryConfig, foot_trajectory, p_terrain


@dataclass
class ControllerConfig:
    mpc: MPC_Config = field(default_factory=MPC_Config)
    gait: GaitConfig = field(default_factory=GaitConfig)
    swing: SwingTrajectoryConfig = field(default_factory=SwingTrajectoryConfig)
    leg: LegControlConfig = field(default_factory=LegControlConfig)
    t_stand: float = 0.5                 # four-leg stance before the gait starts [s]
    # True: re-plan the foothold every tick during swing (MIT code).
    # False: freeze it at lift-off. Both are interesting for the return map.
    update_foothold_in_swing: bool = True
    # Re-solve the MPC every N sim steps. None -> once per mpc.dt (Cheetah default).
    mpc_every_steps: int | None = None
    # MPC initial state: True -> CoM velocity + total angular momentum
    # (State.mpc_state_centroidal); False -> base-link state as in Cheetah 3.
    # Go2's legs are ~45% of its mass, so True tracks speed much better.
    centroidal_state: bool = True


class LocomotionController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 cfg: ControllerConfig = None, placement: FootPlacementPolicy = None):
        self.model, self.data = model, data
        self.cfg = cfg or ControllerConfig()
        self.placement = placement or Cheetah3()
        self.st = State(model, data)
        self.params = robot_params(model)

        self.sim_dt = model.opt.timestep
        self.mpc_every = int(round(self.cfg.mpc.dt / self.sim_dt))
        assert abs(self.mpc_every * self.sim_dt - self.cfg.mpc.dt) < 1e-9, \
            "mpc.dt must be a multiple of the sim timestep"
        if self.cfg.mpc_every_steps:
            self.mpc_every = self.cfg.mpc_every_steps

        self.gait = GaitScheduler(self.cfg.gait)
        self.mpc = ConvexMPC(self.cfg.mpc, self.params["mass"], self.params["I_body"])
        self.legs = LegController(self.st, self.cfg.leg)
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        self.k = 0                                      # tick counter
        self.prev_contacts = np.ones(4, dtype=bool)
        p = self.st.foot_positions
        self.liftoff_pos = p.copy()
        self.footholds = p.copy()
        self.forces = np.zeros((4, 3))
        self.mpc_result = None
        self.info = {}

    def get_state(self) -> dict:
        keys = ["k", "prev_contacts", "liftoff_pos", "footholds", "forces", "mpc_result"]
        return copy.deepcopy({k: getattr(self, k) for k in keys})

    def set_state(self, s: dict):
        for k, v in copy.deepcopy(s).items():
            setattr(self, k, v)

    # ------------------------------------------------------------ gait timing
    @property
    def t(self) -> float:
        return self.k * self.sim_dt

    def gait_time(self, t: float) -> float:
        """Time since the gait started (negative while standing)."""
        return t - self.cfg.t_stand

    def contacts_at(self, t: float) -> np.ndarray:
        tg = self.gait_time(t)
        return np.ones(4, dtype=bool) if tg < -1e-9 else self.gait.contacts(tg)

    def contacts_horizon(self, t: float) -> np.ndarray:
        h, dt = self.cfg.mpc.h, self.cfg.mpc.dt
        out = np.ones((h, 4), dtype=bool)
        tg = self.gait_time(t) + dt * np.arange(h)
        walking = tg > -1e-9
        if walking.any():
            first = int(np.argmax(walking))
            out[first:] = self.gait.contacts_horizon(tg[first], h - first, dt)
        return out

    def cycle_index(self) -> int:
        """Completed gait cycles (-1 while standing)."""
        tg = self.gait_time(self.t)
        return -1 if tg < -1e-9 else self.gait.cycle_index(tg)

    # ------------------------------------------------------------------ tick
    def step(self, v_cmd_b: np.ndarray) -> np.ndarray:
        """
        Compute and apply torques for the current MuJoCo state.
        v_cmd_b: [vx, vy, wz] command in the yaw (heading) frame.
        Call after mj_step1 (kinematics current), before mj_step2.
        Returns (4, 3) joint torques.
        """
        st, cfg, t = self.st, self.cfg, self.t
        v_cmd_b = np.asarray(v_cmd_b, dtype=float)

        # --- state
        pose = st.base_pose
        yaw = pose[5]
        Rz = rot_z(yaw)
        v_base = st.base_lin_vel_world
        w_base = st.base_ang_vel_world
        p_feet = st.foot_positions
        v_cmd_w = np.array([*(Rz[:2, :2] @ v_cmd_b[:2]), v_cmd_b[2]])

        # --- gait events
        contacts = self.contacts_at(t)
        liftoff = ~contacts & self.prev_contacts
        self.liftoff_pos[liftoff] = p_feet[liftoff]
        self.prev_contacts = contacts

        # --- foot placement
        tg = self.gait_time(t)
        elapsed = self.gait.swing_elapsed(tg) if tg >= -1e-9 else np.zeros(4)
        shoulder_b = self.params["foot_b"]
        for i in range(4):
            if contacts[i]:
                self.footholds[i] = p_feet[i]
                continue
            if not (liftoff[i] or cfg.update_foothold_in_swing):
                continue
            r_hip = Rz @ np.array([shoulder_b[i, 0], shoulder_b[i, 1], 0.0])
            t_sw = self.gait.swing_duration(i)
            self.footholds[i] = self.placement(PlacementInput(
                leg=i, p_hip=pose[:3] + r_hip, r_hip=r_hip,
                v_base=v_base, w_base=w_base, v_cmd=v_cmd_w,
                height=pose[2], yaw=yaw,
                t_stance=self.gait.stance_duration(i), t_swing=t_sw,
                t_swing_remaining=max(t_sw - elapsed[i], 0.0)))

        # --- MPC
        if self.k % self.mpc_every == 0:
            if cfg.centroidal_state:
                x0 = st.mpc_state_centroidal(self.params["I_body"], self.params["com_b"], GRAVITY)
            else:
                x0 = st.mpc_state(GRAVITY)
            Xref = ref_gen(x0, v_cmd_b, cfg.mpc)
            c_h = self.contacts_horizon(t)
            r_h = predict_foot_positions(self.gait, tg, c_h, p_feet, self.footholds, Xref,
                                         shoulder_b, self.params["com_b"],
                                         ground_z=p_terrain(None)[2])
            self.mpc_result = self.mpc.solve(x0, Xref, r_h, c_h)
            self.forces = self.mpc_result.forces[0]

        # --- swing references
        swing_ref = np.zeros((4, 3, 3))
        for i in range(4):
            if contacts[i]:
                swing_ref[i, 0] = p_feet[i]
            else:
                swing_ref[i] = foot_trajectory(elapsed[i], self.liftoff_pos[i],
                                               self.footholds[i], cfg.swing,
                                               t_swing=self.gait.swing_duration(i))

        # --- torques
        tau = self.legs.torques(contacts, self.forces, swing_ref)
        self.legs.apply(tau)

        self.info = dict(t=t, contacts=contacts, forces=self.forces.copy(),
                         footholds=self.footholds.copy(), swing_ref=swing_ref,
                         tau=tau, v_cmd_w=v_cmd_w)
        self.k += 1
        return tau
