from structs import ControllerCfg, MPCSol, FootPlacementCfg
from structs import INFO_KEYS, _PHASE_EPS, GRAVITY
from structs import rot_z
from gait import GaitScheduler
from mpc_controller import ConvexMPC, predict_foot_positions
from leg_controller import LegController
from ref_gen import ref_gen, ref_gen_periodic, PeriodicReference   # [CHANGED] + periodic reference
from swing_traj import foot_trajectory, p_terrain
from foot_place import FootPlacement
from state import State, robot_params
import mujoco
import numpy as np

# [REVIEW SUMMARY] fixes in this file (each marked [FIXED] below):
#   self.data = self.data, self.cfg.mpc, self.gait(t), self.t, self.leg/self.legs,
#   step counter named `step` (shadowed the step() method), MPC trigger used the
#   float time, inverted step_per_mpc override, ConvexMPC missing the inertia,
#   mpc_result None before the first solve, lift-off events/positions mixed up,
#   prev_contacts never updated, contacts_horizon row assignment, copy_into
#   direction, full (4,3) arrays passed to foot_trajectory, Jacobian not passed,
#   X_ref orientation, t_sim += 1 on a read-only property.

class QuadrupedController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 cfg: ControllerCfg, foot_placement: FootPlacement,
                 periodic_ref: PeriodicReference | None = None):   # [ADDED]
        self.model = model
        self.data = data                      # [FIXED] was self.data = self.data (AttributeError)
        self.cfg = cfg
        self.foot_placement = foot_placement
        self.periodic_ref = periodic_ref      # [ADDED] used when cfg.mpc_cfg.periodic_ref
        self.st = State(model, data)
        self.params = robot_params(model)

        self.sim_dt = model.opt.timestep
        self.step_per_mpc = int(round(self.cfg.mpc_cfg.dt / self.sim_dt))
        assert abs(self.step_per_mpc * self.sim_dt - self.cfg.mpc_cfg.dt) < 1e-9, \
            "mpc.dt must be a multiple of the sim timestep"    # [FIXED] self.cfg.mpc -> mpc_cfg

        # [FIXED] condition was `if self.step_per_mpc:` (always true), which
        # replaced the value with cfg.step_per_mpc = None.
        if self.cfg.step_per_mpc:
            self.step_per_mpc = self.cfg.step_per_mpc
        self.gait = GaitScheduler(self.cfg.gait_cfg)
        # [FIXED] ConvexMPC needs the inertia; [ADDED] gait period for the phase variable
        self.mpc = ConvexMPC(self.cfg.mpc_cfg, self.params["mass"], self.params["I_body"],
                             period=self.cfg.gait_cfg.period)
        self.leg = LegController(self.st, self.cfg.leg_cfg)
        if self.cfg.mpc_cfg.periodic_ref and self.periodic_ref is None:      # [ADDED]
            raise ValueError("mpc_cfg.periodic_ref=True needs a PeriodicReference")
        self.reset()

    def reset(self):
        # [FIXED] was self.step = 0: an int attribute named `step` hides the
        # step() method, so ctrl.step(v) raised "'int' object is not callable".
        self.n_step = 0   # time step count
        self.prev_contacts = np.ones(4, dtype=bool)
        p = self.st.foot_positions
        self.lift_off = p.copy()              # lift-off positions (4, 3)
        self.foothold = p.copy()
        self.forces = np.zeros((4,3))
        # [FIXED] was None: init_mpc_info() calls MPCSol.empty_like(None) and
        # copy_into(None) fails at the first solve.
        self.mpc_result = MPCSol.zeros(self.cfg.mpc_cfg.h)
        # [ADDED] gait clock offset [s]: gait time = t - t_stand - clock_offset.
        # Shifting it perturbs the gait phase (time-augmented Poincare / ROA).
        self.clock_offset = 0.0
        # [ADDED] index of the last MPC solve on the gait-clock grid (see step())
        self.mpc_tick = None
        self.info_mpc = self.init_mpc_info()
        self.info_leg = self.init_leg_info()

    def get_state(self, info_dict: dict):
        info_dict["step"] = self.n_step
        np.copyto(info_dict["prev_contacts"], self.prev_contacts)
        np.copyto(info_dict["lift_off"], self.lift_off)
        np.copyto(info_dict["foothold"], self.foothold)
        np.copyto(info_dict["forces"], self.forces)
        self.mpc_result.copy_into(info_dict["mpc_result"])
        info_dict["clock_offset"] = self.clock_offset     # [ADDED]
        info_dict["mpc_tick"] = self.mpc_tick             # [ADDED]
        return info_dict                                  # [ADDED] convenience

    def set_state(self, s: dict):
        self.n_step = s["step"]
        np.copyto(self.prev_contacts, s["prev_contacts"])
        np.copyto(self.lift_off, s["lift_off"])
        np.copyto(self.foothold, s["foothold"])
        np.copyto(self.forces, s["forces"])
        s["mpc_result"].copy_into(self.mpc_result)
        self.clock_offset = s.get("clock_offset", 0.0)    # [ADDED]
        self.mpc_tick = s.get("mpc_tick", None)           # [ADDED]

    def init_mpc_info(self) -> dict:
        return {
            "step": 0,
            "prev_contacts": np.empty_like(self.prev_contacts),
            "lift_off": np.empty_like(self.lift_off),
            "foothold": np.empty_like(self.foothold),
            "forces": np.empty_like(self.forces),
            "mpc_result": MPCSol.empty_like(self.mpc_result),
            "clock_offset": 0.0,                          # [ADDED]
            "mpc_tick": None,                             # [ADDED]
        }

    def init_leg_info(self) -> dict:
        return {
            "t": 0,
            "contacts": np.empty((4,), dtype=bool),
            "forces": np.empty_like(self.forces),
            "foothold": np.empty_like(self.foothold),
            "swing_ref": np.empty((4,3,3), dtype=float),
            "tau": np.zeros((4, 3)),                      # [CHANGED] array, copied each tick
            "v_cmd_w": np.empty((3,), dtype=float),
            "phase": 0.0,                                 # [ADDED]
        }

    @property
    def t_sim(self) -> float:
        return self.n_step * self.sim_dt

    def get_gait_time(self, t: float) -> float:
        return t - self.cfg.t_stand - self.clock_offset   # [CHANGED] + clock_offset

    # ---- [ADDED] begin ----
    def get_phase(self, t: float | None = None) -> float:
        """Unwrapped gait phase [cycles] (negative while standing)."""
        t = self.t_sim if t is None else t
        return self.get_gait_time(t) / self.cfg.gait_cfg.period

    def shift_phase(self, d_phase: float) -> float:
        """
        Perturb the gait clock by d_phase cycles (positive = clock ahead).
        Rounded to whole sim steps: a sub-step shift changes where the gait /
        MPC grid falls between sim steps and leaves a small permanent offset.
        Returns the phase shift actually applied.
        """
        n = round(d_phase * self.cfg.gait_cfg.period / self.sim_dt)
        self.clock_offset -= n * self.sim_dt
        return n * self.sim_dt / self.cfg.gait_cfg.period
    # ---- [ADDED] end ----

    def get_contacts(self, t: float) -> np.ndarray:
        t_gait = self.get_gait_time(t)
        # [FIXED] self.gait(t_gait) -> self.gait.contacts(t_gait) (GaitScheduler is not callable)
        return np.ones(4, dtype=bool) if t_gait < -_PHASE_EPS else self.gait.contacts(t_gait)

    def get_contacts_horizon(self, t: float) -> np.array:
        h = self.cfg.mpc_cfg.h
        dt = self.cfg.mpc_cfg.dt                          # [FIXED] self.cfg.mpc -> mpc_cfg
        contacts_horizon = np.ones((h, 4), dtype=bool)
        ts_gait = self.get_gait_time(t) + dt * np.arange(h)
        contacts_flag = ts_gait > -_PHASE_EPS
        if contacts_flag.any():
            contacts_start = int(np.argmax(contacts_flag))
            # [FIXED] was contacts_horizon[contacts_start] = (h - start, 4) block
            # -> shape error; the block fills rows start .. h-1.
            contacts_horizon[contacts_start:] = self.gait.contacts_horizon(ts_gait[contacts_start], h - contacts_start, dt)
        return contacts_horizon

    def get_cycle_index(self) -> int:
        t_gait = self.get_gait_time(self.t_sim)           # [FIXED] self.t does not exist
        return -1 if t_gait < -1e-9 else self.gait.cycle_index(t_gait)

    def step(self, v_cmd_b: np.ndarray) -> np.ndarray:
        st = self.st
        t_sim = self.t_sim
        cfg = self.cfg
        v_cmd_b = np.asarray(v_cmd_b, dtype=float)        # [ADDED] lists/tuples
        base_pose = st.base_pose
        yaw = base_pose[5]
        Rz = rot_z(yaw)
        v_base = st.base_lin_vel_world
        w_base = st.base_ang_vel_world
        p_foot = st.foot_positions
        J = st.foot_jacobians                             # [ADDED] LegController.torques needs J
        v_cmd_w = np.array([*(Rz[:2, :2] @ v_cmd_b[:2]), v_cmd_b[2]])

        contacts = self.get_contacts(t_sim)
        # [FIXED] lift-off events: was written into self.liftoff (undefined) and
        # mixed with the lift-off *positions* self.lift_off; positions were never
        # recorded and prev_contacts never updated, so swing legs started their
        # trajectory from the initial foot position forever.
        liftoff = ~contacts & self.prev_contacts
        self.lift_off[liftoff] = p_foot[liftoff]
        self.prev_contacts = contacts

        t_gait = self.get_gait_time(t_sim)
        elapsed = self.gait.swing_elapsed(t_gait) if t_gait >= -_PHASE_EPS else np.zeros(4)
        p_hip_b = self.params["foot_b"]
        for i in range(4):
            if contacts[i]:
                self.foothold[i] = p_foot[i]
                continue
            if not (liftoff[i] or cfg.update_foothold_in_swing):   # [FIXED] used positions as a bool
                continue
            r_hip = Rz @ np.array([p_hip_b[i,0], p_hip_b[i,1], 0.])
            t_swing = self.gait.swing_duration(i)
            self.foothold[i] = self.foot_placement(
                FootPlacementCfg(
                leg=i, p_hip=base_pose[:3] + r_hip, r_hip=r_hip,
                v_base=v_base, w_base=w_base, v_cmd=v_cmd_w,
                height=base_pose[2], yaw=yaw,
                t_stance=self.gait.stance_duration(i), t_swing=t_swing,
                t_swing_remaining=max(t_swing - elapsed[i], 0.0))
            )

        terrain_z = p_terrain(None)[2]

        # [FIXED] was `self.t_sim % self.step_per_mpc == 0`: t_sim is a float
        # time, so this was true only at t = 0 (and at whole seconds).
        # [CHANGED] solve on the gait-clock grid (every step_per_mpc * dt of gait
        # time), not on the raw step counter. Same behaviour when clock_offset = 0,
        # but a gait phase shift now moves the MPC sampling with the contact
        # schedule. With the counter, a shift misaligned solves and contact
        # switches and the robot settled on a *different* limit cycle (a neutral
        # direction, eigenvalue ~1, in the time-augmented return map).
        solve_dt = self.step_per_mpc * self.sim_dt
        tick = int(np.floor(t_gait / solve_dt + 1e-6))
        if tick != self.mpc_tick:
            self.mpc_tick = tick
            if cfg.centroidal_state:
                x0 = st.mpc_state_centroidal(self.params["I_body"], self.params["com_b"], GRAVITY)
            else:
                x0 = st.mpc_state(GRAVITY)
            phase0 = self.get_phase(t_sim)                                   # [ADDED]
            if cfg.mpc_cfg.periodic_ref:                                     # [ADDED]
                X_ref = ref_gen_periodic(x0, phase0, v_cmd_b, cfg.mpc_cfg, self.periodic_ref)
            else:
                X_ref = ref_gen(x0, v_cmd_b, cfg.mpc_cfg)
            X_ref = X_ref.T   # [FIXED] ref_gen returns (13, h+1); ConvexMPC.solve asserts (h+1, 13)
            contacts_horizon = self.get_contacts_horizon(t_sim)
            foot_prediction = predict_foot_positions(
                self.gait, t_gait, contacts_horizon, p_foot, self.foothold,
                X_ref, p_hip_b, self.params["com_b"], terrain_z
            )
            # [FIXED] was MPCSol.copy_into(self.mpc_result, new) = copy the OLD
            # result into the new one; the controller never saw a new solution.
            sol = self.mpc.solve(x0, X_ref, foot_prediction, contacts_horizon,
                                 phase0=phase0)                              # [CHANGED] + phase0
            sol.copy_into(self.mpc_result)
            np.copyto(self.forces, self.mpc_result.forces[0])   # [CHANGED] copy, not a view into mpc_result

        swing_ref = np.zeros((4,3,3))
        for i in range(4):
            if contacts[i]:
                swing_ref[i,0] = p_foot[i]
            else:
                # [FIXED] passed the full (4, 3) arrays lift_off / foothold; one leg's row is needed
                swing_ref[i] = foot_trajectory(
                    elapsed[i], self.lift_off[i], self.foothold[i], cfg.swing_cfg, self.gait.swing_duration(i)
                )

        tau = self.leg.torques(contacts, self.forces, swing_ref, J)   # [FIXED] self.legs -> self.leg, + J
        self.leg.command(tau)
        self.info_leg["t"] = t_sim
        np.copyto(self.info_leg["contacts"], contacts)
        np.copyto(self.info_leg["forces"], self.forces)
        np.copyto(self.info_leg["foothold"], self.foothold)
        np.copyto(self.info_leg["swing_ref"], swing_ref)
        np.copyto(self.info_leg["tau"], tau)              # [CHANGED] copy into the preallocated array
        np.copyto(self.info_leg["v_cmd_w"], v_cmd_w)
        self.info_leg["phase"] = self.get_phase(t_sim)    # [ADDED]
        self.n_step += 1   # [FIXED] was self.t_sim += 1 (read-only property, and it is a time, not a count)

        return tau
