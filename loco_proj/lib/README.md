# lib: reference libraries for the sim/ program

Four reference files, one per part of `sim/`. Each one explains the tools, the
math and the pitfalls, with small runnable examples on toy problems (a planar
rigid body, a one-leg robot), so you can write the Go2 code in `sim/` yourself.

| file | covers | for writing |
|---|---|---|
| `sim_interface_lib.py` | MuJoCo model/data, names -> indices, options, contact params, state layout, Jacobians, mass matrix, mj_step1/2, snapshots, pushes, rotations, viewer, logging | `sim_env.py`, `state.py` |
| `mpc_lib.py` | SRB dynamics, skew, discretization (ZOH), lifted prediction, cost -> QP, friction pyramid, swing-leg masking, OSQP + scipy.sparse, hard/soft constraints, receding horizon | `mpc_controller.py`, dynamics in `structs.py` |
| `gait_lib.py` | gait clock/phase, contact table, durations, events, contact horizon, standing start, frames, reference trajectories (constant / periodic), foot placement, Bezier swing, policy registry | `gait.py`, `ref_gen.py`, `foot_place.py`, `swing_traj.py` |
| `controller_lib.py` | stance -J^T f, gravity comp, swing PD + feed-forward, friction comp, limits + actuator order, multi-rate MPC timing, the order of one control step, controller state, config dataclasses, debug checklist | `leg_controller.py`, `quad_ctrl.py` |

Run (needs `mujoco`, `numpy`, `scipy`, `osqp`):

```bash
python sim_interface_lib.py        # every example
python controller_lib.py C D       # only some sections
```
