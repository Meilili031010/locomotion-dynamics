# Go2 trot: Cheetah 3 style convex MPC + Poincare analysis

Run everything from this folder with the project venv:

```bash
cd loco_proj/sim
../.venv/bin/python run_trot.py --vx 0.5             # headless, 8 s, prints summary
../.venv/bin/python run_trot.py --vx 0.5 --viewer    # MuJoCo viewer
../.venv/bin/python run_trot.py --vx 0.6 --wz 0.5 --policy raibert --k 0.17
../.venv/bin/python compare_policies.py              # Poincare comparison -> results/
../.venv/bin/python compare_policies.py --vcmd 0.6 0 0.5 --extended
```

Needs `osqp` (installed into `.venv`: `pip install osqp`).

## Pipeline (one controller tick = one 1 ms sim step)

```
gait.py            contacts, swing timing             (GaitScheduler)
foot_placement.py  footholds for swing legs           (policy objects; swap these)
ref_gen.py         13-state reference over horizon    (MPC_Config)
mpc_solver.py      convex MPC -> GRFs, every 25 ms    (ConvexMPC, predict_foot_positions)
swing_traj.py      Bezier swing trajectories + Cheetah 3 footstep terms
leg_controller.py  stance: -J^T f ; swing: Cartesian PD + feed-forward
controller.py      wires the above                    (ControllerConfig, LocomotionController)
state.py           MuJoCo state access, robot SRB parameters
sim_env.py         model (flat floor), sim loop, logging, snapshots
poincare.py        section, fixed point, return-map Jacobian, eigenvalues
compare_policies.py  runs poincare.analyze over a set of policies, saves npz + png
```

Add a foot placement approach: subclass `FootPlacementPolicy` in
`foot_placement.py`, implement `_raw(inp) -> foothold`, register it in
`POLICIES`, add it to `POLICY_SET` in `compare_policies.py`.

## Marking convention in the original files

`gait.py`, `ref_gen.py`, `state.py`, `swing_traj.py`, `step_interpolator.py`:

* `# [FIXED]`   a bug fix, with the reason
* `# [CHANGED]` a changed value/behaviour, with the reason
* `# [ADDED]`   new code; larger blocks are wrapped in
  `# ---- [ADDED] begin ----` / `# ---- [ADDED] end ----`

Files whose header says `[NEW FILE]` / `[NEW CONTENT]` were written from scratch
(`mpc_solver.py` kept its original imports).

## Deviations from Cheetah 3 (and why)

* **MPC state from whole-robot momentum** (`ControllerConfig.centroidal_state=True`).
  Go2 legs are ~45% of the mass; swinging legs kick the base velocity around
  while CoM velocity / angular momentum only change through ground forces.
  `False` gives the paper's base-link state (works, but tracks speed worse:
  0.60 vs 0.86 m/s at a 1.0 m/s command).
* **Composite inertia** of all bodies about the CoM (`state.robot_params`).
* **Leg bias, joint friction/damping compensation, swing feed-forward**
  (`LegControlConfig`). Without them the real ground force missed the MPC force
  by ~20 N vertically and ~4 N/leg horizontally.
* **Friction pyramid mu = 0.28** = 0.4/sqrt(2): the model's foot friction is 0.4
  on a round cone.

## Poincare section

Section = touchdown of FL/RR (start of each cycle). Coordinates
`[z, roll, pitch, vx, vy, vz, wx, wy, wz]` in the heading frame
(`extended=True` adds FL/RR foot xy relative to the base). Fixed point = mean of
the last cycles after settling; the Jacobian is fitted by least squares from
randomly pushed trajectories (eigenvalues), and also estimated column-wise by
central differences on the velocity coordinates. The fit leaves ~15% residual
(the legs' joint state is not in the section), so treat eigenvalues as
estimates; compare policies under identical settings.
