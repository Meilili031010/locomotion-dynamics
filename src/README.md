# Go2 convex-MPC trot: program documentation

A MuJoCo simulation of the Unitree Go2 trotting with the MIT Cheetah 3 convex MPC. Foot placement policies and gaits are pluggable, and energy is measured so methods can be compared.

## 1. Run it

Python with `mujoco`, `numpy`, `scipy`, `osqp` (the `unitree_py310` conda env has them; add `matplotlib` for `--plot`). Run from `src/`:

```bash
python run.py --vx 0.5 --time 8                            # Raibert trot, prints summary + energy
python run.py --policy force_aligned --time 8              # plan-aware foot placement
python run.py --set mpc.align_weight=0.001 ctrl.hip_point=abduction   # leg-axis force cost in the MPC
python run.py --viewer                                     # on macOS: use mjpython instead of python
python Gait_opt.py                                         # cost of trot / pace / bound / walk
```

Every config field can be changed with `--set section.field=value` (sections `sim mpc gait swing leg ctrl exp`) or a JSON file (`--config`, write one with `--dump-config`). Policy arguments: `--policy NAME --k 0.03 --policy-kw "{'w_reg': 0.1}"`.

## 2. Files

| file | role |
|---|---|
| `Def.py` | constants, every config dataclass, result/log containers (`MPCSol`, `MPCPlan`, `SimLog`, `PeriodicReference`, `ForcePush`) |
| `Utils.py` | math: `skew`, `rot_z`, SRB `continuous_dynamics`, `discretize` (exact ZOH) |
| `Sim.py` | MuJoCo model from `SimCfg`, reset, snapshots, pushes, fall check, the `run()` loop |
| `Interface.py` | `Interface`: robot state in LEGS order, Jacobians, torque output; `robot_params()` (mass, composite inertia, CoM, nominal feet) |
| `Gait.py` | `GaitScheduler` (table or duty/offsets), Cheetah 3 placement terms, Bezier swing `foot_trajectory` |
| `Reference_traj.py` | MPC reference: constant velocity `ref_gen`, periodic `ref_gen_periodic` |
| `Foot_placement.py` | policies `openloop`, `Raibert`, `lipm`, `ForceAligned` and the `POLICIES` registry |
| `Controller.py` | `ConvexMPC`, `predict_foot_positions`, `LegController`, `QuadrupedController` |
| `Metrics.py` | `EnergyMeter` (CoT, work, stance torques, force-leg angle), `evaluate(exp)` |
| `Gait_opt.py` | gait parameter vector ↔ experiment, `objective(x)`; the optimizer slot is empty on purpose |
| `Cli.py` | `Experiment` (sim + controller + policy + command), CLI, JSON |
| `run.py` | one run: summary, energy, plot, save log, record a periodic reference |

## 3. Conventions

| item | convention |
|---|---|
| leg order | `LEGS = FL, FR, RL, RR`; joints `hip` (abduction), `thigh` (pitch), `calf` (knee) |
| actuator order | xml: FR, FL, RR, RL. Only `Interface.sim_ctrl` writes `data.ctrl`, through `act_idx`. |
| MPC state (13) | `[p(3), roll, pitch, yaw, v(3), ω(3) world, g]`, g = −9.81 as a state |
| MPC input (12) | ground force on the robot per leg, world frame, LEGS order |
| `Interface.base_v_w` | `[v_world, ω_world]` (linear first) |
| `Interface.foot_jac` | (4, 6, nv), rows `[linear; angular]` |
| `X_ref` | `ref_gen` returns (13, h+1); the controller transposes it to (h+1, 13); row 0 = x0 |
| time | `t_gait = t_sim − t_stand − clock_offset`; negative = standing (all stance) |
| phase | `t_gait / period` (unwrapped cycles) |

## 4. One control step (`QuadrupedController.step`)

It is called between `mj_step1` and `mj_step2`, every sim step (1 ms):

1. Read the state: base pose and velocity, foot positions, foot Jacobians.
2. Contacts now from the gait clock. Lift-off events store the lift-off point.
3. `_update_footholds`: stance foot → foothold = foot; swing foot → `policy(FootPlacementCfg)`. The config carries the last MPC plan (`plan`) and the leg's hip point (`hip_b`).
4. Every MPC step (25 ms, on the gait-clock grid), build and solve the MPC:
   - inputs: x0 (centroidal by default), `X_ref`, (h, 4) contact horizon, predicted foot positions `r_foot`, hip points `r_hip`;
   - `ConvexMPC.solve` returns `MPCSol`, stored as `self.plan` (`MPCPlan`);
   - with `placement_iters > 0` and a plan-aware policy, it repeats: footholds → MPC → plan;
   - the first force row is held until the next solve.
5. Swing references: Bezier from the lift-off point to the foothold (pos/vel/acc).
6. `LegController.torques`:
   - stance: `−Jᵀf + bias + friction`;
   - swing: `Jᵀ(Kp e + Kd ė) + M J⁻¹a + bias + friction`;
   - clipped, then written to `data.ctrl`.

## 5. Foot placement: adding a method

A policy is a callable `FootPlacementCfg -> (3,) world foothold`.

1. Subclass `FootPlacement` and implement `_raw(cfg)`. The base `__call__` clamps the stride to `max_stride` around the hip at touchdown and sets z on the terrain.
2. If it reads the MPC plan, set `uses_plan = True` and use `cfg.plan` (`MPCPlan`):
   - `contacts` (h, 4)
   - `forces` (h, 4, 3)
   - `X_pred` (h, 13), where row k is x_{k+1}
   - `next_stance_rows(leg)`: the rows of that leg's next stance
   - `cfg.plan` is `None` before the first solve.
3. Register it in `POLICIES`. It is then usable as `--policy NAME`, and its arguments come from `--k` / `--policy-kw`.

Also available in `FootPlacementCfg`: `p_hip` and `r_hip` (shoulder point), `v_base`, `w_base`, `v_cmd` (world), `height`, `yaw`, `t_stance`, `t_swing`, `t_swing_remaining`, `hip_b`.

### 5.1 `ForceAligned` (leg force along the hip–foot line)

Idea: a ground force along the leg makes no moment about the hip, so the hip motors do less work. For the leg's next planned stance (rows k), with predicted hip point `h_k` and planned force direction `n_k = f_k/|f_k|`:

```
min_{p_xy}  Σ_k w_k ||(I − n_k n_kᵀ)(h_k − p)||²  +  w_reg ||p − p_nom||²,   w_k = fz_k / Σ fz
```

`||(I − nnᵀ)(h − p)||` is the distance from the foot to the force line through the hip. With z fixed this is a 2×2 linear system. `p_nom` comes from a base policy (`base`, `k`).

### 5.2 MPC alignment cost (`mpc.align_weight`)

Foot placement alone cannot align the force, because the **MPC** picks the force direction after the foot is placed (it re-tilts the force to track velocity). The MPC can therefore include the alignment directly:

```
J += align_weight · Σ_{k, stance i} ||(I − l_ki l_kiᵀ) f_ki||²,   l = unit(hip point − foot)
```

This is still quadratic in the forces, so the problem stays a convex QP (`ConvexMPC._align_cost`). `ctrl.hip_point` selects the hip point:
- `thigh`: the pitch joint, above the foot;
- `abduction`: the abduction axis, 9.5 cm inwards.

### 5.3 Results so far

Trot at 0.5 m/s command, 8 s run, energy measured over the last 5.5 s, Raibert k = 0.03 unless noted. "hip/thigh/calf" = mean |τ| of stance legs [N·m].

| setting | speed | CoT_abs | force-leg angle | hip | thigh | calf |
|---|---|---|---|---|---|---|
| Raibert (baseline) | 0.443 | 0.637 | 7.1° | 5.53 | 1.75 | 13.53 |
| `force_aligned`, w_reg 0.1 | 0.450 | 0.708 | 8.2° | 5.65 | 2.41 | 13.91 |
| `force_aligned`, w_reg 0 | 0.443 | 0.784 | 8.9° | 5.77 | 2.66 | 13.96 |
| MPC align 1e-3, thigh point | 0.389 | 0.722 | 3.0° | 6.22 | 1.41 | 13.65 |
| MPC align 1e-2, thigh point | 0.365 | 0.748 | 1.7° | 6.38 | 1.14 | 13.50 |
| **MPC align 1e-3, abduction point** | 0.417 | **0.586** | 11.4°* | **2.32** | 1.84 | 13.59 |
| MPC align 1e-2, abduction point | 0.197 | 3.497 | 20.7°* | 3.56 | 1.01 | 13.61 |
| `force_aligned` + MPC align 0.1 | fell | | | | | |

\* measured against the abduction-point line.

What the table shows:
- Placement-only alignment does not work: the MPC undoes it.
- The MPC cost does align the force, which lowers the torque of the joint whose axis the line passes through. It costs speed and moves load to the other hip joint.
- Aligning with the abduction axis at a small weight is the one setting that lowered CoT (−8%), mainly by cutting abduction torque 58%.
- The calf (knee) carries most of the torque in every case, and no alignment changes it. Leg-length support stays expensive.

Open points worth testing: speed-matched comparisons (raise the command until speeds match), longer runs, a sweep of `align_weight` between 3e-4 and 3e-3, other speeds, and turning.

## 6. Gaits and gait optimization

`GaitCfg` has two forms:
- **table** (default trot): rows = sub-phases, nonzero entry = swing, with the value = row end phase. All-stance rows are not possible.
- **parametric**: `duty` (stance fraction) + `offsets` (lift-off phase per leg). `--set gait.duty=0.75 "gait.offsets=(0,0.5,0.75,0.25)"`. Trot = duty 0.5, offsets (0.5, 0, 0, 0.5).

Rule: every lift-off/touchdown time must be a multiple of the MPC step (`mpc.dt` = 25 ms), or contacts switch between MPC steps. The controller prints a warning; `Gait_opt.feasible` rejects such gaits. Also keep `mpc.h · mpc.dt` ≈ one period (`apply_params` sets `h = period / dt`).

`Gait_opt.py`:
- `GAIT_PARAMS` = `[period, duty, offset_FR, offset_RL, offset_RR, step_height]` with bounds; FL lifts off at 0.
- `objective(x, exp)` → `(CoT_abs + 5·|speed error|/v_cmd, metrics)`; a fall or infeasible gait gives 100.
- The optimizer slot is marked in the file.

A geometric (Hatton-style) method would use a model-based cost over a closed curve in shape space and use `objective()` only to verify. A black-box method (grid, Bayesian, CMA-ES) can call `objective()` directly. One evaluation is one 6 s simulation.

Current numbers at a 0.5 m/s command:

| gait | cost | speed | CoT_abs |
|---|---|---|---|
| trot | 1.21 | 0.443 | 0.638 |
| walk (period 0.8, duty 0.75) | 1.55 | 0.484 | 1.39 |
| bound | 3.02 | 0.426 | 2.28 |
| pace | fell | | |

## 7. Metrics (`Metrics.EnergyMeter`)

Integrated every sim step from `t_measure` (default: standing + ramp + 1 s):
- `E_abs = Σ|τ q̇|dt` and `E_pos = Σ max(τ q̇, 0)dt`; CoT = E / (m g d);
- `tau_sq_rate` = Σ τ² per second (Joule-heating proxy);
- stance mean |τ| per joint type;
- `stance_align_deg` = fz-weighted angle between the planned force and the foot → hip-point line.

## 8. Fixes made in this version

| file | fix |
|---|---|
| `Def.py` | `PATH` used the current folder, so the model path broke when starting from `src/`; it now uses the repo root |
| `Interface.py` | `base_v_w` returned `[ω, v]` but was read as `[v, ω]`, which gave wrong foot placement and a wrong base-link MPC state |
| `Interface.py` | `jnt_vel` read `qpos` (friction compensation used angles as speeds) |
| `Interface.py` | `foot_jac` rows were `[angular; linear]`, so stance torques used the rotational Jacobian; this made the robot fall while standing |
| `Interface.py` | `com_pos` (`self.data`), `hip_w` (broadcast), `mpc_state(G)` (signature), `sim_ctrl` (`reshape.ravel`), `raise NotImplemented` |
| `Interface.py` | `R_bw` renamed `R_wb` (it is body → world); `jnt_fric` renamed `jnt_tau` (it is the applied torque); `foot_pos` returns a copy |
| `Controller.py` | `LegController` used `st.model`/`st.data`, which crashed at the first swing; MPC fallback used undefined `GRAVITY` |
| `Gait.py` | centrifugal term got g = −9.81, so `sqrt` gave NaN footholds for the `lipm` policy |
| `Utils.py` | removed broken copies of the `Sim.py` functions (undefined names, never used) |

New: `MPCPlan`, plan-aware placement, `ForceAligned`, `mpc.align_weight`, `ctrl.hip_point`, `ctrl.placement_iters`, parametric gaits, `Metrics.py`, `Gait_opt.py`, `--policy-kw`, `--measure-from`, plan included in controller snapshots.

## 9. Known limitations

- Forward speed settles about 11% below the command (0.443 vs 0.5 m/s) with every policy. Possible causes are the MPC weights (Q on v is 10 vs 50 on z) and leg dynamics the SRB model ignores. Compare methods at matched speeds.
- `ForceAligned` uses the plan from the previous MPC step unless `placement_iters > 0` (each iteration is one more QP).
- The hip point is fixed in the base frame; roll/pitch of the predicted body are ignored (yaw only, like the MPC model).
- `Interface.sim_detect_contact` / `sim_get_contact_force` are still stubs (yours to write). They would let you compare the real contact timing and forces with the plan.
- Pace falls with the current MPC weights; gaits other than trot have not been tuned.
