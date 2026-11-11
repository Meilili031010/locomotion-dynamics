# Go2 trot: Cheetah 3 style convex MPC + Poincare / ROA analysis

Run from this folder with the project venv (needs `osqp`):

```bash
cd loco_proj/sim
../.venv/bin/python run.py --vx 0.5 --viewer
../.venv/bin/python poincare_aug.py --vx 0.5 --compare raibert:k=0.03 raibert:k=0.17 openloop
../.venv/bin/python roa.py --vx 0.5 --method grid --dims vx vy --bounds 1 1 --n 11
```
Each program has `--help` with examples. Outputs go to `results/`.

## Files

```
structs.py         all configs (MPC_Cfg, GaitCfg, ControllerCfg, SimCfg, ...) + SRB dynamics
gait.py            GaitScheduler: contacts, swing timing
foot_place.py      foot placement policies (openloop, Raibert, lipm) + POLICIES registry
swing_traj.py      Cheetah 3 footstep terms, Bezier swing trajectory
ref_gen.py         constant-velocity reference, PeriodicReference, ref_gen_periodic
mpc_controller.py  ConvexMPC (phase variable, period constraint), foot prediction
leg_controller.py  stance -J^T f + bias + friction comp; swing Cartesian PD + feed-forward
quad_ctrl.py       QuadrupedController: wires everything, snapshot get_state/set_state
state.py           MuJoCo state access, robot SRB parameters
sim_env.py         model from SimCfg, sim loop, logging, snapshots, pushes
experiment.py      Experiment = SimCfg + ControllerCfg + policy + command; shared CLI
run.py             one simulation (settings, pushes, plots, record a periodic reference)
poincare_aug.py    time-augmented Poincare map (touchdown or clock section)
roa.py             sample-based region of attraction (grid / random / radial)
```

## Changing settings

Any config field: `--set section.field=value` (sections `sim mpc gait swing leg ctrl exp`),
or a JSON file with the same sections (`run.py --dump-config x.json` writes one).
Shortcuts for the simulator:

| flag | meaning |
|---|---|
| `--stiffness K --damping B` | contact solref = (-K, -B), "direct" mode |
| `--solref TC DR` | contact solref = (timeconst, dampratio); keep TC >= 2 dt |
| `--solimp a b c d e` | contact solimp |
| `--friction MU` | floor + feet sliding friction (MPC mu follows: MU/sqrt(2), `exp.mu_auto`) |
| `--condim --cone --impratio --integrator --dt` | MuJoCo options |
| `--obstacles` | keep the bars/stairs of the stock scene (removed by default) |
| `--set sim.mass_scale=1.2 sim.torque_scale=0.8` | robot changes |

## MPC: phase variable, period constraint, periodic reference

* Phase `phi_k = phi_0 + k dt / T` is carried along the horizon (`MPCSol.phase`).
  It does not depend on the forces, so it is not a QP variable; it selects the
  periodic reference and the constrained step pairs.
* Period constraint `x_{k+N} = x_k` (+ reference shift for x, y, yaw; v/w rotated
  by the reference yaw change), N = T/dt. Needs `mpc.h >= N + period_k0` (h=30).
  * `mpc.period_constraint`: `none` | `soft` (weight `period_weight`) | `hard`
    (falls back to soft if infeasible)
  * `mpc.period_pairs`: `last` (x_h = x_{h-N}: last period of the plan is a closed
    orbit), `section` (pair at a gait-phase boundary), `all` (every step: over-
    determined, usually infeasible as hard; kept for comparison)
* Periodic reference: `run.py --record-ref results/ref.npz` records the robot's own
  cycle; `--periodic-ref results/ref.npz` tracks it. `mpc.periodic_ref_terms`
  picks the tracked terms; `("z","v")` is recommended (tracking recorded
  roll/pitch made the MPC re-create leg-induced body motion: pitch doubled).

## Time-augmented Poincare map

State `(x, phi)`, phi = gait clock phase, so the closed loop is autonomous.
Section `touchdown` (default) = actual FL ground contact: `dphi` (clock phase at
touchdown) is a coordinate and the return time `tau` is reported. Section
`clock` = `phi = 0`, the stroboscopic map. Perturbations: 6 base velocity
components + clock shift. Eigenvalues come from a least-squares fit in reduced
coordinates (legs' joint states are not in the section); compare policies under
identical settings, and treat absolute values as estimates (clock and touchdown
sections give different numbers, e.g. 0.32 vs 0.47).

## Region of attraction

Perturbations at the section, `--cycles` crossings, classified converged / not
converged / fell. `grid` (2-D slice), `random` (Sobol in a box), `radial`
(bisection along random directions; assumes the region is star-shaped along each
direction).

## Marking convention

`[FIXED]` bug fix, `[CHANGED]` changed behaviour/value, `[ADDED]` new code
(blocks in `---- [ADDED] begin/end ----`), `[REVIEW NOTE]` observation without a
change. Files headed `[NEW FILE]` were written from scratch.
