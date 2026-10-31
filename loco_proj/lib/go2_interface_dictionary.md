# Go2 MuJoCo interface dictionary

A lookup sheet for rewriting the simulator interface (`sim/sim_env.py` + `sim/state.py`).

- **Part 1:** Go2 model facts (names, indices, limits), from `go2_model_info.txt`
- **Part 2:** MuJoCo python API dictionary, grouped by task
- **Part 3:** reminder list of the functions your interface needs (names and contracts only, no code)

Checked against `mujoco` 3.17. For runnable examples of most entries, see `sim_interface_lib.py` (section letters in brackets, e.g. `[G]`).

---

## Part 1: Go2 model facts

### Sizes

| quantity | value | note |
|---|---|---|
| `model.nq` | 19 | 7 (free joint: xyz + quaternion) + 12 joints |
| `model.nv` | 18 | 6 (free joint) + 12 joints |
| `model.nu` | 12 | one torque motor per joint |
| `model.nbody` | 18 | world, base_link, 4 × (hip, thigh, calf, foot) |
| total mass | 15.21 kg | `model.body_subtreemass[base_id]`; weight ≈ 149 N |
| legs ≈ 45% of the mass | base_link is 6.92 kg | why `sim/` uses the composite inertia + centroidal state |

### Bodies

| id | name | parent | note |
|---|---|---|---|
| 0 | world | | |
| 1 | `base_link` | world | carries the free joint and site `imu` |
| 2–5 | `FL_hip`, `FL_thigh`, `FL_calf`, `FL_foot` | chain | `FL_foot` body has zero mass |
| 6–9 | `FR_*` | | |
| 10–13 | `RL_*` | | |
| 14–17 | `RR_*` | | |

### Joints (body order is FL, FR, RL, RR)

| joint | qposadr | dofadr | axis | range [rad] |
|---|---|---|---|---|
| free joint (unnamed) | 0–6 | 0–5 | | |
| `FL_hip_joint` / `FL_thigh_joint` / `FL_calf_joint` | 7, 8, 9 | 6, 7, 8 | x / y / y | ±1.047 / −1.571…3.491 / −2.723…−0.838 |
| `FR_*_joint` | 10, 11, 12 | 9, 10, 11 | same | same |
| `RL_*_joint` | 13, 14, 15 | 12, 13, 14 | same | thigh −0.524…4.538 |
| `RR_*_joint` | 16, 17, 18 | 15, 16, 17 | same | thigh −0.524…4.538 |

Every leg joint has damping 0.1, armature 0.01 and frictionloss 0.2 (`dof_damping`, `dof_armature`, `dof_frictionloss`).

The rule is `qposadr = dofadr + 1` for every leg joint (the quaternion has 4 numbers, angular velocity has 3). Don't hard-code it; read it from `jnt_qposadr` and `jnt_dofadr`.

### Actuators: order is FR, FL, RR, RL, not the body order

| actuator id | 0 1 2 | 3 4 5 | 6 7 8 | 9 10 11 |
|---|---|---|---|---|
| leg | FR | FL | RR | RL |
| joints | hip, thigh, calf | | | |
| ctrlrange [N·m] | ±23.7, ±23.7, ±45.43 | | | |

All are motors with gear 1, so `ctrl` is the joint torque. Build the joint→actuator map from `model.actuator_trnid[:, 0]`.

### Geoms that matter

| geom | type | body | values |
|---|---|---|---|
| `floor` | plane | world | added by the scene / `make_model` |
| `FL`, `FR`, `RL`, `RR` | sphere r = 0.022 | `*_calf` | friction (0.4, 0.02, 0.01). These are the feet. |
| other collision geoms | box/cylinder/sphere | base, thighs, calves | can touch the floor when the robot falls |
| visual meshes | type 7 (mesh) | all | `contype = conaffinity = 0`, never collide |

On flat ground the foot sphere center rests at z = 0.022. In the default solref/solimp it sinks a fraction of a millimetre.

### Options and keyframe

| item | value in go2.xml | `sim/` setting |
|---|---|---|
| timestep | 0.002 | `SimCfg.timestep = 0.001` |
| integrator | 0 (Euler) | `SimCfg.integrator` |
| cone | elliptic | the MPC friction pyramid uses μ/√2 |
| impratio | 100 | |
| iterations | 100 | |
| site | `imu` on base_link | |
| home pose (dump) | base z 0.27, joints (0, 0.9, −1.8) × 4 | `reset_robot` uses keyframe 0 of the xml, then lifts the base so the feet touch |

Contact friction is the max of the two geoms unless `geom_priority` differs. In the dump the floor had friction 1.0, so contact friction was 1.0, not 0.4. Always check this (`foot_friction()` in Part 3).

Sensors: the dump did not list any. Check `model.nsensor`. `sim/` reads `qpos`/`qvel` directly, which is perfect state; sensors only matter if you want to simulate state estimation.

---

## Part 2: MuJoCo API dictionary

Shapes are for Go2. `m` = `MjModel`, `d` = `MjData`.

### 2.1 Load / build the model  `[A]`

| API | returns | use |
|---|---|---|
| `mujoco.MjModel.from_xml_path(path)` | MjModel | load `go2.xml` / `scene.xml` |
| `mujoco.MjModel.from_xml_string(xml)` | MjModel | test models |
| `mujoco.MjSpec.from_file(path)` | MjSpec | editable model before compile |
| `spec.worldbody.geoms` | list | find stairs/bars of the stock scene |
| `spec.worldbody.add_geom()` | geom spec | add a floor (`dump_model.py` does this) |
| `spec.delete(elem)` | | remove elements |
| `spec.compile()` | MjModel | |
| `mujoco.MjData(m)` | MjData | one per simulation; a scratch one for side computations |
| `mujoco.mj_saveLastXML(path, m)` | | save the edited model, to inspect it |

### 2.2 Names, ids and index maps  `[B]`

| API | Go2 example | note |
|---|---|---|
| `m.body("base_link").id` | 1 | named accessor; `.name` too |
| `m.joint("FL_thigh_joint").id` | 2 | `.bodyid`, `.qposadr`, `.dofadr` also exist |
| `m.geom("FL").id` | 20 | foot sphere |
| `m.actuator("FL_calf").id` | 5 | |
| `m.site("imu").id` | 0 | |
| `mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "FL")` | 20 | C-style; −1 if missing |
| `mujoco.mj_id2name(m, mjtObj..., i)` | name or None | loops over all elements |
| `m.jnt_qposadr[jid]` | (12,) → 7…18 | index into `qpos` |
| `m.jnt_dofadr[jid]` | (12,) → 6…17 | index into `qvel`, `qfrc_*`, Jacobian columns, M |
| `m.actuator_trnid[aid, 0]` | joint id | build joint→actuator map |
| `m.body_parentid[b]`, `m.geom_bodyid[g]` | | tree walking, which body a foot belongs to |
| `m.body_pos[b]` | hip offsets in the base frame | hip bodies are children of base_link |

### 2.3 Simulator options  `[C]`

| field | type | note |
|---|---|---|
| `m.opt.timestep` | float | |
| `m.opt.gravity[:] = [...]` | (3,) | assign into the array |
| `m.opt.integrator` | `mujoco.mjtIntegrator.mjINT_EULER / _IMPLICITFAST / _IMPLICIT / _RK4` | RK4 does not support step1/step2 |
| `m.opt.cone` | `mujoco.mjtCone.mjCONE_PYRAMIDAL / _ELLIPTIC` | |
| `m.opt.impratio`, `m.opt.iterations`, `m.opt.noslip_iterations` | | contact solver |
| `mujoco.mjtIntegrator(v).name` | str | enum to name, for printing |

### 2.4 Contact and robot parameters  `[D]`

| field | shape | note |
|---|---|---|
| `m.geom_solref[g]` | (2,) | (timeconst, dampratio), or (−stiffness, −damping) in direct mode; timeconst ≥ 2·dt |
| `m.geom_solimp[g]` | (5,) | (dmin, dmax, width, midpoint, power) |
| `m.geom_friction[g]` | (3,) | `[:, 0]` = sliding μ |
| `m.geom_condim[g]` | int | 3 = sliding friction (feet), 6 = + rolling/torsional |
| `m.geom_priority[g]` | int | higher one sets contact params |
| `m.geom_size[g]` | (3,) | foot radius = `geom_size[g, 0]` = 0.022 |
| `m.body_mass[b]`, `m.body_inertia[b]` | (3,) principal | mass scaling |
| `m.body_ipos[b]`, `m.body_iquat[b]` | | inertial frame in the body frame |
| `m.body_subtreemass[1]` | float | whole robot mass |
| `m.actuator_ctrlrange[a]` | (2,) | torque limits |
| `m.dof_damping`, `m.dof_frictionloss`, `m.dof_armature` | (nv,) | joint losses to compensate |
| `m.jnt_range[j]` | (2,) | joint limits; degrees unless `<compiler angle="radian"/>` |
| `m.nkey`, `m.key_qpos[k]` | | keyframes |

### 2.5 Raw state and inputs (MjData)  `[E]`

| field | shape | Go2 layout |
|---|---|---|
| `d.time` | float | |
| `d.qpos` | (19,) | `[x y z, qw qx qy qz, FL(3) FR(3) RL(3) RR(3)]` |
| `d.qvel` | (18,) | `[v (world), ω (base/body frame!), 12 joint rates]` |
| `d.qacc` | (18,) | after step2 / forward |
| `d.ctrl` | (12,) | actuator order FR, FL, RR, RL |
| `d.xfrc_applied[b]` | (6,) | external [force, torque] on body b, world frame, at its CoM |
| `d.qfrc_applied` | (18,) | external generalized forces |
| `d.sensordata`, `d.sensor(name).data` | | only if the xml has sensors |

### 2.6 Computed kinematics: valid after `mj_forward` / `mj_step1`  `[F]`

| field | shape | use |
|---|---|---|
| `d.xpos[b]`, `d.xquat[b]` | (3,), (4,) | body frame origin/orientation, world |
| `d.xmat[b]` | (9,) → `.reshape(3, 3)` | `R_wb` of base_link = `d.xmat[1]` |
| `d.xipos[b]`, `d.ximat[b]` | | body CoM frame; for composite inertia |
| `d.geom_xpos[g]`, `d.geom_xmat[g]` | (3,), (9,) | foot positions = `d.geom_xpos[[20, 32, 44, 56]]` |
| `d.site_xpos[s]`, `d.site_xmat[s]` | | IMU site pose |
| `d.subtree_com[1]` | (3,) | whole-robot CoM |
| `d.cvel[b]` | (6,) | [ω, v] about the subtree CoM frame; prefer `mj_objectVelocity` |
| `mujoco.mj_kinematics(m, d)` | | positions only (cheaper than `mj_forward`) |
| `mujoco.mj_comPos(m, d)` | | CoM-related quantities after kinematics |

### 2.7 Dynamics quantities  `[F] [G]`

| API / field | shape | use |
|---|---|---|
| `d.qfrc_bias` | (18,) | C(q, q̇) q̇ + g(q): gravity + Coriolis compensation; leg part = `[6:18]` |
| `d.qfrc_actuator` | (18,) | torque the motors actually applied (after clipping) |
| `d.qfrc_passive` | (18,) | damping + springs (frictionloss is a constraint, not here) |
| `d.qfrc_constraint` | (18,) | contacts + joint limits + frictionloss |
| `mujoco.mj_fullM(m, d, M)` | M (18, 18) you allocate | mass matrix; leg block `M[np.ix_(dofs, dofs)]` |
| `mujoco.mj_subtreeVel(m, d)` | fills `d.subtree_linvel`, `d.subtree_angmom` | CoM velocity and angular momentum (not done by `mj_forward`) |
| `mujoco.mj_inverse(m, d)` | fills `d.qfrc_inverse` | torques that give a desired `qacc` (optional) |
| `d.energy` | (2,) | needs the energy flag enabled; for checks |

### 2.8 Jacobians and velocities  `[G]`

| API | output | use |
|---|---|---|
| `mujoco.mj_jacGeom(m, d, jacp, jacr, g)` | jacp, jacr (3, 18) you allocate | foot Jacobian at the sphere center |
| `mujoco.mj_jacBody(m, d, jacp, jacr, b)` | | at the body origin |
| `mujoco.mj_jacSite(m, d, jacp, jacr, s)` | | at a site |
| `mujoco.mj_jac(m, d, jacp, jacr, point, b)` | | any world point attached to body b |
| `mujoco.mj_jacSubtreeCom(m, d, jacp, b)` | (3, 18) | CoM Jacobian |
| `mujoco.mj_objectVelocity(m, d, mjtObj, id, res6, flg_local)` | res (6,) = [ω, v] | velocity of a body/geom/site; `flg_local` 0 = world frame |
| leg block | `jacp[:, dofs]` (3, 3) | `τ = Jᵀ f`; `v_foot = jacp @ d.qvel` |

### 2.9 Contacts

| API / field | note |
|---|---|
| `d.ncon` | number of active contacts |
| `d.contact[i].geom1`, `.geom2`, `.pos`, `.frame`, `.dist` | which geoms, where; `frame` row 0 = normal |
| `mujoco.mj_contactForce(m, d, i, f6)` | force/torque in the contact frame (`f6[0]` = normal) |
| `mujoco.mj_rnePostConstraint(m, d)` then `d.cfrc_ext[b]` | net external (contact) wrench on each body, for real foot GRF vs MPC |

Use: actual foot contact = any contact whose geoms are {`floor`, foot}. Compare it with the scheduled contact (early or late touchdown).

### 2.10 Stepping, reset  `[H]`

| API | note |
|---|---|
| `mujoco.mj_step(m, d)` | full step; set `ctrl` before the call |
| `mujoco.mj_step1(m, d)` | kinematics, Jacobians, bias for the current state; run the controller after this |
| `mujoco.mj_step2(m, d)` | uses `ctrl`, integrates |
| `mujoco.mj_forward(m, d)` | recompute everything from `qpos`/`qvel` without stepping (after editing the state) |
| `mujoco.mj_resetData(m, d)` | model defaults (`qpos0`) |
| `mujoco.mj_resetDataKeyframe(m, d, k)` | load keyframe k |
| `mujoco.mj_setKeyframe(m, d, k)` | store the current state as keyframe k |
| `mujoco.mj_integratePos(m, qpos, qvel, dt)` / `mj_differentiatePos(...)` | quaternion-safe add/subtract of configurations (perturbations, finite differences) |

### 2.11 Snapshots  `[I]`

| API | note |
|---|---|
| `mujoco.mjtState.mjSTATE_INTEGRATION` | full state needed to continue exactly |
| `mujoco.mj_stateSize(m, spec)` | length of the vector |
| `mujoco.mj_getState(m, d, s, spec)` / `mj_setState(m, d, s, spec)` | save/restore; `mj_forward` after set |
| `mujoco.mj_copyData(dst, m, src)` | copy a whole MjData |

### 2.12 Rotations  `[K]`

| API / formula | note |
|---|---|
| quaternion in `qpos[3:7]` | scalar first (w, x, y, z) |
| `scipy Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")` | [roll, pitch, yaw], R = Rz Ry Rx |
| `mujoco.mju_quat2Mat(m9, q)`, `mju_mat2Quat(q, m9)`, `mju_mulQuat`, `mju_rotVecQuat` | no scipy needed |
| yaw = atan2(2(wz + xy), 1 − 2(y² + z²)) | heading |
| up_z = 1 − 2(x² + y²) | body z · world z, for the fall check |
| ω_world = R_wb @ `qvel[3:6]` | `qvel[3:6]` is in the body frame |

### 2.13 Viewer, rendering, logging  `[L]`

| API | note |
|---|---|
| `from mujoco import viewer`; `viewer.launch_passive(m, d)` | your loop keeps control; `.sync()`, `.is_running()`, `.close()`; on macOS run with `mjpython` |
| `viewer.launch(m, d)` | blocking, no controller |
| `mujoco.Renderer(m, height, width)`; `.update_scene(d)`; `.render()` | offscreen frames for videos |
| `np.savez(path, **arrays)` / `np.load(path)` | logs |
| `mujoco.mj_printModel(m, path)` | dump everything (like `dump_model.py`) |

---

## Part 3: reminder list of interface functions

Names follow the current `sim/` code, so the other modules still fit. **Contract** = inputs → outputs. Write the bodies yourself.

### 3.1 Model setup (`sim_env.py`)

| # | function | contract | key APIs | notes |
|---|---|---|---|---|
| 1 | `make_model(cfg)` | `SimCfg` → `(model, data)` standing at home | `MjSpec.from_file`, `spec.delete`, `compile`, `MjData` | remove scene obstacles unless `cfg.obstacles`; then 2 and 4 |
| 2 | `apply_sim_cfg(model, cfg)` | writes cfg into model in place | `model.opt.*`, `geom_solref/solimp/friction/condim`, `body_mass/inertia`, `actuator_ctrlrange` | `None` field = keep xml value; warn if timeconst < 2·dt |
| 3 | `contact_geoms(model)` | → `[floor_id, FL, FR, RL, RR]` ids | `model.geom(name).id` | used by 2 and the contact checks |
| 4 | `reset_robot(model, data, height_offset, key)` | data at keyframe, feet just touching, zero velocity | `mj_resetDataKeyframe`, `mj_forward`, `geom_xpos` | lift `qpos[2]` by `0.022 − min(foot z)`; `mj_forward` again |
| 5 | `foot_friction(model)` | → μ actually used for floor-foot contacts | `geom_friction`, `geom_priority` | max rule vs priority; feeds MPC μ |
| 6 | `describe(model, cfg)` | → printable summary string | enum `.name`, solref, solimp, mass | print at every run start |

### 3.2 State access (`state.py`, class `State`)

Built once in `__init__` (store as attributes):

| # | item | shape | from |
|---|---|---|---|
| 7 | `foot_geoms` | (4,) | geom ids of FL, FR, RL, RR |
| 8 | `qadr`, `vadr` | (12,) | `jnt_qposadr`, `jnt_dofadr` in LEGS × JOINTS order |
| 9 | `leg_dofs` | (4, 3) | `vadr.reshape(4, 3)` |
| 10 | `act_idx` | (4, 3) | joint→actuator map (actuators are FR, FL, RR, RL) |
| 11 | `tau_limit` | (4, 3) | `actuator_ctrlrange[act_idx, 1]` |
| 12 | `hip_b` | (4, 3) | `body_pos` of the hip bodies (base frame) |
| 13 | `base_body` | int | `model.body("base_link").id` |

Readers (properties or methods; return copies):

| # | function | output | key APIs | watch out |
|---|---|---|---|---|
| 14 | `base_pose` | (6,) [x y z roll pitch yaw] | `qpos[0:7]`, scipy | quaternion order |
| 15 | `R_wb` | (3, 3) | `xmat[base].reshape(3, 3)` | |
| 16 | `base_lin_vel_world` | (3,) | `qvel[0:3]` | already world |
| 17 | `base_ang_vel_world` | (3,) | `R_wb @ qvel[3:6]` | body → world |
| 18 | `joint_positions` / `joint_velocities` | (4, 3) | `qpos[qadr]`, `qvel[vadr]` | |
| 19 | `joint_torques` | (4, 3) | `qfrc_actuator[vadr]` | applied, after clipping |
| 20 | `foot_positions` | (4, 3) | `geom_xpos[foot_geoms]` | sphere center (z ≈ 0.022 on ground) |
| 21 | `foot_jacobians` | (4, 6, nv) | `mj_jacGeom` per foot | rows [linear; angular] |
| 22 | `foot_velocities` | (4, 3) | `J[:, :3] @ qvel` | |
| 23 | `com_pos` | (3,) | `subtree_com[base]` | |
| 24 | `hip_world` | (4, 3) | `p_base + hip_b @ R_wbᵀ` | |
| 25 | `leg_bias` | (4, 3) | `qfrc_bias[vadr]` | gravity + Coriolis |
| 26 | `mass_matrix_leg(i)` (optional) | (3, 3) | `mj_fullM` + `np.ix_` | only needed for swing feed-forward |
| 27 | `mpc_state(g)` | (13,) [p, rpy, v, ω_world, g] | 14, 16, 17 | base-link version |
| 28 | `mpc_state_centroidal(I_body, com_b, g)` | (13,) | `mj_subtreeVel`, `subtree_linvel`, `subtree_angmom` | p = CoM − R com_b; ω = (R I Rᵀ)⁻¹ L |
| 29 | `foot_contacts_actual()` (optional) | (4,) bool | `d.contact`, `ncon` | compare with the gait schedule |
| 30 | `foot_grf_actual()` (optional) | (4, 3) | `mj_contactForce` (rotate by `contact.frame`) or `cfrc_ext` | compare with MPC forces |

Module-level:

| # | function | contract | key APIs |
|---|---|---|---|
| 31 | `robot_params(model, key)` | → dict `mass`, `I_body` (3, 3), `com_b` (3,), `foot_b` (4, 3) | scratch `MjData`, keyframe, `xipos`, `ximat`, `body_inertia`, parallel-axis theorem |

### 3.3 Actuation

| # | function | contract | notes |
|---|---|---|---|
| 32 | `command(tau)` | (4, 3) leg-order torques → `d.ctrl` | `np.clip` with `tau_limit`, then `ctrl[act_idx.reshape(-1)] = tau.reshape(-1)`. Lives in `LegController` in `sim/`, but it is pure interface. |

### 3.4 Snapshots and perturbations (`sim_env.py`)

| # | function | contract | key APIs |
|---|---|---|---|
| 33 | `get_sim_state(model, data)` | → state vector | `mj_stateSize`, `mj_getState` (INTEGRATION) |
| 34 | `set_sim_state(model, data, s)` | restore | `mj_setState`, `mj_forward` |
| 35 | `snapshot(model, data, ctrl)` | → (sim state, controller state) | 33 + `ctrl.get_state` |
| 36 | `restore(model, data, ctrl, snap)` | | 34 + `ctrl.set_state` |
| 37 | `heading_yaw(data)` | → yaw | quaternion formula |
| 38 | `push_velocity(model, data, dv6)` | add heading-frame [dv, dω] | `qvel[0:3] += Rz dv`; `qvel[3:6] += R_wbᵀ Rz dω`; `mj_forward` |
| 39 | `ForcePush(t0, duration, force, torque).apply(data)` | timed external wrench | `xfrc_applied[1]`; set back to 0 when off |

### 3.5 Run loop and logging (`sim_env.py`)

| # | function | contract | notes |
|---|---|---|---|
| 40 | `is_fallen(data)` | → bool | base z < 0.12 or up_z < 0.5 |
| 41 | `SimLog.append(data, ctrl, v_cmd)` | store copies of t, qpos, qvel, phase, MPC state, contacts, forces, footholds, tau, v_cmd, MPC status | `.copy()` everything |
| 42 | `SimLog.arrays()` / `SimLog.save(path, **extra)` | → dict of arrays / `.npz` file | `np.asarray`, `np.savez` |
| 43 | `run(model, data, ctrl, duration, v_cmd, log_every, callbacks, pushes, viewer, realtime, stop_on_fall)` | → `(log, fallen)` | per step: pushes → `mj_step1` → `ctrl.step(v)` → `mj_step2` → log → fall check → callbacks → viewer sync / real-time sleep; `v_cmd` may be a constant or a function of t; close the viewer in `finally` |

### 3.6 Build order and self-checks

Write and test in this order. Each check is a few lines in a `__main__` block.

1. **`make_model` + `describe`:** the printed nq/nv/nu are 19/18/12, mass is 15.21 kg, and options match your `SimCfg`.
2. **Index maps (7–13):** `act_idx` for FL is [3, 4, 5]; FR is [0, 1, 2].
3. **`reset_robot`:** all four foot z ≈ 0.022, base z near 0.27–0.32 depending on the keyframe.
4. **Readers (14–25):** rotate the base by a known rpy, then check that `base_pose` returns it. Check that `foot_jacobians` matches a finite difference of `foot_positions`, and that `leg_bias` matches g(q) in `go2_model_info.txt` when you set the dump's pose (base z 0.27, joints (0, 0.9, −1.8), zero velocity), e.g. FL ≈ (1.065, 0.439, −0.272).
5. **`robot_params`:** mass 15.21; `I_body` symmetric and positive definite; `foot_b` z ≈ −(base height − 0.022).
6. **Stand test with 32 + 43:** PD on joints around home (no MPC). The robot should stay up; summed contact normal force ≈ 149 N.
7. **Snapshots (33–36):** run, snapshot, run 0.5 s, restore, run 0.5 s again. The two results must be bitwise equal.
8. **Pushes (38, 39):** after a 1 m/s `push_velocity`, `base_lin_vel_world` jumps by 1 m/s in the heading direction.
9. **Centroidal state (28):** standing still, v ≈ 0 and ω ≈ 0. While swinging one leg in the air (base fixed by PD), compare base-link ω with centroidal ω.
