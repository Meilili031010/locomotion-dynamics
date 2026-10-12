import mujoco

MODEL_PATH = "/home/mel/ws/unitree_ws/src/unitree_mujoco/unitree_robots/go2/scene.xml"

def name_of(obj_type, idx):
    n = mujoco.mj_id2name(m, obj_type, idx)
    return n if n is not None else "<unnamed>"

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

print("=== 基本维度 ===")
print(f"timestep = {m.opt.timestep}")
print(f"nq = {m.nq}  (广义坐标)")
print(f"nv = {m.nv}  (广义速度)")
print(f"nu = {m.nu}  (执行器数)")
print(f"ncon 初始 = {d.ncon}")

print("\n=== 关节 ===")
for i in range(m.njnt):
    name = name_of(mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"{i:2d}  {name:<20s} type={m.jnt_type[i]}  "
          f"qposadr={m.jnt_qposadr[i]}  dofadr={m.jnt_dofadr[i]}")

print("\n=== 执行器 ===")
for i in range(m.nu):
    name = name_of(mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    joint_id = m.actuator_trnid[i, 0]
    joint_name = name_of(mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    print(f"{i:2d}  {name:<20s} -> {joint_name:<20s} "
          f"gaintype={m.actuator_gaintype[i]} biastype={m.actuator_biastype[i]} "
          f"kp={m.actuator_gainprm[i,0]:.1f} ctrlrange={m.actuator_ctrlrange[i]}")

print("\n=== keyframe ===")
for i in range(m.nkey):
    print(f"{i}: {name_of(mujoco.mjtObj.mjOBJ_KEY, i)}")

print("\n=== 传感器 ===")
for i in range(m.nsensor):
    name = name_of(mujoco.mjtObj.mjOBJ_SENSOR, i)
    print(f"{i:2d}  {name}  dim={m.sensor_dim[i]}")

print("\n=== 初始基座状态 ===")
print("base pos :", d.qpos[0:3])
print("base quat:", d.qpos[3:7], "(MuJoCo 顺序 w,x,y,z)")
print("base vel :", d.qvel[0:6], "(前 3 为线速度，后 3 为角速度)")