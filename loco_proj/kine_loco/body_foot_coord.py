import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial import Polynomial

from foot_traj.trot_rhythm import get_normalized_min_jerk_poly, sin_height

def knee_pos(foot_pos, hip_pos, lap_len=0.3, calf_len=0.3):
    dx = foot_pos - hip_pos
    d = np.linalg.norm(dx)
    if d > (lap_len + calf_len):
        raise ValueError("Foot position is out of reach for the given leg lengths.")
    

period = 1.0
dt = 0.01
step_length = 0.3
step_height = 0.08
ratio = 0.5
heading = 0.0
times = np.arange(0, period, dt)
num_frames = len(times)

# Initial feet positions [FL, FR, RL, RR]
xs0 = np.array(
    [
        [0.3, -0.15, 0.0],  # FL
        [0.3, 0.15, 0.0],  # FR
        [-0.1, -0.15, 0.0],  # RL
        [-0.1, 0.15, 0.0],  # RR
    ]
)


# leg length
lap_len = 0.3
calf_len = 0.3

# Initial hip position
hip_xs = xs0 + np.array([0, 0, lap_len * np.sqrt(2)])


delta_step = step_length * np.array([np.cos(heading), np.sin(heading)])

poly_FL_x, poly_FL_y = get_normalized_min_jerk_poly(
    xs0[0, :2], xs0[0, :2] + delta_step
)
poly_FR_x, poly_FR_y = get_normalized_min_jerk_poly(
    xs0[1, :2], xs0[1, :2] + delta_step
)
poly_RL_x, poly_RL_y = get_normalized_min_jerk_poly(
    xs0[2, :2], xs0[2, :2] + delta_step
)
poly_RR_x, poly_RR_y = get_normalized_min_jerk_poly(
    xs0[3, :2], xs0[3, :2] + delta_step
)

xs = np.zeros((num_frames, 4, 3))

for i, t in enumerate(times):
    if t < ratio * period:
        bar_t = t / (ratio * period)

        # FL
        xs[i, 0] = [
            poly_FL_x(bar_t),
            poly_FL_y(bar_t),
            sin_height(bar_t, step_height),
        ]
        # RR
        xs[i, 3] = [
            poly_RR_x(bar_t),
            poly_RR_y(bar_t),
            sin_height(bar_t, step_height),
        ]

        xs[i, 1] = xs0[1]
        xs[i, 2] = xs0[2]
    else:
        bar_t = (t - ratio * period) / ((1.0 - ratio) * period)

        # FR
        xs[i, 1] = [
            poly_FR_x(bar_t),
            poly_FR_y(bar_t),
            sin_height(bar_t, step_height),
        ]
        # RL
        xs[i, 2] = [
            poly_RL_x(bar_t),
            poly_RL_y(bar_t),
            sin_height(bar_t, step_height),
        ]

        xs[i, 0] = [poly_FL_x(1.0), poly_FL_y(1.0), 0.0]
        xs[i, 3] = [poly_RR_x(1.0), poly_RR_y(1.0), 0.0]
