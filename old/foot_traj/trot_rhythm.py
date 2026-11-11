import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from numpy.polynomial import Polynomial


def get_normalized_min_jerk_poly(p0_xy, pf_xy):
    dp = pf_xy - p0_xy
    coeffs = np.zeros((6, 2))
    coeffs[0] = p0_xy
    coeffs[3] = 10.0 * dp
    coeffs[4] = -15.0 * dp
    coeffs[5] = 6.0 * dp
    return Polynomial(coeffs[:, 0]), Polynomial(coeffs[:, 1])


def sin_height(bar_time, step_height):
    bar_time = np.clip(bar_time, 0.0, 1.0)
    return step_height * np.sin(np.pi * bar_time)


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

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

colors = ["crimson", "darkorange", "forestgreen", "dodgerblue"]
labels = ["FL", "FR", "RL", "RR"]

# Draw liftoff reference markers
for leg_idx in range(4):
    ax.scatter(
        xs0[leg_idx, 0],
        xs0[leg_idx, 1],
        xs0[leg_idx, 2],
        color=colors[leg_idx],
        s=30,
        alpha=0.4,
    )

trails = [
    ax.plot([], [], [], color=colors[j], lw=2, label=labels[j])[0]
    for j in range(4)
]
feet = [
    ax.plot([], [], [], marker="o", color=colors[j], markersize=7)[0]
    for j in range(4)
]

ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(0.0, 0.15)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Quadruped Diagonal Trot Gait Trajectories")
ax.view_init(elev=25, azim=-60)
ax.legend(loc="upper left")


def update(frame):
    for leg_idx in range(4):
        trails[leg_idx].set_data(
            xs[: frame + 1, leg_idx, 0], xs[: frame + 1, leg_idx, 1]
        )
        trails[leg_idx].set_3d_properties(xs[: frame + 1, leg_idx, 2])

        feet[leg_idx].set_data([xs[frame, leg_idx, 0]], [xs[frame, leg_idx, 1]])
        feet[leg_idx].set_3d_properties([xs[frame, leg_idx, 2]])

    return trails + feet


ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=dt * 1000,
    blit=False,
    repeat=False,
)

plt.show()