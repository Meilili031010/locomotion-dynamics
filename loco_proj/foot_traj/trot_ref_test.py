import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

# Test for foot trajectory generation for trot gait (in 2D)
x0 = np.array([0.0, 0.0])

dt = 0.01
step_time = 0.5
step_length = 0.5
step_height = 0.1
v = step_length / step_time

times = np.linspace(0, step_length, int(step_time/dt)+1)
# print(times)

x0dot = np.array([0.0, 0.0])
x0ddot = np.array([0.0, 0.0])

xf = np.array([step_length, 0.0])
xf_dot = np.array([0.0, 0.0])
xf_ddot = np.array([0.0, 0.0])
T = step_time
A = np.array(
            [[T**3, T**4, T**5], [3 * T**2, 4 * T**3, 5 * T**4], [6 * T, 12 * T**2, 20 * T**3]]
        )

c0 = x0[0]
c1 = x0dot[0]
c2 = 0.5 * x0ddot[0]
b = np.array([xf[0] - c0 - c1 * T - c2 * T**2, xf_dot[0] - c1 - 2 * c2 * T, xf_ddot[0] - 2 * c2])
c3, c4, c5 = np.linalg.solve(A, b)
coeffs = [c0, c1, c2, c3, c4, c5]
poly = Polynomial(coeffs)
print("Polynomial coefficients:", [f"{c:.2f}" for c in coeffs])

x_vals = poly(times)
plt.plot(times, x_vals, label="X Foot Trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Position X (m)")
plt.grid(True)
plt.legend()
plt.show()

z_vals = np.sign(np.sin(np.pi * times / step_time)) * step_height * np.sin(np.pi * times / step_time)
plt.plot(times, z_vals, label="Z Foot Trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Position Z (m)")
plt.grid(True)
plt.legend()
plt.show()