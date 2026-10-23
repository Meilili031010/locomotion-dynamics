# [REVIEW NOTE] Legacy helpers, not used by the controller (swing_traj.py
# replaces them). cubic_Bezier() below is incorrect: the z formula adds
# (2s)^3 as a height (not a Bezier weight), and the xy curve uses p0 only in
# the first term. Kept unchanged for reference.
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import comb

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


def cubic_Bezier(bar_time, p0_xy, pf_xy, z_0, z_f, step_height):
    bar_time = np.clip(bar_time, 0.0, 1.0)
    if bar_time < 0.5:
        z = z_0 + (bar_time*2)**3 + 3*(bar_time*2)**2*(1-bar_time*2)*step_height
    else:
        z = z_f + (1-bar_time*2)**3 + 3*(1-bar_time*2)**2*(bar_time*2)*step_height

    return (
        (1 - bar_time) ** 3 * p0_xy
        + 3 * (1 - bar_time) ** 2 * bar_time * pf_xy
        + 3 * (1 - bar_time) * bar_time**2 * pf_xy
        + bar_time**3 * pf_xy
    ), z