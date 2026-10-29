"""
MPC LIBRARY  (convex MPC of the MIT Cheetah 3 paper, numpy / scipy.sparse / OSQP)
================================================================================

Reference for writing   sim/mpc_controller.py   and the dynamics part of
sim/structs.py  (skew, rot_z, continuous_dynamics, discretize).

The examples use a PLANAR single rigid body with two feet (TOY model):
    state x = [px, pz, theta, vx, vz, w, g]        (7)
    input u = [fx_1, fz_1, fx_2, fz_2]             (4)
It has exactly the structure of the Go2 model
    state x = [p(3), rpy(3), v(3), w(3), g]        (13)
    input u = [f_FL, f_FR, f_RL, f_RR] (3 each)    (12)
so every step below carries over; writing the 13-state version is your job.

    python mpc_lib.py          # all examples
    python mpc_lib.py E H      # chosen sections

Sections
    A  State / input layout, gravity as a state
    B  Small matrix helpers         skew(r), rot_z(yaw)
    C  Continuous SRB dynamics      x_dot = Ac x + Bc u
    D  Discretization               Euler, zero-order hold (expm), truncated series
    E  Condensed prediction         X = Sx x0 + Su U
    F  Cost -> QP                   P = Su' Q Su + R,  q = Su' Q (Sx x0 - Xref)
    G  Force constraints            friction pyramid, fz bounds, swing legs removed
    H  OSQP + scipy.sparse          setup / solve / status
    I  Extra linear constraints     hard (l = u) vs soft (penalty)
    J  Receding horizon loop        solve, apply first force, repeat
    K  Robustness                   fallback, timing, sanity checks
"""
import sys
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import expm

# Toy model constants
NX, NU = 7, 4
MASS, INERTIA, G = 10.0, 0.3, -9.81
IX_P, IX_TH, IX_V, IX_W, IX_G = slice(0, 2), 2, slice(3, 5), 5, 6


def _title(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


# ==========================================================================
# A. STATE / INPUT LAYOUT
# ==========================================================================
"""
Go2 (structs.py):  IDX_P = 0:3, IDX_RPY = 3:6, IDX_V = 6:9, IDX_W = 9:12, IDX_G = 12

Gravity as a state: dynamics  v_dot = sum(f)/m + g  is AFFINE (constant g).
Add a constant state g (g_dot = 0, initial value -9.81) and put it into A:
    A[vz_row, g_col] = 1      ->   v_dot_z = ... + g
Now the model is LINEAR x_dot = A x + B u and the prediction has no extra
constant term. Give g zero weight in Q (you cannot "track" it).

Reference X_ref has the same layout. Convention in sim/: X_ref has h+1 rows,
row 0 = x0, rows 1..h are the targets of x_1..x_h.
"""


def example_A():
    _title("A. layout")
    x0 = np.zeros(NX)
    x0[IX_P] = [0.0, 0.25]
    x0[IX_G] = G
    print("toy x0 =", x0, "  (px, pz, theta, vx, vz, w, g)")


# ==========================================================================
# B. SMALL MATRIX HELPERS
# ==========================================================================
"""
skew(r) = [r]x  (3, 3)  so that  skew(r) @ f == np.cross(r, f)
    torque of a foot force about the CoM:  tau = r x f = skew(r) f   (linear in f!)
rot_z(yaw) (3, 3)  heading rotation, heading frame -> world

Cheetah 3 simplification (small roll/pitch):
    rpy_dot ~= Rz(yaw)^T w_world
    I_world ~= Rz I_body Rz^T          (only yaw rotates the inertia)
    -> the model depends on yaw only, and yaw comes from the REFERENCE, so
       A_k is known before solving (the problem stays a convex QP).
"""


def skew(r):
    return np.array([[0.0, -r[2], r[1]],
                     [r[2], 0.0, -r[0]],
                     [-r[1], r[0], 0.0]])


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def example_B():
    _title("B. skew / rot_z")
    r, f = np.array([0.2, -0.1, -0.25]), np.array([5.0, 1.0, 60.0])
    print("skew(r) @ f == cross(r, f):", np.allclose(skew(r) @ f, np.cross(r, f)))
    print("rot_z(pi/2) @ [1,0,0] =", np.round(rot_z(np.pi / 2) @ [1, 0, 0], 6))


# ==========================================================================
# C. CONTINUOUS SRB DYNAMICS
# ==========================================================================
"""
3D (Cheetah 3, eq. 16), for your structs.continuous_dynamics:
    p_dot   = v
    rpy_dot = Rz^T w
    v_dot   = (1/m) sum_i f_i + [0, 0, g]
    w_dot   = I_w^-1 sum_i r_i x f_i          I_w = Rz I_body Rz^T
        r_i = foot_i - CoM (world)  -> needs PREDICTED foot positions over the
              horizon (mpc_controller.predict_foot_positions)
    Ac (13, 13):  [0:3, 6:9] = I,  [3:6, 9:12] = Rz^T,  [8, 12] = 1
    Bc (13, 12):  block i: [6:9] = I/m,  [9:12] = I_w^-1 skew(r_i)

Planar toy (below): torque = r_x f_z - r_z f_x  (2D cross product).
"""


def toy_continuous(r_feet):
    """r_feet (2, 2): foot - CoM, [rx, rz] for each foot."""
    Ac = np.zeros((NX, NX))
    Ac[0:2, 3:5] = np.eye(2)          # p_dot = v
    Ac[2, 5] = 1.0                    # theta_dot = w
    Ac[4, IX_G] = 1.0                 # vz_dot += g
    Bc = np.zeros((NX, NU))
    for i, (rx, rz) in enumerate(r_feet):
        Bc[3:5, 2 * i:2 * i + 2] = np.eye(2) / MASS
        Bc[5, 2 * i:2 * i + 2] = np.array([-rz, rx]) / INERTIA
    return Ac, Bc


def example_C():
    _title("C. continuous dynamics")
    r = np.array([[0.2, -0.25], [-0.2, -0.25]])
    Ac, Bc = toy_continuous(r)
    x = np.zeros(NX); x[IX_G] = G
    u = np.array([0, -MASS * G / 2, 0, -MASS * G / 2])   # each foot carries half
    print("x_dot with feet carrying the weight:", np.round(Ac @ x + Bc @ u, 6), "(all zero)")


# ==========================================================================
# D. DISCRETIZATION   x_{k+1} = Ad x_k + Bd u_k   (u constant over dt)
# ==========================================================================
"""
Forward Euler:      Ad = I + Ac dt,   Bd = Bc dt               (first order)
Zero-order hold:    expm([[Ac, Bc], [0, 0]] * dt) = [[Ad, Bd], [0, I]]   (exact)
    scipy.linalg.expm(M)
Truncated series:   expm(M dt) = I + M dt + (M dt)^2/2 + ...
    For the SRB model M^3 = 0 (nilpotent: position <- velocity <- force), so
    three terms ARE the exact ZOH and much cheaper than expm (what
    structs.discretize does). Check it numerically before trusting it.

Pitfall: with Euler, a force does not move the position within one step
(Bd has no position rows); ZOH gets the dt^2/2 term right.
"""


def zoh(Ac, Bc, dt):
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = Ac, Bc
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def example_D():
    _title("D. discretization")
    Ac, Bc = toy_continuous(np.array([[0.2, -0.25], [-0.2, -0.25]]))
    dt = 0.025
    Ad_e, Bd_e = np.eye(NX) + Ac * dt, Bc * dt
    Ad_z, Bd_z = zoh(Ac, Bc, dt)
    M = np.zeros((NX + NU, NX + NU)); M[:NX, :NX], M[:NX, NX:] = Ac, Bc
    print("M^3 == 0 :", np.allclose(np.linalg.matrix_power(M, 3), 0))
    print("Euler Bd position row (pz <- fz1):", Bd_e[1, 1], "   ZOH:", round(Bd_z[1, 1], 6),
          "= dt^2/(2m) =", dt ** 2 / (2 * MASS))
    E3 = np.eye(NX + NU) + M * dt + 0.5 * (M * dt) @ (M * dt)
    print("3-term series == expm:", np.allclose(E3[:NX, :NX], Ad_z) and np.allclose(E3[:NX, NX:], Bd_z))


# ==========================================================================
# E. CONDENSED (LIFTED) PREDICTION
# ==========================================================================
"""
Stack the horizon:  X = [x_1; ...; x_h] (NX h),  U = [u_0; ...; u_{h-1}] (NU h)
    X = Sx x0 + Su U
    Sx block k  = A_k ... A_1 A_0                            (NX, NX)
    Su block (k, j) = A_k ... A_{j+1} B_j    for j <= k,  else 0   (lower triangular)
Build it recursively (time-varying A_k, B_k because yaw_ref and r_feet change):
    A_cum = A_k @ A_cum_prev
    B_row = A_k @ B_row_prev ;  B_row[:, block k] = B_k
Only the forces are unknowns -> the QP is small (NU h variables, no states).
Alternative "sparse" formulation keeps X as variables with equality
constraints; bigger but sparser. sim/ uses the condensed one.

Always test: Sx x0 + Su U == simulating x_{k+1} = A_k x_k + B_k u_k.
"""


def lifted(A_list, B_list):
    h = len(A_list)
    Sx = np.zeros((NX * h, NX))
    Su = np.zeros((NX * h, NU * h))
    A_cum, B_row = np.eye(NX), np.zeros((NX, NU * h))
    for k in range(h):
        A_cum = A_list[k] @ A_cum
        B_row = A_list[k] @ B_row
        B_row[:, NU * k:NU * (k + 1)] = B_list[k]
        Sx[NX * k:NX * (k + 1)] = A_cum
        Su[NX * k:NX * (k + 1)] = B_row
    return Sx, Su


def example_E():
    _title("E. lifted prediction")
    h, dt = 10, 0.025
    rng = np.random.default_rng(0)
    AB = [zoh(*toy_continuous(rng.normal(0, 0.2, (2, 2))), dt) for _ in range(h)]
    Sx, Su = lifted([a for a, _ in AB], [b for _, b in AB])
    x0 = rng.normal(size=NX); x0[IX_G] = G
    U = rng.normal(size=NU * h)
    x, X_sim = x0.copy(), []
    for k, (A, B) in enumerate(AB):
        x = A @ x + B @ U[NU * k:NU * (k + 1)]
        X_sim.append(x)
    print("Sx", Sx.shape, " Su", Su.shape,
          " rollout matches:", np.allclose(Sx @ x0 + Su @ U, np.concatenate(X_sim)))


# ==========================================================================
# F. COST -> QP
# ==========================================================================
"""
J = (X - Xref)' Q (X - Xref) + U' R U        Q, R diagonal, tiled over the horizon
  = U' (Su' Q Su + R) U + 2 (Sx x0 - Xref)' Q Su U + const
QP standard form (OSQP): minimize 1/2 U' P U + q' U
    P = 2 (Su' Q Su + R)   q = 2 Su' Q (Sx x0 - Xref)
    (the factor 2 can be dropped from both: same minimizer)

Numpy tricks:
    np.tile(Q_diag, h)           diagonal of the stacked Q
    Qs[:, None] * Su             == diag(Qs) @ Su, without building diag
    Xref[1:].reshape(-1)         (h, NX) -> stacked vector, row-major = x_1, x_2 ...
R (> 0, tiny e.g. 1e-5) keeps P positive definite and the forces smooth.

Unconstrained solution (useful for checking OSQP):  U* = -P^-1 q
"""


def qp_cost(Sx, Su, x0, Xref, Q_diag, R_stacked):
    """Su may already have swing columns removed; R_stacked must match its columns."""
    h = Sx.shape[0] // NX
    Qs = np.tile(Q_diag, h)
    QSu = Qs[:, None] * Su
    P = Su.T @ QSu + np.diag(R_stacked)
    q = QSu.T @ (Sx @ x0 - Xref[1:].reshape(-1))
    return P, q


def example_F():
    _title("F. cost -> QP (unconstrained check)")
    h, dt = 10, 0.025
    r = np.array([[0.2, -0.25], [-0.2, -0.25]])
    A, B = zoh(*toy_continuous(r), dt)
    Sx, Su = lifted([A] * h, [B] * h)
    x0 = np.array([0, 0.22, 0, 0, 0, 0, G])
    Xref = np.tile([0, 0.25, 0, 0, 0, 0, G], (h + 1, 1))
    P, q = qp_cost(Sx, Su, x0, Xref, np.array([1, 50, 10, 1, 1, 1, 0.0]), np.full(NU * h, 1e-5))
    U = -np.linalg.solve(P, q)
    print("P symmetric:", np.allclose(P, P.T), " first force [fx1 fz1 fx2 fz2]:",
          np.round(U[:4], 1), "(fz sum > weight 98 N: pushes up)")


# ==========================================================================
# G. FORCE CONSTRAINTS
# ==========================================================================
"""
Per stance foot (3D, friction pyramid):
     fx - mu fz <= 0,   -fx - mu fz <= 0,   fy - mu fz <= 0,   -fy - mu fz <= 0
     fz_min <= fz <= fz_max
  as rows C_f f  in [lb, ub]:
     C_f = [[1,0,-mu], [-1,0,-mu], [0,1,-mu], [0,-1,-mu], [0,0,1]]
     lb  = [-inf]*4 + [fz_min],    ub = [0]*4 + [fz_max]
Same block for every stance foot at every step:
     C = sparse.kron(sparse.eye(n_active), C_f)       block diagonal
     np.tile(lb, n_active), np.tile(ub, n_active)

Swing feet: force must be 0. Cheapest way: REMOVE their columns.
     contacts (h, 4) bool  -> per-force-component mask
     active = np.repeat(contacts.reshape(-1), 3)        (12 h,)
     Su_a = Su[:, active]   R_a = R_stacked[active]
     after solving: U = zeros(12 h); U[active] = sol
     (if no foot is in contact at all: nothing to optimize -> skip the QP)
Planar toy: C_f = [[1,-mu], [-1,-mu], [0,1]].
"""


def friction_rows(mu, fz_min, fz_max, n_active):
    C_f = np.array([[1.0, -mu], [-1.0, -mu], [0.0, 1.0]])
    lb = np.array([-np.inf, -np.inf, fz_min])
    ub = np.array([0.0, 0.0, fz_max])
    C = sparse.kron(sparse.eye(n_active), sparse.csc_matrix(C_f), format="csc")
    return C, np.tile(lb, n_active), np.tile(ub, n_active)


def example_G():
    _title("G. constraints + swing-leg masking")
    contacts = np.array([[1, 1], [1, 0], [0, 1]], dtype=bool)   # h = 3, 2 feet
    active = np.repeat(contacts.reshape(-1), 2)                   # 2 components per foot (planar)
    print("active mask:", active.astype(int))
    C, lb, ub = friction_rows(0.5, 0.0, 150.0, int(contacts.sum()))
    print("C shape:", C.shape, "(3 rows per stance foot, 2 force components each)")
    print(C.toarray()[:3])


# ==========================================================================
# H. OSQP + scipy.sparse
# ==========================================================================
"""
OSQP solves   min 1/2 x' P x + q' x   s.t.  l <= A x <= u
    import osqp
    prob = osqp.OSQP()
    prob.setup(P=P_csc, q=q, A=A_csc, l=l, u=u,
               verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, polishing=False)
    res = prob.solve()
    res.x                 solution (None or NaN on failure)
    res.info.status       "solved", "solved inaccurate", "primal infeasible", "maximum iterations reached", ...
    res.info.solve_time / run_time
    prob.update(q=..., l=..., u=..., Px=..., Ax=...)    re-solve with same sparsity (warm start)
    prob.warm_start(x=...)
P: pass the UPPER triangle as CSC:  sparse.csc_matrix(np.triu(P))   (or sparse.triu(P, format="csc"))
A: any CSC matrix; equality rows have l == u; one-sided rows use -np.inf / np.inf.
Version note: osqp >= 1.0 uses `polishing`; osqp 0.6 called it `polish`.

scipy.sparse cheat sheet
    sparse.csc_matrix(dense)     sparse.eye(n)     sparse.kron(A, B)
    sparse.vstack([A, B], format="csc")   sparse.block_diag([...])
    S.toarray()   S.shape   S.nnz
"""


def solve_qp(P, q, C, lb, ub):
    prob = osqp.OSQP()
    prob.setup(P=sparse.csc_matrix(np.triu(P)), q=q, A=C, l=lb, u=ub,
               verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000)
    res = prob.solve()
    return res.x, res.info.status


def example_H():
    _title("H. OSQP on a tiny QP")
    # min (x0-1)^2 + (x1-2)^2   s.t.  x0 + x1 <= 2,  x0 >= 0
    P = 2 * np.eye(2)
    q = np.array([-2.0, -4.0])
    A = sparse.csc_matrix([[1.0, 1.0], [1.0, 0.0]])
    x, status = solve_qp(P, q, A, np.array([-np.inf, 0.0]), np.array([2.0, np.inf]))
    print("status:", status, " x:", np.round(x, 4), " (expected [0.5, 1.5])")


# ==========================================================================
# I. EXTRA LINEAR CONSTRAINTS:  G U = g   (e.g. the period constraint)
# ==========================================================================
"""
Any linear condition on predicted states  D x_b - E x_a = c  becomes linear
in U because x_j = Sx_j x0 + Su_j U  (rows j of the lifted matrices):
    G = D Su_b - E Su_a,     g = c - D Sx_b x0 + E Sx_a x0
(x_0 is the measured state: a constant, it moves to the right side.)
Then
  hard:  append G as rows of A with l = u = g         (may make the QP infeasible)
  soft:  P += w G'G,  q -= w G'g                      (penalty w ||G U - g||^2)
  pattern used in mpc_controller: try hard, if not "solved" re-solve soft.
Only use G[:, active] (swing columns were removed).
"""


def example_I():
    _title("I. hard vs soft linear constraint")
    P, q = 2 * np.eye(2), np.array([-2.0, -4.0])
    G, g = np.array([[1.0, -1.0]]), np.array([0.0])           # want x0 == x1
    C0 = sparse.csc_matrix((0, 2))
    A = sparse.vstack([C0, sparse.csc_matrix(G)], format="csc")
    x_h, st = solve_qp(P, q, A, g, g)
    x_s, _ = solve_qp(P + 10 * G.T @ G, q - 10 * G.T @ g, sparse.csc_matrix(np.zeros((1, 2))),
                      np.array([-np.inf]), np.array([np.inf]))
    print("hard:", np.round(x_h, 3), st, "  soft (w=10):", np.round(x_s, 3), "(close, not exact)")


# ==========================================================================
# J. RECEDING HORIZON LOOP
# ==========================================================================
"""
Every MPC step (dt_mpc, e.g. 25 ms = 25 sim steps):
    x0      <- measured state
    X_ref   <- reference over the horizon            (gait_lib: reference generation)
    contacts<- (h, 4) from the gait scheduler        (gait_lib)
    r_feet  <- predicted foot - CoM over the horizon
    build A_k, B_k -> Sx, Su -> P, q -> constraints -> OSQP
    apply only forces[0]; hold them until the next solve (zero-order hold);
    the leg controller turns them into torques       (controller_lib)
Below: the toy body starts too low and tilted, with an alternating contact
schedule; the "plant" is the discrete model itself.
"""


def toy_mpc_step(x0, Xref, contacts, r_feet, h, dt, mu=0.5):
    A, B = zoh(*toy_continuous(r_feet), dt)
    Sx, Su = lifted([A] * h, [B] * h)
    active = np.repeat(contacts.reshape(-1), 2)
    Qd = np.array([1.0, 50, 20, 1, 1, 1, 0])
    R_a = np.tile(np.full(NU, 1e-5), h)[active]
    P, q = qp_cost(Sx, Su[:, active], x0, Xref, Qd, R_a)
    C, lb, ub = friction_rows(mu, 0.0, 200.0, int(contacts.sum()))
    sol, status = solve_qp(P, q, C, lb, ub)
    U = np.zeros(NU * h)
    U[active] = sol
    return U.reshape(h, NU), status


def example_J():
    _title("J. receding horizon on the toy model")
    h, dt = 10, 0.025
    r_feet = np.array([[0.2, -0.25], [-0.2, -0.25]])
    A, B = zoh(*toy_continuous(r_feet), dt)
    x = np.array([0, 0.20, 0.15, 0, 0, 0, G])
    x_target = np.array([0, 0.25, 0, 0, 0, 0, G])
    for n in range(24):
        k = n + np.arange(h)
        contacts = np.stack([(k // 4) % 2 == 0, (k // 4) % 2 == 1], axis=1) | (k[:, None] < 8)
        forces, status = toy_mpc_step(x, np.tile(x_target, (h + 1, 1)), contacts, r_feet, h, dt)
        x = A @ x + B @ forces[0].reshape(-1)
        if n % 4 == 0:
            print(f"step {n:2d} x={x[0]:+.3f} z={x[1]:.3f} theta={x[2]:+.3f} f0={np.round(forces[0], 1)} {status}")


# ==========================================================================
# K. ROBUSTNESS
# ==========================================================================
"""
* Check res.x is not None, np.all(np.isfinite(res.x)) and "solved" in status
  (accept "solved inaccurate" or not? decide and log it).
* Fallback when the QP fails: weight split over stance feet,
  fz_i = m |g| / n_stance, fx = fy = 0  (keeps the robot up for one step).
* Time each solve:  t0 = time.perf_counter(); ...; time.perf_counter() - t0
  (must be << dt_mpc on the real robot; in sim it only slows the run).
* Assert shapes at the entry of solve(): X_ref (h+1, 13), contacts (h, 4),
  r_feet (h, 4, 3). Most bugs in sim/ were silent shape/transpose mix-ups.
* mu in the MPC: pyramid inside the simulator's cone -> mu_sim / sqrt(2).
* fz_max is a FORCE limit [N], not a torque limit.
* Return a small result object (MPCSol): forces (h, 4, 3), X_pred (h, 13),
  status, solve_time  -> log status every step, plot X_pred vs the real state.
"""


def example_K():
    _title("K. fallback forces")
    contacts = np.array([True, False, False, True])
    n = contacts.sum()
    fz = contacts * (15.0 * 9.81) / max(n, 1)
    print("fallback fz per leg for a 15 kg robot:", np.round(fz, 2))
    t0 = time.perf_counter()
    solve_qp(2 * np.eye(2), np.array([-2.0, -4.0]), sparse.csc_matrix([[1.0, 1.0]]),
             np.array([-np.inf]), np.array([2.0]))
    print(f"one tiny OSQP solve took {1e3 * (time.perf_counter() - t0):.3f} ms")


EXAMPLES = dict(A=example_A, B=example_B, C=example_C, D=example_D, E=example_E,
                F=example_F, G=example_G, H=example_H, I=example_I, J=example_J,
                K=example_K)

if __name__ == "__main__":
    for key in (sys.argv[1:] or EXAMPLES):
        EXAMPLES[key.upper()]()
