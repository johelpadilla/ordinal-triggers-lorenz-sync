"""Two-layer master-slave Lorenz synchronisation bench.

A trigger layer decides when the control is recomputed; a correction layer
(saturated sliding-mode law) computes it from the measured error; the control is
held between events (zero-order hold). This module provides the plant, the
sensor scenarios, the controller and the three baseline policies
A (continuous), B (Girard-type dynamic magnitude trigger) and
C (naive ordinal level trigger).

The sliding-mode controller and the dynamic-threshold mechanism are a generic
re-implementation inspired by Taghavifar et al. (2026),
doi:10.1007/s11768-026-00360-3; they are not the law of that paper. The dynamic
trigger follows the structure of Girard (2015), doi:10.1109/TAC.2014.2366855.
"""
from __future__ import annotations

import numpy as np

from cancellation import cancellation

# ------------------------------------------------------------------ parameters
# Master: canonical Lorenz; slave: about 10 % parameter mismatch.
SIGMA_M, RHO_M, BETA_M = 10.0, 28.0, 8.0 / 3.0
SIGMA_S, RHO_S, BETA_S = 11.0, 30.8, (8.0 / 3.0) * 1.10
DIST_AMP = 0.5  # bounded disturbance on the slave

DT = 0.01
T_FINAL = 20.0
N_STEPS = int(round(T_FINAL / DT))  # 2000
T_TRANS = 6.0  # metrics are computed after the transient
I_TRANS = int(round(T_TRANS / DT))
E_BAND = 2.0  # settling band on the true error
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))

# Sliding surface s = e + k1 sig(e)^a + k2 sig(e)^b, 0 < a < 1 < b, adaptive gain.
K1, K2 = 0.8, 0.8
ALPHA, BETA_P = 0.5, 1.5
K_LIN = 8.0
MU_SMOOTH = 0.30
ETA0 = 8.0
GAMMA_ETA = 1.5
ETA_MAX = 60.0
U_MAX = 40.0  # actuator saturation

# Policy B: dynamic magnitude trigger with a Girard-type internal variable.
SIGMA_B = 0.50
EPS0_B = 0.60
LAMBDA_B = 2.0
THETA_B = 1.0
ETA_D0 = 0.5

# Policy C: naive ordinal level trigger (window W, evaluated every PASO samples).
W = 20
PASO = 2
THETA_TAU = 0.95
B_EDGE = np.array([1.0, 1.0, 1.0])

# Sensors
NOISE_STD = 0.80
DRIFT_RATE = 0.25  # state units per time unit, per channel

SCENARIOS = ("clean", "noise", "distortion", "drift")
CONDITIONS = ("A", "B", "C")


# ------------------------------------------------------------------ plant and control
def f_lorenz(v, sigma, rho, beta):
    x, y, z = v
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float)


def sig_pow(e, p):
    return np.sign(e) * np.abs(e) ** p


def sliding_s(e):
    return e + K1 * sig_pow(e, ALPHA) + K2 * sig_pow(e, BETA_P)


def compute_u(e, eta):
    s = sliding_s(e)
    u = -eta * np.tanh(s / MU_SMOOTH) - K_LIN * s
    u = np.clip(u, -U_MAX, U_MAX)
    return u, s


def update_eta(eta, s):
    eta = eta + DT * GAMMA_ETA * float(np.sum(np.abs(s)))
    return min(eta, ETA_MAX)


def disturbance(t):
    return DIST_AMP * np.array(
        [np.sin(1.1 * t), np.cos(0.9 * t), np.sin(0.6 * t + 0.3)], dtype=float
    )


def rk4_step(v, dt, sigma, rho, beta, add):
    def f(x):
        return f_lorenz(x, sigma, rho, beta) + add

    k1 = f(v)
    k2 = f(v + 0.5 * dt * k1)
    k3 = f(v + 0.5 * dt * k2)
    k4 = f(v + dt * k3)
    return v + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# ------------------------------------------------------------------ Kendall tau_b
def tau_b_fast(x, y):
    """Kendall tau_b (identical to scipy.stats.kendalltau, variant='b')."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.nan
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    iu = np.triu_indices(n, 1)
    sx = np.sign(dx[iu])
    sy = np.sign(dy[iu])
    p = float(np.dot(sx, sy))
    n0 = n * (n - 1) // 2
    tx = int(np.sum(sx == 0))
    ty = int(np.sum(sy == 0))
    den = np.sqrt((n0 - tx) * (n0 - ty))
    if den == 0.0:
        return np.nan
    return p / den


def tau_pairs(xm_win, ys_win):
    """tau_b on the three paired edges (x, x'), (y, y'), (z, z')."""
    return np.array(
        [tau_b_fast(xm_win[:, i], ys_win[:, i]) for i in range(3)], dtype=float
    )


def measure(xs, t, scenario, noise_row):
    if scenario == "clean":
        return xs.copy()
    if scenario == "noise":
        return xs + noise_row
    if scenario == "distortion":
        # strictly increasing distortion of comparable scale
        return xs + 10.0 * np.tanh(xs / 15.0)
    if scenario == "drift":
        return xs + DRIFT_RATE * t
    raise ValueError(scenario)


# ------------------------------------------------------------------ one trajectory
def simulate(seed, scenario, condition):
    rng = np.random.default_rng(1000 + int(seed))
    xm = rng.uniform([-10.0, -10.0, 10.0], [10.0, 10.0, 30.0])
    xs = rng.uniform([-10.0, -10.0, 10.0], [10.0, 10.0, 30.0])
    noise = np.random.default_rng(50_000 + int(seed)).normal(0.0, NOISE_STD, size=(N_STEPS + 1, 3))

    t = 0.0
    eta = float(ETA0)
    eta_d = float(ETA_D0)

    ys = measure(xs, t, scenario, noise[0])
    e_meas = ys - xm
    u, s = compute_u(e_meas, eta)
    eta = update_eta(eta, s)
    e_last = e_meas.copy()
    n_updates = 1
    event_steps = [0]

    hist_xm = np.zeros((W, 3))
    hist_ys = np.zeros((W, 3))
    hist_xm[0] = xm
    hist_ys[0] = ys
    n_hist = 1

    buf_xm = np.zeros((N_STEPS + 1, 3))
    buf_ys = np.zeros((N_STEPS + 1, 3))
    buf_xm[0] = xm
    buf_ys[0] = ys

    err_true = np.empty(N_STEPS + 1)
    err_meas_s = np.empty(N_STEPS + 1)
    err_true[0] = float(np.linalg.norm(xs - xm))
    err_meas_s[0] = float(np.linalg.norm(e_meas))
    effort_acc = float(np.dot(u, u)) * DT
    diverged = False

    for k in range(1, N_STEPS + 1):
        t = k * DT
        d = disturbance(t)
        # disturbance and control are held constant over the step (ZOH)
        xm = rk4_step(xm, DT, SIGMA_M, RHO_M, BETA_M, np.zeros(3))
        xs = rk4_step(xs, DT, SIGMA_S, RHO_S, BETA_S, u + d)
        if not np.all(np.isfinite(xs)) or np.linalg.norm(xs) > 1e3 or np.linalg.norm(xm) > 1e3:
            diverged = True
            err_true[k:] = np.nan
            err_meas_s[k:] = np.nan
            break

        ys = measure(xs, t, scenario, noise[k])
        e_meas = ys - xm
        e_true = xs - xm
        err_true[k] = float(np.linalg.norm(e_true))
        err_meas_s[k] = float(np.linalg.norm(e_meas))
        buf_xm[k] = xm
        buf_ys[k] = ys
        effort_acc += float(np.dot(u, u)) * DT

        if n_hist < W:
            hist_xm[n_hist] = xm
            hist_ys[n_hist] = ys
            n_hist += 1
        else:
            hist_xm[:-1] = hist_xm[1:]
            hist_ys[:-1] = hist_ys[1:]
            hist_xm[-1] = xm
            hist_ys[-1] = ys

        fire = False
        if condition == "A":
            fire = True
        elif condition == "B":
            eps_vec = e_meas - e_last
            norm_eps = float(np.linalg.norm(eps_vec))
            norm_e = float(np.linalg.norm(e_meas))
            slack = SIGMA_B * norm_e + EPS0_B - norm_eps
            eta_d = max(0.0, eta_d + DT * (-LAMBDA_B * eta_d + slack))
            # Girard-type dynamic rule: fire when internal variable plus weighted slack is non-positive
            if eta_d + THETA_B * slack <= 0.0:
                fire = True
        elif condition == "C":
            if n_hist >= W and (k % PASO == 0):
                taus = tau_pairs(hist_xm, hist_ys)
                if np.all(np.isfinite(taus)):
                    if float(np.mean(taus)) < THETA_TAU:
                        fire = True
        else:
            raise ValueError(condition)

        if fire:
            u, s = compute_u(e_meas, eta)
            eta = update_eta(eta, s)
            e_last = e_meas.copy()
            n_updates += 1
            event_steps.append(k)

    # metrics on the true error xs - xm
    post = err_true[I_TRANS:]
    post = post[np.isfinite(post)]
    if post.size == 0:
        rms = np.nan
        settled = False
        t_settle = np.nan
    else:
        rms = float(np.sqrt(np.mean(post ** 2)))
        below = np.isfinite(err_true) & (err_true < E_BAND)
        t_settle = np.nan
        settled = False
        # first instant after which the error stays inside the band
        for i in range(N_STEPS + 1):
            if not np.isfinite(err_true[i]):
                break
            if np.all(below[i:]):
                t_settle = i * DT
                settled = True
                break

    if post.size:
        post_m = err_meas_s[I_TRANS:]
        post_m = post_m[np.isfinite(post_m)]
        rms_meas = float(np.sqrt(np.mean(post_m ** 2))) if post_m.size else np.nan
    else:
        rms_meas = np.nan

    iets = np.diff(np.asarray(event_steps, dtype=float)) * DT if len(event_steps) >= 2 else np.array([])
    min_iet = float(np.min(iets)) if iets.size else np.nan
    med_iet = float(np.median(iets)) if iets.size else np.nan
    effort = effort_acc / T_FINAL  # time average of ||u||^2

    # post-transient tau_b and kappa diagnostics (measurement vs master)
    mean_tau_post = []
    kappa_post = []
    if not diverged:
        start = max(I_TRANS, W - 1)
        for kk in range(start, N_STEPS + 1, PASO):
            taus = tau_pairs(buf_xm[kk - W + 1: kk + 1], buf_ys[kk - W + 1: kk + 1])
            if np.all(np.isfinite(taus)):
                mean_tau_post.append(float(np.mean(taus)))
                _de, _db, _ce, kapv = cancellation(taus - B_EDGE)
                if np.isfinite(kapv):
                    kappa_post.append(float(kapv))

    return {
        "seed": int(seed),
        "scenario": scenario,
        "condition": condition,
        "rms": rms,
        "rms_meas": rms_meas,
        "t_settle": t_settle,
        "settled": int(settled),
        "n_updates": int(n_updates),
        "min_iet": min_iet,
        "med_iet": med_iet,
        "effort": effort,
        "mean_tau_post": float(np.nanmean(mean_tau_post)) if mean_tau_post else np.nan,
        "mean_kappa_post": float(np.nanmean(kappa_post)) if kappa_post else np.nan,
        "diverged": int(diverged),
        "n_events": int(len(event_steps)),
    }
