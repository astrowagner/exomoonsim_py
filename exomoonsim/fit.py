"""
Orbit and sine fitting (numpy-only, no scipy dependency).

- fit_planet_orbit: fits a 7-element Keplerian orbit to astrometric (separation,
  position-angle) data, using a compact bounded Levenberg-Marquardt solver.
  Replaces the hand-rolled Newton-Raphson/Marquardt fitter (newt_raph_func.pro +
  calc_deriv_vb.pro).

- sine_fit_fixed_period: fits y = amp*sin(2*pi*x/P + phase) + offset at a FIXED
  period P.  Frequency fixed -> the model is linear -> solved exactly by linear
  least squares (replaces sinefitfp.pro / curvefit).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from .orbits import orbit_xy

__all__ = ["model_rho_theta", "fit_planet_orbit", "sine_fit_fixed_period",
           "levmar", "OrbitFit", "SineFit"]

OrbitFit = namedtuple("OrbitFit", "period t_peri ecc a inc big_omega omega success chi2 nfev")
SineFit = namedtuple("SineFit", "amp phase offset model")


# --------------------------------------------------------------------------- #
#  Generic bounded Levenberg-Marquardt least-squares
# --------------------------------------------------------------------------- #
def levmar(resid, p0, lower, upper, max_iter=200, tol=1e-12, eps=1e-7):
    """Minimize sum(resid(p)**2) with box bounds via Levenberg-Marquardt.

    resid : callable p -> residual vector
    p0    : initial parameters
    lower, upper : per-parameter bounds (same length as p0)
    Returns (p, chi2, nfev, success).
    """
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    p = np.clip(np.asarray(p0, float), lower, upper)
    r = np.asarray(resid(p), float)
    cost = float(r @ r)
    nfev = 1
    lam = 1e-3
    n = p.size
    success = False

    for _ in range(max_iter):
        # forward-difference Jacobian
        J = np.empty((r.size, n))
        for j in range(n):
            step = eps * max(1.0, abs(p[j]))
            pj = p.copy()
            pj[j] = min(pj[j] + step, upper[j])
            step = pj[j] - p[j]
            if step == 0.0:            # hit the bound; step the other way
                pj[j] = max(p[j] - eps * max(1.0, abs(p[j])), lower[j])
                step = pj[j] - p[j]
            J[:, j] = (np.asarray(resid(pj)) - r) / step
            nfev += 1

        JTJ = J.T @ J
        JTr = J.T @ r
        diag = np.diag(JTJ).copy()
        diag[diag == 0] = 1.0

        stepped = False
        for _ in range(30):
            try:
                dp = np.linalg.solve(JTJ + lam * np.diag(diag), -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            p_new = np.clip(p + dp, lower, upper)
            r_new = np.asarray(resid(p_new), float)
            nfev += 1
            cost_new = float(r_new @ r_new)
            if cost_new < cost:
                dmax = np.max(np.abs(p_new - p))
                p, r, cost = p_new, r_new, cost_new
                lam = max(lam / 10.0, 1e-12)
                stepped = True
                if dmax < tol * (1.0 + np.max(np.abs(p))):
                    success = True
                    return p, cost, nfev, success
                break
            lam *= 10.0
            if lam > 1e12:
                break
        if not stepped:
            success = True     # converged (no improving step found)
            break

    return p, cost, nfev, success


# --------------------------------------------------------------------------- #
#  Orbit fit
# --------------------------------------------------------------------------- #
def model_rho_theta(params, time):
    """(separation, position-angle in deg) predicted by a 7-element orbit."""
    period, t_peri, ecc, a, inc, big_omega, omega = params
    x, y, _ = orbit_xy(a, ecc, period, inc, omega, big_omega, t_peri, time)
    rho = np.hypot(x, y)
    theta = np.degrees(np.arctan2(y, x)) % 360.0
    return rho, theta


def fit_planet_orbit(time, x_obs, y_obs, sigma, guess, max_iter=200):
    """Fit a Keplerian orbit to astrometric data, in Cartesian (x, y).

    Cartesian residuals are far smoother than (separation, position-angle) ones
    -- no hypot/atan2 stiffness or angle wrapping -- and the measurement noise is
    added to x, y in the first place, so this is the natural, well-conditioned fit.

    time         : observation times (years)
    x_obs, y_obs : measured on-sky coordinates (arcsec)
    sigma        : 1-sigma astrometric error (scalar or array, arcsec)
    guess        : (period, t_peri, ecc, a, inc, big_omega, omega) start
    Returns an OrbitFit namedtuple.
    """
    x_obs = np.asarray(x_obs, float)
    y_obs = np.asarray(y_obs, float)
    sigma = np.broadcast_to(sigma, x_obs.shape)

    def resid(p):
        period, t_peri, ecc, a, inc, big_omega, omega = p
        xm, ym, _ = orbit_xy(a, ecc, period, inc, omega, big_omega, t_peri, time)
        return np.concatenate([(x_obs - xm) / sigma, (y_obs - ym) / sigma])

    lower = np.array([1e-6, -np.inf, 0.0, 1e-9, 0.0, -720.0, -720.0])
    upper = np.array([np.inf, np.inf, 0.95, np.inf, 180.0, 1080.0, 1080.0])

    p, cost, nfev, ok = levmar(resid, guess, lower, upper, max_iter=max_iter)
    if not np.all(np.isfinite(p)):
        return OrbitFit(*([np.nan] * 7), success=False, chi2=np.nan, nfev=nfev)
    return OrbitFit(period=p[0], t_peri=p[1], ecc=p[2], a=p[3], inc=p[4],
                    big_omega=p[5] % 360.0, omega=p[6] % 360.0,
                    success=ok, chi2=cost, nfev=nfev)


# --------------------------------------------------------------------------- #
#  Fixed-period sine fit (exact, linear)
# --------------------------------------------------------------------------- #
def sine_fit_fixed_period(x, y, period, want_model=True):
    """Exact linear least-squares sine fit at fixed period.

    y ~ amp*sin(w*x + phase) + offset,  w = 2*pi/period.  Solved directly from the
    3x3 normal equations (faster than a general lstsq in the period-search loop).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 2.0 * np.pi / period
    s = np.sin(w * x)
    c = np.cos(w * x)
    n = x.size
    # normal-equations matrix for design columns [s, c, 1]
    M = np.array([
        [s @ s, s @ c, s.sum()],
        [s @ c, c @ c, c.sum()],
        [s.sum(), c.sum(), float(n)],
    ])
    b = np.array([s @ y, c @ y, y.sum()])
    try:
        c_sin, c_cos, offset = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        c_sin, c_cos, offset = np.linalg.lstsq(M, b, rcond=None)[0]
    amp = float(np.hypot(c_sin, c_cos))
    phase = float(np.arctan2(c_cos, c_sin))
    model = (c_sin * s + c_cos * c + offset) if want_model else None
    return SineFit(amp=amp, phase=phase, offset=float(offset), model=model)
