"""
Two-body orbit projection and Kepler's-equation solver.

Port of binary_star_orbit.pro / calc_Ei.pro / solve_trans.pro, modernized to
double-precision, fully vectorized numpy.  Given a set of orbital elements and a
time (or array of times), returns the on-sky (x, y) position.
"""
from __future__ import annotations

import numpy as np

__all__ = ["kepler_solve", "orbit_xy", "true_anomaly"]


def kepler_solve(M, ecc, tol=1e-10, maxiter=100):
    """Solve Kepler's equation  E - e*sin(E) = M  for the eccentric anomaly E.

    Vectorized Newton iteration with the Heintz (1978) starting guess.  ``M`` may
    be a scalar or array; ``ecc`` is a scalar.
    """
    M = np.asarray(M, dtype=float)
    # Heintz initial approximation
    E = M + ecc * np.sin(M) + 0.5 * ecc**2 * np.sin(2.0 * M)
    for _ in range(maxiter):
        dE = (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def true_anomaly(E, ecc):
    """True anomaly from eccentric anomaly (quadrant-correct)."""
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(E / 2.0),
        np.sqrt(1.0 - ecc) * np.cos(E / 2.0),
    )


def orbit_xy(semi_major_axis, ecc, period, inclination, omega, big_omega,
             t_peri, time):
    """On-sky position of a body on a Keplerian orbit.

    Parameters mirror binary_star_orbit.pro:
      semi_major_axis : semi-major axis (any length unit; sets the output unit)
      ecc             : eccentricity
      period          : orbital period (years)
      inclination     : inclination (degrees)
      omega           : argument of periapsis (degrees)
      big_omega       : longitude of ascending node (degrees)
      t_peri          : time of periastron passage (years)
      time            : time(s) at which to evaluate (years); scalar or array

    Returns
      (x, y, z) arrays (same length as ``time``); z is the line-of-sight term.
    """
    time = np.asarray(time, dtype=float)

    o = np.radians(omega)
    big_o = np.radians(big_omega)
    inc = np.radians(inclination)

    mean_anom = (2.0 * np.pi / period) * (time - t_peri)
    E = kepler_solve(mean_anom, ecc)
    nu = true_anomaly(E, ecc)
    r = semi_major_axis * (1.0 - ecc * np.cos(E))

    # position in the orbital plane (z = 0)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # inclination rotation about x
    x_rot = x_orb
    y_rot = y_orb * np.cos(inc)
    z_rot = y_orb * np.sin(inc)

    # longitude-of-node rotation about z
    x_rot2 = x_rot * np.cos(big_o) - y_rot * np.sin(big_o)
    y_rot2 = x_rot * np.sin(big_o) + y_rot * np.cos(big_o)

    # argument-of-periapsis rotation about z
    x_final = x_rot2 * np.cos(o) - y_rot2 * np.sin(o)
    y_final = x_rot2 * np.sin(o) + y_rot2 * np.cos(o)

    return x_final, y_final, z_rot
