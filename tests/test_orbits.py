"""Tests for the orbit physics (Kepler solver + projection)."""
import numpy as np
from exomoonsim.orbits import kepler_solve, orbit_xy, true_anomaly


def test_kepler_residual_small():
    """E - e sin(E) - M should be ~0 across the full range of M and e."""
    M = np.linspace(0, 2 * np.pi, 2000)
    for e in (0.0, 0.1, 0.5, 0.9):
        E = kepler_solve(M, e)
        assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-9


def test_circular_orbit_is_constant_radius():
    """A circular, face-on orbit has constant on-sky separation = a."""
    t = np.linspace(0, 5, 200)
    x, y, z = orbit_xy(1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, t)
    r = np.hypot(x, y)
    assert np.allclose(r, 1.0, atol=1e-9)
    assert np.allclose(z, 0.0, atol=1e-9)      # face-on -> no line-of-sight term


def test_separation_within_apo_peri():
    """Projected separation never exceeds apoapsis a(1+e)."""
    t = np.linspace(0, 3, 500)
    a, e = 1.8, 0.3
    x, y, _ = orbit_xy(a, e, 1.5, 45.0, 150.0, 150.0, 0.3, t)
    assert np.max(np.hypot(x, y)) <= a * (1 + e) + 1e-9


def test_scalar_time_ok():
    """A scalar time works and returns finite coordinates (0-d, numpy-style)."""
    x, y, z = orbit_xy(1.0, 0.2, 1.0, 30.0, 10.0, 20.0, 0.0, 0.5)
    assert np.isfinite(float(np.asarray(x)))
    assert np.isfinite(float(np.asarray(y)))


def test_true_anomaly_zero_at_periapsis():
    assert abs(true_anomaly(np.array([0.0]), 0.3)[0]) < 1e-12
