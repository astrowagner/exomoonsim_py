"""Tests for the orbit fit and the fixed-period sine fit."""
import numpy as np
from exomoonsim.orbits import orbit_xy
from exomoonsim.fit import fit_planet_orbit, sine_fit_fixed_period


def test_orbit_fit_recovers_truth():
    """From a 5%-off start, the fit should recover injected elements closely."""
    rng = np.random.default_rng(0)
    truth = np.array([2.0, 0.3, 0.25, 1.4, 45.0, 150.0, 120.0])   # P,T0,e,a,i,Om,om
    t = np.linspace(0.0, 6.0, 400)
    x, y, _ = orbit_xy(truth[3], truth[2], truth[0], truth[4],
                       truth[6], truth[5], truth[1], t)
    sigma = 1e-5
    xo = x + rng.normal(0, sigma, t.size)
    yo = y + rng.normal(0, sigma, t.size)
    guess = truth * (1 + rng.normal(0, 0.05, 7))
    fit = fit_planet_orbit(t, xo, yo, sigma, guess)
    assert fit.success
    assert abs(fit.period - truth[0]) < 1e-3
    assert abs(fit.ecc - truth[2]) < 0.02
    assert abs(fit.a - truth[3]) < 0.01
    assert abs(fit.inc - truth[4]) < 1.0


def test_sine_fit_is_exact():
    """Fixed-period linear sine fit recovers amplitude and offset."""
    x = np.linspace(0, 30, 300)
    amp, off, period = 2.5, 4.0, 5.0
    y = amp * np.sin(2 * np.pi * x / period + 0.9) + off
    s = sine_fit_fixed_period(x, y, period)
    assert abs(s.amp - amp) < 1e-6
    assert abs(s.offset - off) < 1e-6
    assert np.max(np.abs(s.model - y)) < 1e-6
