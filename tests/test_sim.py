"""Tests for the single-trial simulator (fast config)."""
import numpy as np
from exomoonsim.sim import run_trial
from conftest import fast_params


def test_trial_outputs_finite_and_sane():
    """A trial returns finite significance / errors and recovers the moon period."""
    p = fast_params(moon_a=20.0, moon_mass=5.0)   # strong signal
    r = run_trial(p, seed=1)
    assert np.isfinite(r["sig"])
    assert np.isfinite(r["perr"]) and r["perr"] >= 0
    assert np.isfinite(r["amperr"]) and r["amperr"] >= 0
    assert r["fit_success"]
    # sidereal period recovered to within a few percent of truth
    assert abs(r["sidereal_period"] - r["moon_period"]) / r["moon_period"] < 0.05


def test_strong_moon_is_significant():
    p = fast_params(moon_a=20.0, moon_mass=5.0)
    r = run_trial(p, seed=2)
    assert r["sig"] > 5.0


def test_weak_moon_less_significant_than_strong():
    """A tiny moon produces far lower significance than a big one."""
    strong = run_trial(fast_params(moon_a=20.0, moon_mass=5.0), seed=3)
    weak = run_trial(fast_params(moon_a=20.0, moon_mass=0.02), seed=3)
    assert weak["sig"] < strong["sig"]


def test_reproducible_with_seed():
    p = fast_params(moon_a=20.0, moon_mass=5.0)
    a = run_trial(p, seed=7)
    b = run_trial(p, seed=7)
    assert a["sig"] == b["sig"] and a["perr"] == b["perr"]


def test_zero_mass_moon_no_crash():
    """Zero-mass moon (no-signal control cell): no divide-by-zero; amp error is inf."""
    p = fast_params(moon_a=10.0, moon_mass=0.0)
    r = run_trial(p, seed=1)
    assert r["input_amp"] == 0.0
    assert np.isinf(r["amperr"])      # fails the amplitude cut -> not "detected"
    assert np.isfinite(r["sig"])      # significance is still computable
