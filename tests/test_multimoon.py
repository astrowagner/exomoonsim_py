"""Tests for multiple moons (Phase 1: simulate + primary scoring)."""
import numpy as np
from exomoonsim.sim import SimParams, Moon, run_trial
from conftest import fast_params


def test_scalar_equals_single_moon_list():
    """A scalar moon and the equivalent one-element `moons` list are identical."""
    a = run_trial(fast_params(moon_a=20.0, moon_mass=3.0), seed=1)
    p = fast_params()
    p.moons = [Moon(a=20.0, mass=3.0, inc=p.moon_inc, ecc=p.moon_ecc,
                    omega=p.moon_omega, bigomega=p.moon_bigomega, t0=p.moon_t0)]
    b = run_trial(p, seed=1)
    assert a["sig"] == b["sig"] and a["perr"] == b["perr"]
    assert a["moon_period"] == b["moon_period"] and a["input_amp"] == b["input_amp"]


def test_primary_is_largest_m_times_a():
    """Primary = largest reflex (m*a), independent of list order."""
    p = fast_params()
    p.moons = [Moon(a=6.0, mass=0.5), Moon(a=20.0, mass=3.0), Moon(a=10.0, mass=0.2)]
    assert p.primary_index() == 1                 # reflex ~ 3, 60, 2
    p.moons = [Moon(a=20.0, mass=3.0), Moon(a=6.0, mass=0.5)]
    assert p.primary_index() == 0
    p.moons = [Moon(a=6.0, mass=0.5), Moon(a=20.0, mass=3.0)]
    assert p.primary_index() == 1


def test_recovery_scored_against_primary():
    """With two moons, recovery is scored against the strong (primary) one."""
    p = fast_params()
    strong, weak = Moon(a=20.0, mass=5.0), Moon(a=8.0, mass=0.2)
    p.moons = [weak, strong]
    r = run_trial(p, seed=1, return_diagnostics=True)
    assert abs(r["moon_period"] - p._period_days(strong)) < 1e-9
    assert r["perr"] < 5.0                         # primary recovered
    assert len(r["diag"]["companion_synodics"]) == 1


def test_faint_companion_keeps_primary():
    """A tiny companion doesn't change the primary or kill detection."""
    solo = run_trial(fast_params(moon_a=20.0, moon_mass=5.0), seed=3)
    p = fast_params(moon_a=20.0, moon_mass=5.0)
    p.moons = [Moon(a=20.0, mass=5.0), Moon(a=7.0, mass=0.1)]
    withc = run_trial(p, seed=3)
    assert withc["sig"] > 5.0
    assert abs(withc["moon_period"] - solo["moon_period"]) < 1e-9
