"""Smoke tests for plotting + per-trial diagnostics (Agg backend, no display)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
from exomoonsim.sim import run_trial
from exomoonsim.plots import plot_trial
from conftest import fast_params


def test_run_trial_diagnostics_present_and_shaped():
    r = run_trial(fast_params(moon_a=20.0, moon_mass=5.0), seed=1,
                  return_diagnostics=True)
    assert "diag" in r
    d = r["diag"]
    for k in ("tdays", "x_obs", "y_obs", "x_true", "y_true", "x_fit", "y_fit",
              "rres", "period_grid", "periodogram", "fold_phase", "fold_amp",
              "fold_model", "best_period", "true_synodic"):
        assert k in d, f"missing diagnostic key: {k}"
    # matched lengths where they must align
    assert d["tdays"].shape == d["rres"].shape == d["x_obs"].shape
    assert d["period_grid"].shape == d["periodogram"].shape
    assert d["fold_phase"].shape == d["fold_model"].shape == d["fold_amp"].shape
    assert np.isfinite(d["true_synodic"]) and d["true_synodic"] > 0


def test_diagnostics_off_by_default():
    r = run_trial(fast_params(moon_a=20.0, moon_mass=5.0), seed=1)
    assert "diag" not in r          # fast survey path is unchanged


def test_plot_trial_returns_figure():
    r = run_trial(fast_params(moon_a=20.0, moon_mass=5.0), seed=1,
                  return_diagnostics=True)
    fig = plot_trial(r["diag"])
    assert fig is not None and len(fig.axes) >= 4
