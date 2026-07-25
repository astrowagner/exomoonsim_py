"""Tests for the false-positive (spurious-period) analysis."""
import matplotlib
matplotlib.use("Agg")

import numpy as np

from exomoonsim.survey import false_positive_frac, SurveyResult, SurveyConfig
from exomoonsim.plots import plot_survey


def test_false_positive_frac_counts_spurious():
    # one cell, 4 trials of (sig, perr, amperr)
    trials = np.array([[[
        [10.0, 1.0, 5.0],           # sig>5, perr<5   -> true detection (not FP)
        [10.0, 20.0, 5.0],          # sig>5, perr>=5  -> FALSE POSITIVE
        [2.0, 30.0, 5.0],           # sig<5           -> neither
        [np.nan, np.nan, np.nan],   # failed fit      -> neither
    ]]])
    fp = false_positive_frac(trials, chsq_thr=5.0, perr_thr=5.0)
    assert fp.shape == (1, 1)
    assert abs(fp[0, 0] - 0.25) < 1e-9       # exactly 1 of 4


def test_detection_and_false_positive_are_exclusive():
    rng = np.random.default_rng(0)
    trials = rng.uniform(0.0, 30.0, size=(2, 3, 25, 3))
    sig, perr, amp = trials[..., 0], trials[..., 1], trials[..., 2]
    det = np.mean((sig > 5) & (perr < 5) & (amp < 25), axis=2)
    fp = false_positive_frac(trials, 5.0, 5.0)
    assert np.all(det + fp <= 1.0 + 1e-9)    # a trial can't be both


def test_plot_survey_has_six_panels():
    a = np.array([5.0, 10.0, 20.0]); m = np.array([0.1, 1.0, 5.0])
    base = np.linspace(0.1, 0.9, 9).reshape(3, 3)
    res = SurveyResult(
        a_grid=a, mass_grid=m, sigcube=base * 100, perrcube=base * 10,
        amperrcube=base * 20, detfrac=base, fpcube=base * 0.3,
        trials=np.zeros((3, 3, 4, 3)), ntrials=4,
        config=SurveyConfig(a_grid=a, mass_grid=m, ntrials=4))
    fig = plot_survey(res)
    assert len(fig.axes) >= 6                # six map panels (+ colorbars)
