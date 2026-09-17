"""# Tutorial 4 — Surveys: detection maps, false positives, and the mass floor

One trial tells you whether *one* moon was found. A **survey** repeats trials over
a grid of moon mass and separation and asks, for each cell, what fraction of
noise realizations yield a detection — a detectability map. This is the
workflow behind Papers I–III's survey figures and the "success maps" of the
student projects.

By the end you will be able to:

1. run a parallel survey with `SurveyConfig` / `run_survey` and read its outputs,
2. plot a detection-fraction (success) map,
3. measure the false-positive fraction and see that it lives on the detection boundary,
4. read off the mass floor per separation, and switch the survey to the matched filter.

`FAST = True` (a 3×5 grid, ~2–4 min) or `FAST = False` (the paper's 9×101 grid,
50 trials per cell at 50 µas — **hours**; see the note at the end).

**Multiprocessing note.** `run_survey` runs trials in parallel worker processes. On
macOS/Windows the worker start method re-imports the running script, so in the
`.py` version every step that touches the survey sits under
`if __name__ == "__main__":`. (In the notebook that guard is unnecessary and is
omitted.) Any survey script of your own needs the same guard — see
`examples/getting_started.py`.
"""
# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt

from exomoonsim.sim import SimParams, run_trial
from exomoonsim.survey import SurveyConfig, run_survey

FAST = True

if FAST:
    A_GRID    = np.array([5.0, 10.0, 20.0])                 # moon semimajor axis (R_Jup)
    MASS_GRID = np.array([0.0, 0.05, 0.1, 0.3, 1.0])        # moon mass (M_Earth); 0 = pure-noise control
    NTRIALS   = 4
    BASE = SimParams(astrometric_precision=2e-5, texp=6.0, pend=40.0, ptestwidth=0.01)
    NCTRL = 24
else:
    A_GRID    = np.array([5, 6, 7, 8, 9, 10, 15, 20, 25], float)
    MASS_GRID = np.round(np.arange(101) * 0.01, 2)
    NTRIALS   = 50
    BASE = SimParams(astrometric_precision=5e-5)             # paper: 50 uas, hourly, pend=500
    NCTRL = 300

VIR = plt.get_cmap("viridis")


def plot_map(ax, res, cube, title, vmax=1.0, fmt="%.0f", scale=100):
    """Grid map with separation on x, mass on y, and per-cell percentages (a 'success map')."""
    aa, mm = res.config.a_grid, res.config.mass_grid
    def edges(v):
        v = np.asarray(v, float); v = np.where(v > 0, v, v[v > 0].min() / 3)   # log axes; nudge mass=0
        lv = np.log10(v); mid = (lv[1:] + lv[:-1]) / 2
        return 10 ** np.r_[2 * lv[0] - mid[0], mid, 2 * lv[-1] - mid[-1]]
    mplot = np.where(mm > 0, mm, mm[mm > 0].min() / 3)
    pcm = ax.pcolormesh(edges(aa), edges(mm), cube.T, cmap="viridis", vmin=0, vmax=vmax, shading="flat")
    for i, a in enumerate(aa):
        for j, m in enumerate(mplot):
            ax.text(a, m, fmt % (scale * cube[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cube[i, j] < 0.55 * vmax else "black")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(aa); ax.set_xticklabels(["%g" % x for x in aa])
    ax.set_yticks(mplot); ax.set_yticklabels(["%g" % m for m in mm])
    ax.set_xlabel(r"Moon semimajor axis ($R_{\rm Jup}$)"); ax.set_ylabel(r"Moon mass ($M_\oplus$)")
    ax.set_title(title, fontsize=10)
    return pcm

# %% [markdown]
# ## 1. Run a survey
#
# `SurveyConfig` bundles the grid, the trials per cell, and a `base` `SimParams`
# holding everything else (precision, cadence, search settings). `run_survey`
# farms the trials out to all your CPU cores and returns a `SurveyResult` with, per
# cell: the detection fraction, the median significance, period and amplitude
# errors, and the false-positive fraction. The mass = 0 row is a deliberate
# **pure-noise control**: with no moon injected, anything it "detects" is a false
# alarm.

# %%
if __name__ == "__main__":
    cfg = SurveyConfig(a_grid=A_GRID, mass_grid=MASS_GRID, ntrials=NTRIALS, base=BASE)
    survey = run_survey(cfg)                                   # progress prints every ~5%
    survey.save("tutorial_survey.npz")                          # the '.sav'-equivalent: reload any time
    print("\ndetfrac shape (a x mass):", survey.detfrac.shape)
    print("detection cut: dchi >", survey.config.chsq_thr)

# %% [markdown]
# ## 2. The detection (success) map
#
# Each cell is the fraction of trials in which the moon was detected — the
# significance cleared the cut *and* the recovered period was right. Separation
# increases along x, mass along y.

# %%
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    pcm = plot_map(ax, survey, survey.detfrac, "Detection fraction (%%)  --  %.0f µas, single sinusoid"
                   % (BASE.astrometric_precision * 1e6))
    fig.colorbar(pcm, ax=ax, label="detection fraction"); plt.tight_layout(); plt.show()

# %% [markdown]
# Heavier and wider moons (larger reflex m·a) are detected more often. With only a
# few trials per cell the fractions are quantized (here to 25%); the paper uses 50
# trials per cell to smooth that out.
#
# ## 3. False positives: detections at the wrong period
#
# A **false positive** is a trial that clears the cut but locks onto a noise peak —
# its recovered period is more than 5% from the truth. The false-positive fraction
# is the natural complement to completeness: it tells you how reliable a claimed
# detection is. Two things to look for: the mass = 0 control row *is* the
# pure-noise false-alarm rate, and spurious detections concentrate along the
# detection boundary — securely detected moons essentially never fire at the wrong
# period (Paper III §4.4).

# %%
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    pcm = plot_map(ax, survey, np.nan_to_num(survey.fpcube), "False-positive fraction (%)", vmax=0.5)
    fig.colorbar(pcm, ax=ax, label="false-positive fraction"); plt.tight_layout(); plt.show()
    fp = np.nan_to_num(survey.fpcube)
    n_ctrl = NTRIALS * len(A_GRID)
    print("pure-noise false-alarm rate (mass=0 row): %.2f  from %d control trials "
          "(the paper's rate is a few percent, so expect ~%.1f false alarms here)"
          % (fp[:, 0].mean(), n_ctrl, 0.04 * n_ctrl))
    print("false-positive fraction where detfrac >= 0.75: %.2f" % fp[survey.detfrac >= 0.75].mean())

# %% [markdown]
# ## 4. The mass floor, and how it scales
#
# Reading down each column, the **mass floor** is the smallest mass detected in at
# least half the trials. The paper maps this floor against per-epoch precision and
# separation (Paper III Fig. 13): it scales linearly with precision
# (m_min ∝ σ), and — with the matched filter — strongly with separation
# (m_min ∝ a^−0.8), reaching sub-lunar masses at few-µas precision. That
# figure is `paper/make_precision_scaling_fig.py` (bisection in mass at each cell).

# %%
if __name__ == "__main__":
    print("mass floor (detfrac >= 0.5) per separation:")
    for i, a in enumerate(A_GRID):
        ok = np.where(survey.detfrac[i] >= 0.5)[0]
        print("   a = %4.0f R_Jup:  m_min = %s M_Earth" % (a, ("%.2f" % MASS_GRID[ok[0]]) if ok.size else "none"))

# %% [markdown]
# ## 5. Switching the survey to the matched filter
#
# Everything above used the Papers I/II single-sinusoid statistic. To survey with
# the projected-ellipse matched filter of Paper III, set `stat="ellipse"` — but the
# cut must first be **calibrated** to the same pure-noise false-alarm rate as the
# sinusoid's Δχ² > 5 (Tutorial 2). Then compare the two maps: the matched filter
# deepens the floor most at wide separation (Paper III §5).

# %%
def calibrate_cut(base, nctrl, sine_cut=5.0):
    """Matched-filter threshold matching the single-sinusoid Dchi^2 > sine_cut false-alarm rate."""
    noise = SimParams(**{**{f: getattr(base, f) for f in ("astrometric_precision", "texp", "pend", "ptestwidth")},
                         "moon_a": 10.0, "moon_mass": 0.0})
    sine = np.array([run_trial(noise, seed=k, stat="sine")["sig"] for k in range(nctrl)])
    ell  = np.array([run_trial(noise, seed=10_000 + k, stat="ellipse")["sig"] for k in range(nctrl)])
    far = min(max(float(np.mean(sine > sine_cut)), 1.0 / nctrl), 0.5)
    return float(np.quantile(ell, 1.0 - far)), far

# %%
if __name__ == "__main__":
    thr, far = calibrate_cut(BASE, NCTRL)
    print("calibrated matched-filter cut: dchi_mf > %.1f  (matched false-alarm rate %.3f)" % (thr, far))
    cfg_mf = SurveyConfig(a_grid=A_GRID, mass_grid=MASS_GRID, ntrials=NTRIALS, base=BASE,
                          stat="ellipse", chsq_thr=thr)
    survey_mf = run_survey(cfg_mf)
    survey_mf.save("tutorial_survey_ellipse.npz")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    plot_map(ax[0], survey, survey.detfrac, "Single sinusoid: detection fraction (%)")
    pcm = plot_map(ax[1], survey_mf, survey_mf.detfrac, "Matched filter (calibrated cut): detection fraction (%)")
    ax[1].set_ylabel(""); fig.colorbar(pcm, ax=ax, label="detection fraction", fraction=0.03)
    plt.show()
    print("cells detected (detfrac >= 0.5):  sinusoid %d   matched filter %d"
          % ((survey.detfrac >= 0.5).sum(), (survey_mf.detfrac >= 0.5).sum()))

# %% [markdown]
# ## Take-aways
#
# * A survey = many trials per (mass, separation) cell → a detection-fraction map;
#   `SurveyResult.save` keeps everything so you can re-plot without re-running.
# * Always include a mass = 0 control: it measures the false-alarm rate, and the
#   false-positive map shows spurious detections confined to the detection boundary.
# * The mass floor scales linearly with precision and, with the matched filter,
#   strongly with separation.
# * The matched filter needs a calibrated cut; at matched false-alarm rate it
#   detects more of the faint, wide-separation cells.
#
# **Scaling to the paper.** `FAST = False` selects the 9×101 grid at 50 µas with 50
# trials per cell (45,450 trials): the single-sinusoid survey is
# `run_50muas.py` (tens of minutes on a many-core machine), the matched-filter
# version `run_50muas_ellipse.py` (much longer — the ellipse fit is ~10–50× the cost
# per trial). The student-project driver `run_sumin_survey.py` runs the
# mass × separation success map at a chosen precision with the same machinery.
