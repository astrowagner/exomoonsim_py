"""# Tutorial 2 — Recovering several moons: iterative prewhitening and the calibrated cut

Real satellite systems have several moons. Their astrometric signals superpose
linearly (Paper III Eq. 1), so we can recover them one at a time: find the
strongest, subtract its fitted model, look again. This is **iterative
prewhitening**.

By the end you will be able to:

1. **calibrate** the matched-filter detection cut to a false-alarm rate,
2. run a blind multi-moon recovery and read its round-by-round periodograms,
3. reproduce the paper's headline: at a precision where a single sinusoid finds
   only one of four moons, the matched filter finds all four.

`FAST = True` runs in a few minutes; `FAST = False` is the paper configuration
(Paper III §4.1–4.2, Figs. 3–6; the four-moon step then takes ~10 min).
"""
# %% [markdown]
# ## Setup

# %%
import numpy as np
import matplotlib.pyplot as plt

from exomoonsim.sim import SimParams, Moon, run_trial

FAST = True

if FAST:
    # 6-hr cadence. NOTE: at 6-hr cadence, 20 uas gives the same signal-to-noise per campaign
    # as the paper's 50 uas hourly run (fewer epochs, each more precise) -- so the four-moon
    # comparison below reproduces the paper's 50 uas result in a fraction of the time.
    SEARCH = dict(texp=6.0, pend=45.0, ptestwidth=0.01)
    PREC_TWO, PREC_FOUR = 1e-5, 2e-5
    NCTRL = 24
else:
    SEARCH = dict(texp=1.0, pend=60.0, ptestwidth=0.0015)
    PREC_TWO, PREC_FOUR = 1e-5, 5e-5
    NCTRL = 300

UAS = 1e6
VIR = plt.get_cmap("viridis")

# %% [markdown]
# ## 1. Calibrating the matched-filter cut
#
# Tutorial 1 ended on a subtlety: the single-sinusoid significance is a *reduced*
# χ² (Papers I/II cut: Δχ² > 5), while the matched filter's is a *raw*
# four-degree-of-freedom χ² on a different scale. A raw χ² easily exceeds 5 on pure
# noise, so we cannot reuse that number. Instead we **calibrate**: simulate
# moon-free campaigns, run the identical search with both statistics, measure the
# fraction of pure-noise trials where the sinusoid's Δχ² > 5 fires (its
# false-alarm rate), and choose the matched-filter threshold that fires at the same
# rate. The two statistics then have identical false-alarm rates by construction,
# and any difference in what they detect is real sensitivity.
#
# The threshold depends on the search configuration (how many trial periods you
# look at — the "look-elsewhere" effect) but not on precision, so calibrate once per
# search setup and reuse it.

# %%
def calibrate_cut(search, nctrl, sine_cut=5.0, seed0=0):
    """Matched-filter threshold matching the single-sinusoid Dchi^2 > sine_cut false-alarm rate."""
    noise = SimParams(moon_a=10.0, moon_mass=0.0, astrometric_precision=PREC_FOUR, **search)  # mass=0: no moon
    sine = np.array([run_trial(noise, seed=seed0 + k, stat="sine")["sig"] for k in range(nctrl)])
    ell  = np.array([run_trial(noise, seed=seed0 + 10_000 + k, stat="ellipse")["sig"] for k in range(nctrl)])
    far = float(np.mean(sine > sine_cut))
    far = min(max(far, 1.0 / nctrl), 0.5)
    return float(np.quantile(ell, 1.0 - far)), far

THR, FAR = calibrate_cut(SEARCH, NCTRL)
print("single-sinusoid false-alarm rate at dchi>5 : %.3f" % FAR)
print("matched-filter cut with the same rate      : dchi_mf > %.1f" % THR)

# %% [markdown]
# (With `FAST=False`, 300 control trials give the paper's value, ≈28.9 for this
# search configuration.)
#
# ## 2. A two-moon system, round by round
#
# Inject a 0.3 M⊕ moon at 20 R_Jup and a 0.2 M⊕ moon at 12 R_Jup (the Paper III
# fiducial pair). `run_trial` with the matched filter and our calibrated cut does the
# whole prewhitening loop: search → claim the peak if it clears the cut → subtract
# its fitted ellipse model → repeat → stop when nothing clears. The diagnostics keep
# every round's periodogram so we can watch it happen.

# %%
p2 = SimParams(astrometric_precision=PREC_TWO,
               moons=[Moon(a=20.0, mass=0.30, inc=50.0, ecc=0.05),
                      Moon(a=12.0, mass=0.20, inc=50.0, ecc=0.05)], **SEARCH)
d2 = run_trial(p2, seed=42, return_diagnostics=True, recover_thr=THR, stat="ellipse")["diag"]

BLANK = 0.25    # run_trial excludes +-25% in period around each claimed moon in later rounds

def out_of_window_peak(d):
    """Largest peak of the terminating round OUTSIDE the excluded windows around claimed periods."""
    last = [x for x in d["recoveries"] if not x["recovered"]][0]
    pg, sg = np.asarray(last["period_grid"]), np.asarray(last["periodogram"], float)
    keep = np.ones(pg.size, bool)
    for x in d["recoveries"]:
        if x["recovered"]:
            keep &= np.abs(pg / x["best_period"] - 1.0) >= BLANK
    return float(np.nanmax(sg[keep]))

rounds = d2["recoveries"]                       # one entry per search round; last one terminates
print("rounds run: %d" % len(rounds))
for k, rd in enumerate(rounds, 1):
    if rd["recovered"]:
        print("  round %d: claimed P = %6.2f d   dchi_mf = %7.1f   mass = %.3f Mearth   i = %.0f deg"
              % (k, rd["best_period"], rd["sig"], rd["mass"], rd["inclination"]))
    else:
        print("  round %d: leftover at the claimed periods = %.0f (excluded); largest peak OUTSIDE "
              "those windows = %.1f < cut %.1f  ->  stop" % (k, rd["peak_sig"], out_of_window_peak(d2), THR))

# %%
truth = [d2["true_synodic"]] + list(d2["companion_synodics"])
fig, ax = plt.subplots(len(rounds), 1, figsize=(9, 2.6 * len(rounds)), sharex=True)
for k, (a, rd) in enumerate(zip(np.atleast_1d(ax), rounds), 1):
    a.semilogx(rd["period_grid"], rd["periodogram"], color=VIR(0.25), lw=0.9)
    for t in truth: a.axvline(t, color="0.6", ls="--", lw=0.8)
    a.axhline(THR, color=VIR(0.05), ls=":", lw=1.0)
    tag = ("claims P=%.1f d" % rd["best_period"]) if rd["recovered"] else "nothing clears the cut -> stop"
    a.set_title("Round %d: %s" % (k, tag), fontsize=9.5); a.set_ylabel(r"$\Delta\chi^2_{\rm mf}$")
np.atleast_1d(ax)[-1].set_xlabel("Trial synodic period (days)")
plt.tight_layout(); plt.show()

# %% [markdown]
# Dashed lines are the true synodic periods; the dotted line is the calibrated cut.
# Round 1 claims the stronger moon, and subtracting its model — sidebands included
# — cleans the residual enough to reveal the second in round 2. Round 3 finds
# nothing above the cut and stops. The recovered masses and inclinations come from
# the same ellipse fit that did the detecting (Tutorial 3 explores that).
#
# ## 3. The four-moon system: sinusoid vs. matched filter
#
# Now the paper's stress test: four low-mass moons (0.05–0.1 M⊕) at 8–30 R_Jup on
# offset orbital phases. At the paper's 50 µas hourly precision the single sinusoid
# recovers only the highest-reflex moon and, to find the rest, has to relax its cut
# and accept false positives; the matched filter recovers **all four** at the
# calibrated cut (Paper III Fig. 5, Table 3) — though the faintest, innermost moon
# is a marginal detection (Δχ²_mf ≈ 29 against a cut of 28.9). We run both
# statistics on identical data. In `FAST` mode (6-hr cadence) expect the sinusoid to
# find one moon and the matched filter three, with the faint fourth falling below
# the cut — the coarser cadence costs exactly the moon that was marginal to begin
# with. Set `FAST = False` to see it clear.

# %%
MOONS = [(20, 0.30 / 3), (12, 0.20 / 3), (30, 0.15 / 3), (8, 0.25 / 3)]   # (a [R_Jup], mass [Mearth])
PHASES = (0.15, 0.55, 0.80, 0.35)

def four_moon_params(prec):
    p = SimParams(astrometric_precision=prec,
                  moons=[Moon(a=a, mass=m, inc=50, ecc=0.05) for a, m in MOONS], **SEARCH)
    for mn, fr in zip(p.moons, PHASES):                 # offset orbital phases
        mn.t0 = fr * p._period_days(mn) / 365.25
    return p

def is_real(d, rd):
    return any(abs(rd["best_period"] / t - 1) < 0.05
               for t in [d["true_synodic"]] + list(d["companion_synodics"]))

p4 = four_moon_params(PREC_FOUR)
d_sine = run_trial(p4, seed=7, return_diagnostics=True, recover_thr=5.0, stat="sine")["diag"]
d_mf   = run_trial(p4, seed=7, return_diagnostics=True, recover_thr=THR, stat="ellipse")["diag"]

for lab, d, cut in [("single sinusoid (dchi>5)", d_sine, 5.0), ("matched filter (dchi_mf>%.1f)" % THR, d_mf, THR)]:
    rec = [x for x in d["recoveries"] if x["recovered"]]
    real = [x for x in rec if is_real(d, x)]
    print("%-32s recovered %d/4 real moons, %d spurious" % (lab, len(real), len(rec) - len(real)))
    for x in sorted(rec, key=lambda x: -x["sig"]):
        print("     a=%4.1f R_J  P=%6.2f d  sig=%7.1f  mass=%.3f  i=%.0f  %s"
              % (x["a_rjup"], x["best_period"], x["sig"], x["mass"], x["inclination"],
                 "" if is_real(d, x) else "<- SPURIOUS"))

# %% [markdown]
# If a moon is missing from the matched-filter list, where did it go? Look at the
# terminating round's periodogram at the innermost moon's true period — the
# shortest synodic period, ~5 d — and compare its peak with the cut:

# %%
last = [x for x in d_mf["recoveries"] if not x["recovered"]][0]
pg, sg = np.asarray(last["period_grid"]), np.asarray(last["periodogram"], float)
P_inner = min([d_mf["true_synodic"]] + list(d_mf["companion_synodics"]))    # shortest period = a=8 R_Jup
near = np.abs(pg / P_inner - 1.0) < 0.05
print("innermost moon (P_syn = %.2f d): matched-filter peak = %.1f  vs cut %.1f  ->  %s"
      % (P_inner, np.nanmax(sg[near]), THR,
         "detected" if np.nanmax(sg[near]) > THR else "below the cut at this cadence (marginal at the paper's hourly cadence)"))

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, (lab, d) in zip(ax, [("Single sinusoid", d_sine), ("Matched filter", d_mf)]):
    ia, im = zip(*d["input_moons"])
    a.scatter(ia, im, s=110, facecolors="none", edgecolors=VIR(0.55), lw=1.6, label="input", zorder=5)
    rec = [x for x in d["recoveries"] if x["recovered"]]
    for x in rec:
        col = VIR(0.85) if is_real(d, x) else "crimson"
        a.errorbar(x["a_rjup"], x["mass"], yerr=x.get("mass_err", 0), fmt="o", ms=6, color=col, capsize=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"Moon semimajor axis ($R_{\rm Jup}$)")
    a.set_xticks([8, 10, 12, 20, 30]); a.set_xticklabels(["8", "10", "12", "20", "30"])
    a.set_yticks([0.05, 0.07, 0.1]); a.set_yticklabels(["0.05", "0.07", "0.1"])
    a.xaxis.set_minor_formatter(plt.NullFormatter()); a.yaxis.set_minor_formatter(plt.NullFormatter())
    n = sum(is_real(d, x) for x in rec)
    a.set_title("%s: %d/4 recovered" % (lab, n), fontsize=10)
ax[0].set_ylabel(r"Moon mass ($M_\oplus$)"); ax[0].legend(frameon=False, fontsize=8, loc="lower left")
plt.tight_layout(); plt.show()

# %% [markdown]
# Open circles are the injected moons, filled points the recovered ones (a red point
# would be a spurious claim). The matched filter captures each moon's full sideband
# power, so faint moons that sit below the sinusoid's cut clear the calibrated
# matched-filter cut instead — and with no relaxed threshold there are no false
# positives to trade off. The innermost moon is the marginal case: right at the cut
# in the paper's full-cadence run, below it at this coarser cadence. That is exactly
# the regime where a detection claim needs the false-positive analysis of Tutorial 4.
#
# ## Take-aways
#
# * Superposition + prewhitening lets a *single* search recover a whole satellite
#   system, in order of reflex amplitude.
# * The matched-filter cut is set empirically, to the same false-alarm rate as the
#   classic Δχ² > 5 cut — a fair comparison and a reliable detection criterion.
# * At a precision where the sinusoid finds one of four low-mass moons, the matched
#   filter finds three cleanly and the fourth marginally (all four in the paper config).
#
# **Try:** raise `PREC_FOUR` until the matched filter starts losing moons, or add a
# fifth moon to `MOONS`. **Paper config:** `FAST = False` (Figs. 3–6, Table 3).
