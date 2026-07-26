#!/usr/bin/env python3
"""Four-moon demonstration figures (Paper III) + the four-moon recovery table.

A four-moon system of low-mass satellites (0.10, 0.067, 0.050, 0.083 Mearth at
20, 12, 30, 8 Rjup; all i=50 deg, offset orbital phases) around the alpha Cen A
giant-planet candidate, over a five-year campaign. We run it at two per-epoch
precisions to show that the number of recoverable moons is set by precision:

  20 uas -> all four moons recovered
  50 uas -> only the most massive (largest reflex) moon clears the threshold

Produces:
  four_moon_summary.png    2x2: initial periodogram + recovered mass-a plane, at
                           20 uas (top) and 50 uas (bottom)
  four_moon_phasefold.png  each moon's phase-fold (top row) and residual (bottom
                           row), one column per moon, for the 20 uas run
and prints the 20 uas recovery table (Table 3) and the 50 uas recovery count.

The synodic-period search is restricted to <60 d (all four moons lie below 40 d)
to keep the run fast; cadence, baseline, and host match the fiducial system.

Reproducible: fixed seed, imports the repo's exomoonsim package. Run:
    python make_four_moon_figs.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))       # repo root -> exomoonsim package
from exomoonsim.sim import SimParams, Moon, run_trial

FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

VIR = plt.get_cmap("viridis")
C_OBS, C_TRUE, C_FIT, C_COM = VIR(0.28), VIR(0.55), VIR(0.82), VIR(0.05)
UAS = 1e6
SEED = 7
MOONS = [(20, 0.30 / 3), (12, 0.20 / 3), (30, 0.15 / 3), (8, 0.25 / 3)]  # a (Rjup), m (Mearth)
FRACS = (0.15, 0.55, 0.80, 0.35)


def run(prec, thr=5.0):
    moons = [Moon(a=a, mass=m, inc=50, ecc=0.05) for a, m in MOONS]
    p = SimParams(astrometric_precision=prec, texp=1.0, pend=60.0, ptestwidth=0.0015, moons=moons)
    for mn, fr in zip(p.moons, FRACS):
        mn.t0 = fr * p._period_days(mn) / 365.25
    return run_trial(p, seed=SEED, return_diagnostics=True, recover_thr=thr)["diag"]


def is_real(d, r):   # a recovered signal is real if its period matches a true synodic
    tsyns = [d["true_synodic"]] + list(d["companion_synodics"])
    return any(abs(r["best_period"] / ts - 1) < 0.05 for ts in tsyns)


import pickle
CACHE = os.path.join(HERE, "data", "_fourmoon_cache.pkl")   # speeds re-runs; delete to recompute
_cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
def cached(key, prec, thr=5.0):
    if key not in _cache:
        _cache[key] = run(prec, thr)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pickle.dump(_cache, open(CACHE, "wb"))
    return _cache[key]
d20 = cached("20", 2e-5)
d50 = cached("50", 5e-5)
d50_t3 = cached("50t3", 5e-5, 3.0)   # 50 uas with a relaxed Delta chi^2 > 3 cut


def periodogram_panel(a, d, first=True):
    r0 = d["recoveries"][0]
    a.semilogx(r0["period_grid"], r0["periodogram"], color=C_OBS, lw=0.9)
    for mk in [d["true_synodic"]] + list(d["companion_synodics"]):
        a.axvline(mk, color="0.6", ls="--", lw=0.7)
    a.axhline(d["recover_thr"], color=C_COM, ls=":", lw=0.9)
    a.set_ylabel(r"$\chi^2_{\rm flat}-\chi^2_{\rm sine}$")


def system_panel(a, d, legend=False, mark_spurious=False):
    inp = list(d["input_moons"]); done = [x for x in d["recoveries"] if x["recovered"]]
    ia, im = zip(*inp)
    a.scatter(ia, im, s=95, facecolors="none", edgecolors=C_TRUE, lw=1.6, zorder=5)
    nreal = nsp = 0
    for r in done:
        if mark_spurious and not is_real(d, r):
            nsp += 1
            a.plot(r["a_rjup"], r["mass"], "x", color="crimson", ms=9, mew=2, zorder=6)
        else:
            nreal += 1
            a.errorbar(r["a_rjup"], r["mass"], yerr=r.get("mass_err", 0.0), fmt="o", ms=6,
                       color=C_FIT, ecolor=C_FIT, capsize=3, lw=1.2, zorder=4)
            a.annotate(r"$i=%.0f^\circ$" % r["inclination"], (r["a_rjup"], r["mass"]),
                       textcoords="offset points", xytext=(7, 4), fontsize=7.5, color=C_FIT)
    a.set_xscale("log"); a.set_yscale("log")
    allx = [v[0] for v in inp] + [r["a_rjup"] for r in done]
    ally = [v[1] for v in inp] + [r["mass"] for r in done]
    xt = [c for c in (4, 5, 8, 10, 15, 20, 30) if min(allx) * 0.7 <= c <= max(allx) * 1.4]
    yt = [c for c in (0.04, 0.05, 0.07, 0.1, 0.15, 0.2) if min(ally) * 0.6 <= c <= max(ally) * 1.6]
    a.set_xticks(xt); a.set_xticklabels(["%g" % c for c in xt])
    a.set_yticks(yt); a.set_yticklabels(["%g" % c for c in yt])
    a.xaxis.set_minor_formatter(ticker.NullFormatter())
    a.yaxis.set_minor_formatter(ticker.NullFormatter())
    a.set_xlim(min(allx) * 0.72, max(allx) * 1.32); a.set_ylim(min(ally) * 0.6, max(ally) * 1.5)
    a.set_ylabel(r"Moon Mass ($M_\oplus$)")
    handles = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
                      markeredgecolor=C_TRUE, markersize=8, label="Input"),
               Line2D([0], [0], marker="o", ls="none", color=C_FIT, markersize=6, label="Recovered")]
    if mark_spurious:
        handles.append(Line2D([0], [0], marker="x", ls="none", color="crimson",
                              markersize=8, mew=2, label="False Positive"))
    if legend or mark_spurious:
        a.legend(handles=handles, fontsize=7.5, loc="lower left")
    return nreal, nsp


# ---------------- Figure A: 3x2 precision + threshold comparison ----------- #
L = "abcdef"
fig, ax = plt.subplots(3, 2, figsize=(11.2, 10.2))
for row, (d, lab, mark) in enumerate([(d20, "20", False), (d50, "50", False), (d50_t3, "50", True)]):
    periodogram_panel(ax[row, 0], d)                     # threshold line drawn at d["recover_thr"]
    nreal, nsp = system_panel(ax[row, 1], d, legend=(row == 0), mark_spurious=mark)
    ntot = len(d["input_moons"]); thr = int(round(d["recover_thr"]))
    ax[row, 0].set_title(r"(%s) $%s\,\mu$as, $\Delta\chi^2>%d$: Initial Periodogram"
                         % (L[2 * row], lab, thr), fontsize=9.5)
    if mark:
        ax[row, 1].set_title(r"(%s) $%s\,\mu$as, $\Delta\chi^2>%d$: %d/%d moons $+$ %d false positive%s"
                             % (L[2 * row + 1], lab, thr, nreal, ntot, nsp, "" if nsp == 1 else "s"),
                             fontsize=9.5)
        for x in [x for x in d["recoveries"] if x["recovered"] and not is_real(d, x)]:
            ax[row, 0].axvline(x["best_period"], color="crimson", lw=1.0, alpha=0.9)
    else:
        ax[row, 1].set_title(r"(%s) $%s\,\mu$as, $\Delta\chi^2>%d$: Recovered (%d/%d)"
                             % (L[2 * row + 1], lab, thr, nreal, ntot), fontsize=9.5)
for c in (0, 1):
    ax[2, c].set_xlabel(["Trial Synodic Period (days)", r"Moon Semimajor Axis ($R_{\rm Jup}$)"][c])
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "four_moon_summary.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure B: 20 uas phase-folds (by significance) ----------- #
done = [x for x in d20["recoveries"] if x["recovered"]]
order = sorted(range(len(done)), key=lambda k: -done[k]["sig"])
fig, ax = plt.subplots(2, 4, figsize=(14.0, 3.9))
for col, k in enumerate(order):
    r = done[k]
    ph, am = r["fold_phase"], r["fold_amp"] * UAS
    er, mo = r["fold_err"] * UAS, r["fold_model"] * UAS
    o = np.argsort(ph)
    at, ab = ax[0, col], ax[1, col]
    at.errorbar(ph[o], am[o], yerr=er[o], fmt="o", ms=2.2, color=C_OBS, alpha=0.7, lw=0.5, label="Binned")
    at.plot(ph[o], mo[o], "-", color=C_FIT, lw=1.4, label="Sine Fit")
    at.set_title(r"$a=%.0f\,R_{\rm Jup}$,  $P_{\rm syn}=%.1f$ d,  $\Delta\chi^2=%.0f$"
                 % (r["a_rjup"], r["best_period"], r["sig"]), fontsize=8.5)
    ab.axhline(0, color="0.6", ls=":", lw=0.8)
    ab.errorbar(ph[o], am[o] - mo[o], yerr=er[o], fmt="o", ms=2.2, color=C_OBS, alpha=0.7, lw=0.5)
    ab.set_xlabel("Phase-Folded Day")
ax[0, 0].set_ylabel(r"Amplitude ($\mu$as)")
ax[1, 0].set_ylabel(r"Residual ($\mu$as)")
ax[0, 0].legend(fontsize=7, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "four_moon_phasefold.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- tables / numbers ----------------------------------------- #
inmap = {round(a): m for a, m in MOONS}
print("Table 3 -- 20 uas, four-moon recovery (by significance):")
print("  a_in  m_in   |  m_rec    i_rec   P_syn(d)   dchi")
for r in sorted(done, key=lambda r: -r["sig"]):
    a = r["a_rjup"]
    print("  %4.0f  %5.3f  |  %6.4f  %5.1f   %7.2f   %5.0f"
          % (a, inmap.get(round(a), float("nan")), r["mass"], r["inclination"], r["best_period"], r["sig"]))
d50done = [x for x in d50["recoveries"] if x["recovered"]]
print("50 uas: recovered %d/4  (primary dchi=%.0f)"
      % (len(d50done), d50done[0]["sig"] if d50done else 0))
print("wrote 2 figures to", FIGDIR)
