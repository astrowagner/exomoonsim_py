#!/usr/bin/env python3
"""Four-moon figures with the projected-ellipse matched filter as the primary detector (Paper III).

Matched-filter version of make_four_moon_figs.py: the same four-moon system, recovered by iterative
prewhitening on the projected-ellipse statistic with the detection cut CALIBRATED to the
single-sinusoid Delta chi^2 > 5 false-alarm rate (thr ~ 28.95 for this search config, from
paper/ellipse_cut.py).  Unlike the single sinusoid -- which recovers all four moons at 20 uas but
only the highest-reflex one at 50 uas -- the matched filter recovers ALL FOUR at both precisions,
so the figure is a clean 2-row (20 and 50 uas) comparison with no relaxed-cut row.

Writes figs/four_moon_summary.png (2x2) and figs/four_moon_phasefold.png, and prints the recovery
tables.  Recovery is cached (data/_fourmoon_ellipse_cache.pkl) so replotting is cheap.

    python make_four_moon_figs_ellipse.py            # compute (cached) + plot
    python make_four_moon_figs_ellipse.py --thr 28.95   # override the calibrated cut
"""
import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from exomoonsim.sim import SimParams, Moon, run_trial

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
CACHE = os.path.join(HERE, "data", "_fourmoon_ellipse_cache.pkl")
VIR = plt.get_cmap("viridis")
C_OBS, C_TRUE, C_FIT, C_COM = VIR(0.28), VIR(0.55), VIR(0.82), VIR(0.05)
UAS = 1e6
SEED = 7
MOONS = [(20, 0.30 / 3), (12, 0.20 / 3), (30, 0.15 / 3), (8, 0.25 / 3)]
FRACS = (0.15, 0.55, 0.80, 0.35)
SEARCH = dict(texp=1.0, pend=60.0, ptestwidth=0.0015)
THR_DEFAULT = 28.95                                       # ellipse cut (matched FAR), this search config


def run(prec, thr):
    moons = [Moon(a=a, mass=m, inc=50, ecc=0.05) for a, m in MOONS]
    p = SimParams(astrometric_precision=prec, moons=moons, **SEARCH)
    for mn, fr in zip(p.moons, FRACS):
        mn.t0 = fr * p._period_days(mn) / 365.25
    return run_trial(p, seed=SEED, return_diagnostics=True, recover_thr=thr, stat="ellipse")["diag"]


def is_real(d, r):
    tsyns = [d["true_synodic"]] + list(d["companion_synodics"])
    return any(abs(r["best_period"] / ts - 1) < 0.05 for ts in tsyns)


def periodogram_panel(a, d):
    r0 = d["recoveries"][0]
    a.semilogx(r0["period_grid"], r0["periodogram"], color=C_OBS, lw=0.9)
    for mk in [d["true_synodic"]] + list(d["companion_synodics"]):
        a.axvline(mk, color="0.6", ls="--", lw=0.7)
    a.axhline(d["recover_thr"], color=C_COM, ls=":", lw=0.9)
    a.set_ylabel(r"Matched-Filter $\Delta\chi^2_{\rm ell}$")


def system_panel(a, d, legend=False):
    inp = list(d["input_moons"]); done = [x for x in d["recoveries"] if x["recovered"]]
    ia, im = zip(*inp)
    a.scatter(ia, im, s=95, facecolors="none", edgecolors=C_TRUE, lw=1.6, zorder=5)
    nreal = 0
    for r in done:
        if is_real(d, r):
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
    if legend:
        handles = [Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
                          markeredgecolor=C_TRUE, markersize=8, label="Input"),
                   Line2D([0], [0], marker="o", ls="none", color=C_FIT, markersize=6, label="Recovered")]
        a.legend(handles=handles, fontsize=7.5, loc="lower left")
    return nreal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=THR_DEFAULT)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    cache = {} if args.recompute or not os.path.exists(CACHE) else pickle.load(open(CACHE, "rb"))
    if "20" not in cache or "50" not in cache:
        cache["20"] = run(2e-5, args.thr)
        cache["50"] = run(5e-5, args.thr)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        pickle.dump(cache, open(CACHE, "wb"))
    d20, d50 = cache["20"], cache["50"]

    # Figure A: 2x2, 20 and 50 uas, both 4/4
    L = "abcd"
    fig, ax = plt.subplots(2, 2, figsize=(11.2, 7.0))
    for row, (d, lab) in enumerate([(d20, "20"), (d50, "50")]):
        periodogram_panel(ax[row, 0], d)
        nreal = system_panel(ax[row, 1], d, legend=(row == 0))
        ntot = len(d["input_moons"]); thr = int(round(d["recover_thr"]))
        ax[row, 0].set_title(r"(%s) $%s\,\mu$as: Matched-Filter Periodogram" % (L[2 * row], lab),
                             fontsize=9.5)
        ax[row, 1].set_title(r"(%s) $%s\,\mu$as: Recovered (%d/%d)" % (L[2 * row + 1], lab, nreal, ntot),
                             fontsize=9.5)
    for c in (0, 1):
        ax[1, c].set_xlabel(["Trial Synodic Period (days)", r"Moon Semimajor Axis ($R_{\rm Jup}$)"][c])
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "four_moon_summary.png"),
                                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure B: 50 uas phase-folds (the harder case; all four now detected), by significance
    done = [x for x in d50["recoveries"] if x["recovered"]]
    order = sorted(range(len(done)), key=lambda k: -done[k]["sig"])
    fig, ax = plt.subplots(2, 4, figsize=(14.0, 3.9))
    for col, k in enumerate(order):
        r = done[k]
        ph, am = r["fold_phase"], r["fold_amp"] * UAS
        er, mo = r["fold_err"] * UAS, r["fold_model"] * UAS
        o = np.argsort(ph)
        at, ab = ax[0, col], ax[1, col]
        at.errorbar(ph[o], am[o], yerr=er[o], fmt="o", ms=2.2, color=C_OBS, alpha=0.7, lw=0.5, label="Binned")
        at.plot(ph[o], mo[o], "-", color=C_FIT, lw=1.4, label="Model Fit")
        at.set_title(r"$a=%.0f\,R_{\rm Jup}$,  $P_{\rm syn}=%.1f$ d,  $\Delta\chi^2_{\rm ell}=%.0f$"
                     % (r["a_rjup"], r["best_period"], r["sig"]), fontsize=8.5)
        ab.axhline(0, color="0.6", ls=":", lw=0.8)
        ab.errorbar(ph[o], am[o] - mo[o], yerr=er[o], fmt="o", ms=2.2, color=C_OBS, alpha=0.7, lw=0.5)
        ab.set_xlabel("Phase-Folded Day")
    ax[0, 0].set_ylabel(r"Amplitude ($\mu$as)"); ax[1, 0].set_ylabel(r"Residual ($\mu$as)")
    ax[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "four_moon_phasefold.png"),
                                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    inmap = {round(a): m for a, m in MOONS}
    for d, lab in [(d20, "20"), (d50, "50")]:
        dn = sorted([x for x in d["recoveries"] if x["recovered"]], key=lambda r: -r["sig"])
        print("=== %s uas: %d/4 recovered ===" % (lab, len(dn)))
        print("  a_in  m_in   |  m_rec    i_rec   P_syn(d)   dchi_ell")
        for r in dn:
            a = r["a_rjup"]
            print("  %4.0f  %5.3f  |  %6.4f  %5.1f   %7.2f   %6.0f"
                  % (a, inmap.get(round(a), float("nan")), r["mass"], r["inclination"],
                     r["best_period"], r["sig"]))
    print("wrote 2 figures to", FIGDIR)
