#!/usr/bin/env python3
"""Projected-ellipse matched filter vs single-sinusoid detection statistic (Paper III).

Two panels, single moon (i = 50 deg) around the fiducial alpha Cen A planet, 1 hr cadence, 5 yr:

  (a) SENSITIVITY SCALING (near-noiseless, m = 0.1 Mearth): round-1 search-peak significance
      vs separation for both statistics (normalized to a = 10 R_Jup).  The matched filter
      (fitting Equation~model at each trial period) tracks the ideal a^2; the single sinusoid
      is flat, rolls over, and aliases onto the ~1-yr window at wide separations.

  (b) MASS SENSITIVITY (10 uas, matched false-alarm rate): minimum detectable moon mass vs
      separation.  The single sinusoid's floor flattens beyond a ~ 10 R_Jup; the matched
      filter keeps improving, reaching sub-lunar masses at wide separations.

Reads committed data (data/matched_filter_compare.csv, data/matched_filter_mmin.csv), produced
by run_trial(stat='ellipse') sweeps (see analysis/matched_filter_*.py).  Panel (a) recomputes
if its CSV is missing (COMPUTE-HEAVY).  Run:  python make_matched_filter_fig.py
"""
import os
import sys
import csv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
CMP = os.path.join(HERE, "data", "matched_filter_compare.csv")
MMIN = os.path.join(HERE, "data", "matched_filter_mmin.csv")
OUT = os.path.join(FIGDIR, "matched_filter.png")
AS = [3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 120]


def _compute_cmp():
    from multiprocessing import Pool
    from exomoonsim.sim import SimParams, run_trial

    def one(a):
        o = {}
        for stat in ("sine", "ellipse"):
            d = run_trial(SimParams(moon_a=a, moon_mass=0.1, moon_inc=50, moon_ecc=0.0,
                                    astrometric_precision=1e-7, pend=500.0),
                          seed=1, return_diagnostics=True, recover_thr=1e12, stat=stat)["diag"]
            o[stat] = float(np.nanmax(d["periodogram"]))
        return a, o["sine"], o["ellipse"]
    with Pool(min(len(AS), os.cpu_count())) as p:
        R = np.array(p.map(one, AS), float)
    with open(CMP, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a_rjup", "sine_sig", "ellipse_sig"])
        for a, s, e in R:
            w.writerow([a, round(s, 3), round(e, 3)])
    return R


def _read(path, cols):
    rows = list(csv.DictReader(open(path)))
    return np.array([[float(r[c]) for c in cols] for r in rows], float)


if __name__ == "__main__":
    cmp = _read(CMP, ["a_rjup", "sine_sig", "ellipse_sig"]) if os.path.exists(CMP) else _compute_cmp()
    a1, sine, ell = cmp.T
    i10 = int(np.argmin(np.abs(a1 - 10)))
    mm = _read(MMIN, ["a_rjup", "mmin_sine", "mmin_ellipse"])
    a2, ms, me = mm.T

    VIR = plt.get_cmap("viridis")
    C_E, C_S = VIR(0.28), VIR(0.72)
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.4))

    # (a) significance scaling
    ax[0].plot(a1, (a1 / a1[i10]) ** 2, "--", color="0.55", lw=1.2, label=r"$\propto a^{2}$ (ideal)")
    ax[0].plot(a1, ell / ell[i10], "o-", color=C_E, ms=6, lw=1.6, label="Matched filter")
    ax[0].plot(a1, sine / sine[i10], "D-", color=C_S, ms=6, lw=1.6, label="Single sinusoid")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    ax[0].set_ylabel(r"Search-Peak $\Delta\chi^2$ (rel. to $a=10$)")
    ax[0].set_title(r"(a) Sensitivity recovers $\propto a^2$", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    # (b) minimum detectable mass
    ax[1].plot(a2, ms, "D-", color=C_S, ms=6, lw=1.6, label=r"Single sinusoid ($\Delta\chi^2>5$)")
    ax[1].plot(a2, me, "o-", color=C_E, ms=6, lw=1.6, label="Matched filter (calibrated)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    ax[1].set_ylabel(r"Min. Detectable Mass ($M_\oplus$, $10\,\mu$as)")
    ax[1].set_title("(b) Detects fainter moons (same false-alarm rate)", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8.5, loc="lower left")

    for x in ax:
        x.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
