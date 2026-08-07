#!/usr/bin/env python3
"""Does the projected-ellipse matched filter recover the a^2 detection sensitivity?

Compares the two detection statistics now available in run_trial (via the `stat` switch)
on a single moon (m = 0.1 Mearth, i = 50 deg) around the fiducial alpha Cen A planet,
1 hr cadence, 5 yr, near-noiseless so the intrinsic scaling is clean:

    stat='sine'    : the current phase-folded single-sinusoid statistic (the survey default)
    stat='ellipse' : the 4-coefficient projected-ellipse matched filter (captures sidebands)

For each separation a we take the round-1 search-peak significance for both statistics and
plot them (normalized to a = 10 R_Jup, since the two have different absolute normalizations)
against a, with an a^2 reference.  Expectation: the matched filter tracks a^2 (it recovers the
full signal power), while the single sinusoid stays flat / rolls over.

COMPUTE-HEAVY: the ellipse statistic does a least-squares fit at every trial period, so the
ellipse search is several times slower than the sine search.  Parallel over separations --
run on a multicore machine.  Writes analysis/matched_filter_compare.png; cached, so a rerun
replots instantly.

Run:  python analysis/matched_filter_compare.py
"""
import os
import sys
import numpy as np
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams, run_trial

OUT = os.path.join(HERE, "matched_filter_compare.png")
CACHE = os.path.join(HERE, "_mf_compare.npy")
AS = [3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 120]     # R_Jup


def one(a):
    """Round-1 search-peak significance for both statistics at separation a (near-noiseless)."""
    out = {}
    for stat in ("sine", "ellipse"):
        d = run_trial(SimParams(moon_a=a, moon_mass=0.1, moon_inc=50, moon_ecc=0.0,
                                astrometric_precision=1e-7, pend=500.0),
                      seed=1, return_diagnostics=True, recover_thr=1e12, stat=stat)["diag"]
        out[stat] = float(np.nanmax(d["periodogram"]))
    return a, out["sine"], out["ellipse"]


if __name__ == "__main__":
    if os.path.exists(CACHE):
        R = np.load(CACHE)
    else:
        with Pool(min(len(AS), os.cpu_count())) as p:
            R = np.array(p.map(one, AS), float)
        np.save(CACHE, R)
    a, sine, ell = R.T
    i10 = 3                                            # a[3] = 10 R_Jup

    def slope(v):
        return np.log(v[-1] / v[0]) / np.log(a[-1] / a[0])
    print("overall log-slope a=%d->%d:  sine %.2f   ellipse %.2f"
          % (a[0], a[-1], slope(sine), slope(ell)))

    VIR = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.plot(a, (a / a[i10]) ** 2, "--", color="0.55", lw=1.2, label=r"$\propto a^{2}$ (ideal)")
    ax.plot(a, ell / ell[i10], "o-", color=VIR(0.28), ms=6, lw=1.6,
            label="Projected-ellipse matched filter")
    ax.plot(a, sine / sine[i10], "D-", color=VIR(0.75), ms=6, lw=1.6,
            label="Single-sinusoid (current default)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    ax.set_ylabel(r"Search-Peak Significance (rel. to $a=10\,R_{\rm Jup}$)")
    ax.set_title("Matched Filter Recovers the $a^2$ Sensitivity", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
