#!/usr/bin/env python3
"""Minimum detectable moon mass vs astrometric precision (scaling relation).

For a single moon at a fixed separation (a = 10 R_Jup around the fiducial
alpha Cen A giant-planet host), we find the smallest moon mass whose reflex
signal clears the detection threshold (Delta chi^2 > 5), as a function of the
per-epoch astrometric precision sigma, from 10 uas up to 1 mas (two decades).

Method: at each precision, bisect on moon mass until the median Delta chi^2 over
NTRIALS noise realizations equals the threshold. That gives m_min(sigma) with no
assumption about the functional form -- so the recovered slope is a genuine test.

Expectation: the sine-vs-flat Delta chi^2 scales as (amplitude/sigma)^2, and the
reflex amplitude is linear in moon mass, so Delta chi^2 ~ (m/sigma)^2. At a fixed
threshold that predicts m_min proportional to sigma (slope 1 on a log-log plot).

Resumable: each precision's result is checkpointed to a pickle, so the run can be
stopped and restarted. Run from the repo root:
    python analysis/mass_sensitivity_vs_precision.py
"""
import os
import sys
import pickle
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams, run_trial

# ----------------------------- configuration ------------------------------ #
A_RJUP = 10.0                 # fixed moon separation
THR = 5.0                     # detection threshold in Delta chi^2
NTRIALS = 5                   # noise realizations per (precision, mass); median taken
NITER = 6                     # bisection iterations in log-mass (~1.5% mass resolution)
PRECS_UAS = [1, 2, 5, 10, 20, 50, 100, 200, 500,
             1000, 2000, 5000, 10000]              # per-epoch precision, micro-arcsec (1 uas - 10 mas)
BASE = dict(moon_a=A_RJUP, moon_inc=50.0, moon_ecc=0.0,
            pend=10.0, ptestwidth=0.008)          # short period window -> fast trials
# (moon synodic period ~7.2 d at a=10 R_Jup; pend=10 d covers it with margin)

CK = os.path.join(HERE, "_mass_sensitivity_cache.pkl")
_cache = pickle.load(open(CK, "rb")) if os.path.exists(CK) else {}


def median_sig(prec_arcsec, mass):
    """Median Delta chi^2 over NTRIALS noise realizations at (precision, mass)."""
    s = [run_trial(SimParams(moon_mass=mass, astrometric_precision=prec_arcsec, **BASE),
                   seed=1000 + k)["sig"] for k in range(NTRIALS)]
    return float(np.median(s))


def find_mmin(prec_uas):
    """Bisect on moon mass for median Delta chi^2 == THR at this precision."""
    prec = prec_uas * 1e-6                         # uas -> arcsec
    guess = 0.027 * (prec_uas / 10.0)              # sig(1 Mearth,10uas)~6800 -> anchor
    lo, hi = guess / 8.0, guess * 8.0
    while median_sig(prec, lo) >= THR:             # ensure lo is below threshold
        lo /= 2.0
    while median_sig(prec, hi) < THR:              # ensure hi is above threshold
        hi *= 2.0
    for _ in range(NITER):
        mid = np.sqrt(lo * hi)                      # geometric midpoint (log-mass bisect)
        if median_sig(prec, mid) >= THR:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


if __name__ == "__main__":
    t0 = time.time()
    for pu in PRECS_UAS:
        if pu in _cache:
            print("cached  %6.0f uas -> m_min = %.4f Mearth" % (pu, _cache[pu]))
            continue
        m = find_mmin(pu)
        _cache[pu] = m
        pickle.dump(_cache, open(CK, "wb"))
        print("solved  %6.0f uas -> m_min = %.4f Mearth   (%.0f s elapsed)"
              % (pu, m, time.time() - t0))
        if time.time() - t0 > 15:                  # per-call budget: exit and resume
            break
    done = all(pu in _cache for pu in PRECS_UAS)
    print(("DONE " if done else "PARTIAL "),
          {pu: round(_cache[pu], 4) for pu in PRECS_UAS if pu in _cache})

    if not done:
        sys.exit(0)                                # more precisions to solve; rerun

    # ------------------------- fit + figure -------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(PRECS_UAS, float)                 # precision, micro-arcsec
    y = np.array([_cache[p] for p in PRECS_UAS])   # m_min, Earth masses
    M_PL = 0.3 * 317.8                             # planet mass in Earth masses (0.3 M_Jup)

    # Fit the linear (unsaturated) regime: moon mass well below the planet mass.
    LIN_MAX = 1000.0                               # uas; above this the reflex saturates
    mfit = x <= LIN_MAX
    (slope, intercept), cov = np.polyfit(np.log10(x[mfit]), np.log10(y[mfit]), 1, cov=True)
    serr = float(np.sqrt(cov[0, 0]))

    print("\nLinear-regime fit (sigma <= 1 mas):  m_min proportional to sigma^%.3f (+/- %.3f)"
          % (slope, serr))
    print("Fitted m_min(10 uas) = %.3f Mearth" % (10.0 ** (slope + intercept)))
    print("Theory: Delta chi^2 ~ (m/sigma)^2 at fixed threshold  =>  slope = 1")
    print("Planet mass = %.0f Mearth; at 10 mas the required moon is %.2f x M_planet"
          % (M_PL, y[-1] / M_PL))

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
    ax.plot(xs, 10.0 ** (slope * np.log10(xs) + intercept), "-", color="0.35", lw=1.8,
            label=r"Linear-regime fit: $m_{\min}\propto\sigma^{%.2f\pm%.2f}$" % (slope, serr))
    ax.plot(xs, y[0] * (xs / x[0]), "--", color="crimson", lw=1.2,
            label=r"Slope $=1$ ($m_{\min}\propto\sigma$)")
    ax.axhline(M_PL, color="0.55", ls="-.", lw=1.1)
    ax.text(1.3, M_PL * 1.12, r"planet mass ($0.3\,M_{\rm Jup}$)", fontsize=8, color="0.4")
    ax.scatter(x[mfit], y[mfit], s=70, zorder=5, color="#2b6cb0", edgecolor="k", lw=0.6,
               label=r"Simulation ($m_{\min}\ll M_{\rm pl}$)")
    ax.scatter(x[~mfit], y[~mfit], s=70, zorder=5, color="#dd8452", edgecolor="k", lw=0.6,
               marker="D", label=r"Simulation (reflex saturating)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Per-Epoch Astrometric Precision $\sigma$ ($\mu$as)")
    ax.set_ylabel(r"Minimum Detectable Moon Mass ($M_\oplus$)")
    ax.set_title(r"Mass Sensitivity vs Precision ($a=10\,R_{\rm Jup}$, $\Delta\chi^2>5$)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    fig.tight_layout()
    out = os.path.join(HERE, "mass_sensitivity_vs_precision.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)

    with open(os.path.join(HERE, "mass_sensitivity_vs_precision.csv"), "w") as fh:
        fh.write("precision_uas,m_min_Mearth\n")
        for p in PRECS_UAS:
            fh.write("%g,%.5f\n" % (p, _cache[p]))
