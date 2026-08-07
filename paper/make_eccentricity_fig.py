#!/usr/bin/env python3
"""Bias of the circular ellipse fit for eccentric moon orbits (Paper III, limitation test).

The projected-ellipse model of Equation~(model) assumes a *circular* moon reflex (a single
sidereal frequency). A moon on an eccentric orbit adds harmonics of that frequency, which the
circular model cannot absorb, biasing the recovered mass and inclination and leaving harmonic
residual power. Here we quantify that bias.

For a single high-S/N moon (a=20 R_Jup, m=1 Mearth, i=50 deg, 10 uas, 5-yr hourly campaign) we
sweep the orbital eccentricity e and, at each e, run the full blind recovery over a grid of
argument-of-periapsis and orbital phase (and a few noise seeds), recording the recovered mass,
inclination, and synodic period. Averaging over orientation/phase/noise isolates the systematic
e-dependence from the (few-percent) phase scatter that is present even at e=0.

Compute-moderate; parallel across the per-e configurations, one eccentricity cached at a time
(data/_ecc_cache.pkl) so it resumes. Writes figs/eccentricity_bias.png. Run:
    python make_eccentricity_fig.py            # compute (cached) then plot
    python make_eccentricity_fig.py --plot     # replot from cache
"""
import os
import sys
import pickle
import argparse
import numpy as np
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams, run_trial

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
CACHE = os.path.join(HERE, "data", "_ecc_cache.pkl")
OUT = os.path.join(FIGDIR, "eccentricity_bias.png")

# ------------------------------- configuration ----------------------------- #
ECCS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]                    # moon orbital eccentricity
A, MASS, INC = 20.0, 1.0, 50.0                           # fixed moon: sep (R_Jup), mass, inclination
PREC = 1e-5                                              # 10 uas
OMEGAS = [0.0, 90.0, 180.0, 270.0]                       # argument of periapsis (deg)
PHASES = [0.0, 0.5]                                      # orbital phase, fraction of the period
SEEDS = [1, 2]                                           # noise realizations
SEARCH = dict(pend=60.0, ptestwidth=0.006)              # coarse period grid: fast, still finds ~20 d peak
P_YR = None                                              # moon period (yr), filled in below


def _one(task):
    """Recover one (eccentricity, omega, phase, seed) config; return (m_rec, i_rec, P_rec)."""
    e, om, phf, seed, per_yr = task
    p = SimParams(moon_a=A, moon_mass=MASS, moon_inc=INC, moon_ecc=e, moon_omega=om,
                  moon_t0=phf * per_yr, astrometric_precision=PREC, **SEARCH)
    rec = [x for x in run_trial(p, seed=seed, return_diagnostics=True)["diag"]["recoveries"]
           if x["recovered"]]
    if not rec:
        return (np.nan, np.nan, np.nan)
    r = rec[0]
    return (r["mass"], r["inclination"], r["best_period"])


def compute_ecc(e, per_yr, workers):
    tasks = [(e, om, phf, s, per_yr) for om in OMEGAS for phf in PHASES for s in SEEDS]
    with Pool(workers) as pool:
        res = np.array(pool.map(_one, tasks), float)     # (N, 3): mass, incl, period
    m, i, P = res[:, 0], res[:, 1], res[:, 2]
    ok = np.isfinite(m)
    return dict(e=e, n=int(ok.sum()),
                m_mean=float(np.nanmean(m)), m_std=float(np.nanstd(m)),
                i_mean=float(np.nanmean(i)), i_std=float(np.nanstd(i)),
                P_mean=float(np.nanmean(P)), P_std=float(np.nanstd(P)))


def plot(cache):
    es = sorted(cache)
    m = np.array([cache[e]["m_mean"] for e in es]); ms = np.array([cache[e]["m_std"] for e in es])
    ii = np.array([cache[e]["i_mean"] for e in es]); is_ = np.array([cache[e]["i_std"] for e in es])
    VIR = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
    ax[0].axhline(0, color="0.6", ls=":", lw=1.0)
    ax[0].errorbar(es, (m / MASS - 1.0) * 100, yerr=ms / MASS * 100, fmt="o-",
                   color=VIR(0.28), ms=6, lw=1.4, capsize=3)
    ax[0].set_xlabel("Moon Orbital Eccentricity $e$")
    ax[0].set_ylabel("Recovered-Mass Bias (%)")
    ax[0].set_title("Mass Bias of the Circular Fit", fontsize=9.5)
    ax[1].axhline(INC, color="0.6", ls=":", lw=1.0)
    ax[1].errorbar(es, ii, yerr=is_, fmt="o-", color=VIR(0.6), ms=6, lw=1.4, capsize=3)
    ax[1].set_xlabel("Moon Orbital Eccentricity $e$")
    ax[1].set_ylabel(r"Recovered Inclination (deg)")
    ax[1].set_title(r"Inclination Bias (input $50^\circ$)", fontsize=9.5)
    for a in ax:
        a.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    per_yr = 20.0 / 365.25            # ~20 d synodic at a=20 R_Jup; only used to set the phase grid
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}

    if args.plot:
        plot(cache)
    else:
        todo = [e for e in ECCS if e not in cache]
        if not todo:
            print("all eccentricities cached; plotting")
            plot(cache)
        else:
            e = todo[0]                                  # one eccentricity per run (fits sandbox)
            print("computing e=%.2f (%d configs) ..." % (e, len(OMEGAS) * len(PHASES) * len(SEEDS)),
                  flush=True)
            cache[e] = compute_ecc(e, per_yr, args.workers)
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            pickle.dump(cache, open(CACHE, "wb"))
            r = cache[e]
            print("  e=%.2f: mass=%.4f+/-%.4f  incl=%.2f+/-%.2f  (n=%d)"
                  % (e, r["m_mean"], r["m_std"], r["i_mean"], r["i_std"], r["n"]), flush=True)
            print("  %d/%d eccentricities done" % (len(cache), len(ECCS)), flush=True)
