#!/usr/bin/env python3
"""Where inclination/mass characterization degrades for longer-period planets (Paper III).

The projected-ellipse fit (Equation~model) recovers the moon's inclination and a
degeneracy-free mass because the star-planet position angle phi(t) sweeps through a range of
angles over the campaign, letting a 1-D (separation) time series constrain the 2-D projected
reflex. For a longer-period planet the campaign covers only a small arc of the planetary orbit,
phi barely moves, and that leverage is lost. Note the moon signal is still *detected* (its
significance even rises as the planet-motion sidebands pull toward the carrier) -- it is the
*characterization* that fails.

We fix the moon (a=20 R_Jup, m=1 Mearth, i=50 deg, circular; 10 uas, 5-yr hourly campaign) and
vary the planet semimajor axis a_pl -> planet period P_pl, i.e. the fraction of the planetary
orbit the campaign covers (T_camp / P_pl). At each a_pl we average the blind recovery over moon
orbital phase and noise, recording the recovered mass and inclination and the actual phi sweep.

Compute-light (circular moon -> fast single-round recovery); parallel, one a_pl cached at a time.
Writes figs/planet_period_bias.png. Run:
    python make_planetperiod_fig.py            # compute (cached) then plot
    python make_planetperiod_fig.py --plot     # replot from cache
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
CACHE = os.path.join(HERE, "data", "_planetperiod_cache.pkl")
OUT = os.path.join(FIGDIR, "planet_period_bias.png")

# ------------------------------- configuration ----------------------------- #
APL = [1.8, 2.5, 3.5, 5.0, 7.0, 10.0, 15.0]              # planet semimajor axis (au) -> P_pl = a^1.5 yr
A, MASS, INC = 20.0, 1.0, 50.0                           # fixed moon (circular)
PREC = 1e-5                                              # 10 uas
T_CAMP = 5.0                                             # campaign length (yr)
PLANET_PHASES = [0.0, 0.25, 0.5, 0.75]                   # which arc of the planet orbit the campaign
                                                         #   catches (fraction of P_pl) -- key averaging
PHASES = [0.0, 0.5]                                      # moon orbital phase (fraction of period)
SEEDS = [1, 2]                                           # noise realizations
SEARCH = dict(pend=60.0, ptestwidth=0.006)
P_MOON_YR = 20.0 / 365.25                                # ~moon period, for phasing


def _one(task):
    apl, plphf, phf, seed = task
    p = SimParams(planet_a=apl, planet_t0=plphf * apl ** 1.5, moon_a=A, moon_mass=MASS,
                  moon_inc=INC, moon_ecc=0.0, moon_t0=phf * P_MOON_YR,
                  astrometric_precision=PREC, **SEARCH)
    d = run_trial(p, seed=int(seed), return_diagnostics=True)["diag"]
    ph = np.unwrap(np.arctan2(d["y_fit"], d["x_fit"]))
    sweep = float(np.degrees(ph.max() - ph.min()))
    rec = [x for x in d["recoveries"] if x["recovered"]]
    if not rec:
        return (np.nan, np.nan, sweep)
    r = rec[0]
    return (r["mass"], r["inclination"], sweep)


def compute_apl(apl, workers):
    tasks = [(apl, plphf, phf, s) for plphf in PLANET_PHASES for phf in PHASES for s in SEEDS]
    with Pool(workers) as pool:
        res = np.array(pool.map(_one, tasks), float)
    m, i, sw = res[:, 0], res[:, 1], res[:, 2]
    return dict(apl=apl, Ppl=apl ** 1.5, frac=T_CAMP / apl ** 1.5,
                sweep=float(np.nanmean(sw)),
                m_mean=float(np.nanmean(m)), m_std=float(np.nanstd(m)),
                i_mean=float(np.nanmean(i)), i_std=float(np.nanstd(i)))


def plot(cache):
    ks = sorted(cache, key=lambda a: cache[a]["frac"])
    frac = np.array([cache[a]["frac"] for a in ks])
    mm = np.array([cache[a]["m_mean"] for a in ks]); ms = np.array([cache[a]["m_std"] for a in ks])
    ii = np.array([cache[a]["i_mean"] for a in ks]); is_ = np.array([cache[a]["i_std"] for a in ks])
    VIR = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for a in (ax if hasattr(ax, "__len__") else [ax]):
        a.axvspan(frac.min() * 0.8, 0.5, color="0.9", zorder=0)   # shade the small-arc regime

    ax[0].axhline(INC, color="0.5", ls=":", lw=1.0)
    ax[0].errorbar(frac, ii, yerr=is_, fmt="o-", color=VIR(0.6), ms=6, lw=1.4, capsize=3, zorder=3)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"Fraction of Planet Orbit Covered ($T_{\rm camp}/P_{\rm pl}$)")
    ax[0].set_ylabel(r"Recovered Inclination (deg)")
    ax[0].set_title(r"Inclination Recovery (input $50^\circ$)", fontsize=9.5)

    ax[1].axhline(0, color="0.5", ls=":", lw=1.0)
    ax[1].errorbar(frac, (mm / MASS - 1) * 100, yerr=ms / MASS * 100, fmt="o-",
                   color=VIR(0.28), ms=6, lw=1.4, capsize=3, zorder=3)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"Fraction of Planet Orbit Covered ($T_{\rm camp}/P_{\rm pl}$)")
    ax[1].set_ylabel("Recovered-Mass Bias (%)")
    ax[1].set_title("Mass Recovery", fontsize=9.5)
    for a in ax:
        a.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    import time
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    if args.plot:
        plot(cache)
    else:
        t0 = time.time()
        for apl in APL:
            if apl in cache:
                continue
            cache[apl] = compute_apl(apl, args.workers)
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            pickle.dump(cache, open(CACHE, "wb"))
            r = cache[apl]
            print("  a_pl=%4.1f au  frac=%.2f  sweep=%3.0f deg  m=%.3f+/-%.3f  i=%.1f+/-%.1f"
                  % (apl, r["frac"], r["sweep"], r["m_mean"], r["m_std"], r["i_mean"], r["i_std"]),
                  flush=True)
            if time.time() - t0 > 35:
                break
        if all(a in cache for a in APL):
            print("all done; plotting"); plot(cache)
        else:
            print("partial (%d/%d) -- re-run to continue" % (len(cache), len(APL)))
