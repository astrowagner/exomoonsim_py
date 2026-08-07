#!/usr/bin/env python3
"""Minimum detectable moon mass vs separation: single sinusoid vs matched filter (for Sumin).

Recalibrates the projected-ellipse matched-filter threshold to the SAME pure-noise false-alarm
rate as the single-sinusoid Delta chi^2 > 5 cut, then, at each separation, bisects on moon mass
for the faintest moon each statistic detects (median significance over trials crosses the cut).
Shows the matched filter reaching lower masses -- the sensitivity gain in mass terms -- especially
at wide separations.

Single moon, i = 50 deg, 10 uas, 1 hr cadence, 5 yr.  COMPUTE-HEAVY (ellipse fit per trial
period); parallel across separations, one cached at a time.  Run on a multicore machine:

    python analysis/matched_filter_mmin.py
    python analysis/matched_filter_mmin.py --plot      # replot from cache
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

AS = [3, 5, 8, 10, 20, 50, 80, 100, 120]             # R_Jup
PREC = 1e-5                                           # 10 uas
NTRIALS = 7                                           # median over noise realizations
NITER = 6                                             # mass bisection iterations
NCTRL = 300                                           # zero-signal control trials for calibration
CACHE = os.path.join(HERE, "_mf_mmin.pkl")
OUT = os.path.join(HERE, "matched_filter_mmin.png")
SINE_THR = 5.0


def _ctrl(args):
    seed, stat = args
    return run_trial(SimParams(moon_a=10.0, moon_mass=0.0, moon_inc=50, moon_ecc=0.0,
                               astrometric_precision=PREC, pend=500.0), seed=seed, stat=stat)["sig"]


def calibrate(workers):
    with Pool(workers) as pool:
        sine = np.array(pool.map(_ctrl, [(s, "sine") for s in range(NCTRL)]))
        ell = np.array(pool.map(_ctrl, [(1000 + s, "ellipse") for s in range(NCTRL)]))
    far = min(max(float(np.mean(sine > SINE_THR)), 1.0 / NCTRL), 0.5)
    thr = float(np.quantile(ell, 1.0 - far))
    print("calibration: sine FAR(dchi>5)=%.3f -> ellipse threshold=%.1f" % (far, thr), flush=True)
    return thr


def _med_sig(a, mass, stat):
    return float(np.median([
        run_trial(SimParams(moon_a=a, moon_mass=mass, moon_inc=50, moon_ecc=0.0,
                            astrometric_precision=PREC, pend=500.0), seed=1000 + k, stat=stat)["sig"]
        for k in range(NTRIALS)]))


def _mmin(a, stat, thr):
    guess = 0.05 * (10.0 / a)                          # rough anchor (m_min ~ 1/a)
    lo, hi = guess / 20.0, guess * 20.0
    while _med_sig(a, lo, stat) >= thr:
        lo /= 2.0
    while _med_sig(a, hi, stat) < thr:
        hi *= 2.0
    for _ in range(NITER):
        mid = np.sqrt(lo * hi)
        if _med_sig(a, mid, stat) >= thr:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


def _cell(task):
    a, stat, thr = task
    return a, stat, _mmin(a, stat, thr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}

    if not args.plot:
        if "thr" not in cache:
            cache["thr"] = calibrate(args.workers); pickle.dump(cache, open(CACHE, "wb"))
        thr = cache["thr"]
        tasks = [(a, s, SINE_THR if s == "sine" else thr) for a in AS for s in ("sine", "ellipse")
                 if (a, s) not in cache]
        if tasks:
            print("bisecting m_min for %d (a, stat) cells ..." % len(tasks), flush=True)
            with Pool(args.workers) as pool:
                for a, s, m in pool.map(_cell, tasks):
                    cache[(a, s)] = m
                    print("  a=%3d %-7s  m_min=%.4f" % (a, s, m), flush=True)
            pickle.dump(cache, open(CACHE, "wb"))

    aa = np.array(AS, float)
    msine = np.array([cache[(a, "sine")] for a in AS])
    mell = np.array([cache[(a, "ellipse")] for a in AS])
    VIR = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.plot(aa, msine, "D-", color=VIR(0.72), ms=6, lw=1.6, label="Single sinusoid ($\\Delta\\chi^2>5$)")
    ax.plot(aa, mell, "o-", color=VIR(0.28), ms=6, lw=1.6, label="Matched filter (calibrated cut)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    ax.set_ylabel(r"Minimum Detectable Moon Mass ($M_\oplus$)")
    ax.set_title(r"Matched Filter Detects Fainter Moons ($10\,\mu$as, same false-alarm rate)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT, flush=True)
