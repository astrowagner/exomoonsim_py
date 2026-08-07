#!/usr/bin/env python3
"""10 uas detection-fraction map with the projected-ellipse matched filter (for Sumin).

The matched-filter statistic is a raw chi^2 (different scale from the single-sinusoid reduced
chi^2), so its detection threshold must be recalibrated.  This script:

  1. Calibrates the ellipse threshold to the SAME pure-noise false-alarm rate as the
     single-sinusoid Delta chi^2 > 5 cut, using a zero-mass (no-signal) control at 10 uas.
  2. Runs the (mass x separation) survey at 10 uas with stat='ellipse' and that threshold.
  3. Plots the ellipse success map -- and, if the single-sinusoid 10 uas map
     (sumin_survey_10uas.npz) is present, shows the two side by side for comparison.

Same grid as run_sumin_survey.py.  COMPUTE-HEAVY (the ellipse fit runs a least squares at
every trial period).  Run on a multicore machine:

    python analysis/matched_filter_map.py
    python analysis/matched_filter_map.py --plot      # replot from the saved npz + threshold
"""
import os
import sys
import argparse
import numpy as np
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir)
sys.path.insert(0, ROOT)
from exomoonsim.sim import SimParams, run_trial
from exomoonsim.survey import SurveyConfig, run_survey

MASS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
A = [3, 5, 8, 10, 20, 50, 80, 100, 120]
PREC = 1e-5                                           # 10 uas
NCTRL = 300                                           # zero-signal control trials for calibration
OUT_NPZ = os.path.join(ROOT, "sumin_survey_ellipse_10uas.npz")
THR_TXT = os.path.join(ROOT, "sumin_ellipse_threshold_10uas.txt")
MAP_FIG = os.path.join(ROOT, "sumin_success_map_ellipse_10uas.png")
SINE_NPZ = os.path.join(ROOT, "sumin_survey_10uas.npz")   # from run_sumin_survey.py (optional)


def _ctrl(args):
    seed, stat = args
    return run_trial(SimParams(moon_a=10.0, moon_mass=0.0, moon_inc=50, moon_ecc=0.0,
                               astrometric_precision=PREC, pend=500.0), seed=seed, stat=stat)["sig"]


def calibrate(workers):
    """Ellipse threshold matching the single-sinusoid Delta chi^2 > 5 false-alarm rate."""
    with Pool(workers) as pool:
        sine = np.array(pool.map(_ctrl, [(s, "sine") for s in range(NCTRL)]))
        ell = np.array(pool.map(_ctrl, [(1000 + s, "ellipse") for s in range(NCTRL)]))
    far = float(np.mean(sine > 5.0))                 # sine false-alarm rate at the fiducial cut
    far = min(max(far, 1.0 / NCTRL), 0.5)
    thr = float(np.quantile(ell, 1.0 - far))          # ellipse cut with the same FAR
    print("calibration: sine FAR(dchi>5) = %.3f  ->  ellipse threshold = %.1f" % (far, thr), flush=True)
    return thr


def _log_edges(v):
    lv = np.log10(np.asarray(v, float)); mid = (lv[1:] + lv[:-1]) / 2.0
    return 10.0 ** np.concatenate([[2 * lv[0] - mid[0]], mid, [2 * lv[-1] - mid[-1]]])


def _draw(ax, aa, mm, det, title):
    pcm = ax.pcolormesh(_log_edges(aa), _log_edges(mm), det.T, cmap="viridis",
                        vmin=0, vmax=1, shading="flat")
    for i, av in enumerate(aa):
        for j, mv in enumerate(mm):
            ax.text(av, mv, "%.0f" % (det[i, j] * 100), ha="center", va="center",
                    fontsize=7, color="white" if det[i, j] < 0.55 else "black")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(aa); ax.set_xticklabels(["%g" % x for x in aa])
    ax.set_yticks(mm); ax.set_yticklabels(["%g" % x for x in mm])
    ax.set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
    ax.set_ylabel(r"Moon Mass ($M_\oplus$)")
    ax.set_title(title, fontsize=10)
    return pcm


def plot():
    z = np.load(OUT_NPZ)
    aa, mm, det = z["a_grid"], z["mass_grid"], z["detfrac"]
    have_sine = os.path.exists(SINE_NPZ)
    fig, ax = plt.subplots(1, 2 if have_sine else 1, figsize=(13 if have_sine else 7.4, 5.6),
                           squeeze=False)
    if have_sine:
        zs = np.load(SINE_NPZ)
        pcm = _draw(ax[0, 0], zs["a_grid"], zs["mass_grid"], zs["detfrac"],
                    r"Single Sinusoid ($10\,\mu$as)")
        _draw(ax[0, 1], aa, mm, det, r"Projected-Ellipse Matched Filter ($10\,\mu$as)")
    else:
        pcm = _draw(ax[0, 0], aa, mm, det, r"Matched-Filter Detection Fraction ($10\,\mu$as)")
    fig.colorbar(pcm, ax=ax.ravel().tolist(), label="Detection Fraction", fraction=0.046)
    fig.savefig(MAP_FIG, dpi=200, bbox_inches="tight")
    print("wrote", MAP_FIG, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true", help="replot from saved npz")
    args = ap.parse_args()

    if not args.plot:
        thr = calibrate(args.workers)
        open(THR_TXT, "w").write("%.6f\n" % thr)
        cfg = SurveyConfig(a_grid=np.array(A, float), mass_grid=np.array(MASS, float),
                           ntrials=10, base=SimParams(astrometric_precision=PREC, pend=500.0),
                           nworkers=args.workers, stat="ellipse", chsq_thr=thr)
        print("running %dx%dx10 = %d ellipse trials ..." % (len(A), len(MASS), len(A) * len(MASS) * 10),
              flush=True)
        res = run_survey(cfg)
        res.save(OUT_NPZ); print("saved", OUT_NPZ, flush=True)
    plot()
