#!/usr/bin/env python3
"""Detection-fraction survey (for Sumin): moon mass x semimajor-axis grid.

Runs the single-moon astrometric-detection survey over
    mass = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]  Mearth
    a    = [3, 5, 8, 10, 20, 50, 80, 100, 120]                R_Jup
with 10 trials per cell, 1 hr cadence, and a 5 yr duration (the cadence and
duration are the SimParams defaults), at a chosen per-epoch precision. Each cell
is a single moon at the code defaults i = 50 deg, e = 0.05.

For each precision it writes a full set of products, tagged by precision:
    sumin_survey_<P>uas.npz        results cube (IDL .sav analogue; key = detfrac)
    sumin_survey_<P>uas.png        standard 4-panel survey figure
    sumin_success_map_<P>uas.png   polished detection-fraction map

`detfrac` (shape n_a x n_mass) is the percent-success / detection-fraction map; the
npz also holds sigcube, perrcube, amperrcube, fpcube, a_grid, mass_grid.

Usage:
    python run_sumin_survey.py                        # 10 uas (default), full run + figures
    python run_sumin_survey.py --precision-uas 50     # 50 uas run
    python run_sumin_survey.py --precision-uas 100    # 100 uas run
    python run_sumin_survey.py --precision-uas 50 --plot   # just remake figures from the npz
"""
import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exomoonsim.sim import SimParams
from exomoonsim.survey import SurveyConfig, run_survey

MASS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]     # Earth masses
A = [3, 5, 8, 10, 20, 50, 80, 100, 120]                      # R_Jup


def _names(pu):
    tag = "%duas" % int(round(pu))
    return (os.path.join(HERE, "sumin_survey_%s.npz" % tag),
            os.path.join(HERE, "sumin_survey_%s.png" % tag),
            os.path.join(HERE, "sumin_success_map_%s.png" % tag))


def _log_edges(v):
    """Cell edges (log-spaced midpoints) for pcolormesh from grid-point values."""
    lv = np.log10(np.asarray(v, float))
    mid = (lv[1:] + lv[:-1]) / 2.0
    return 10.0 ** np.concatenate([[2 * lv[0] - mid[0]], mid, [2 * lv[-1] - mid[-1]]])


def success_map(npz, out, pu):
    z = np.load(npz)
    a, m, det = z["a_grid"], z["mass_grid"], z["detfrac"]    # det: (n_a, n_mass)
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    pcm = ax.pcolormesh(_log_edges(a), _log_edges(m), det.T, cmap="viridis",
                        vmin=0, vmax=1, shading="flat")       # x = separation, y = mass
    for i, av in enumerate(a):                               # annotate each cell with the %
        for j, mv in enumerate(m):
            ax.text(av, mv, "%.0f" % (det[i, j] * 100), ha="center", va="center",
                    fontsize=7, color="white" if det[i, j] < 0.55 else "black")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(a); ax.set_xticklabels(["%g" % x for x in a])
    ax.set_yticks(m); ax.set_yticklabels(["%g" % x for x in m])
    ax.set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
    ax.set_ylabel(r"Moon Mass ($M_\oplus$)")
    ax.set_title(r"Detection Fraction  ($%d\,\mu$as, %d trials/cell, 1 hr, 5 yr)"
                 % (int(round(pu)), int(z["ntrials"])))
    cb = fig.colorbar(pcm, ax=ax); cb.set_label("Detection Fraction")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision-uas", type=float, default=10.0,
                    help="per-epoch astrometric precision in micro-arcsec")
    ap.add_argument("--plot", action="store_true", help="remake figures from the saved npz")
    args = ap.parse_args()
    npz, std_fig, map_fig = _names(args.precision_uas)

    if not args.plot:
        base = SimParams(astrometric_precision=args.precision_uas * 1e-6, pend=500.0)  # 1 hr & 5 yr = defaults
        cfg = SurveyConfig(a_grid=np.array(A, float), mass_grid=np.array(MASS, float),
                           ntrials=10, base=base, nworkers=0, base_seed=12345)
        ntot = len(A) * len(MASS) * cfg.ntrials
        print("running %d a x %d mass x 10 trials = %d trials at %g uas ..."
              % (len(A), len(MASS), ntot, args.precision_uas), flush=True)
        res = run_survey(cfg)
        res.save(npz); print("saved", npz, flush=True)
        from exomoonsim.plots import plot_survey
        plot_survey(res, filename=std_fig); print("saved", std_fig, flush=True)

    success_map(npz, map_fig, args.precision_uas)
