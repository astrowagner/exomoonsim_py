#!/usr/bin/env python3
"""50 uas survey with the projected-ellipse matched-filter statistic (Paper III re-architecture).

Same 9 (semimajor axis) x 101 (mass) grid, 50 trials/cell, 50 uas as run_50muas.py, but with
stat="ellipse" and the detection cut CALIBRATED to the same pure-noise false-alarm rate as the
single-sinusoid Delta chi^2 > 5 cut.  Writes a parallel npz next to the single-sinusoid one so the
two can be compared directly (detection maps, false-positive fraction, significance).

Heavy: 45,450 ellipse trials plus the calibration control (2 x NCTRL trials).  Run on a multicore
machine.  Output: runs/survey_50muas_ntrials50_ellipse.npz  (+ .png).

    python run_50muas_ellipse.py                 # all cores
    python run_50muas_ellipse.py --nctrl 300     # calibration control trials (default 300)
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))
from exomoonsim.sim import SimParams
from exomoonsim.survey import SurveyConfig, run_survey
from exomoonsim.plots import plot_survey
from ellipse_cut import calibrate_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nctrl", type=int, default=300, help="zero-mass control trials for calibration")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    masses    = np.round(np.arange(101) * 0.01, 2)
    a_grid    = np.array([5, 6, 7, 8, 9, 10, 15, 20, 25], float)
    ntrials   = 50
    precision = 5e-5

    # 1) calibrate the ellipse cut to the single-sinusoid FAR at this survey's search config
    print("calibrating ellipse threshold (pend=500, texp=1, 50 uas) ...", flush=True)
    thr, far = calibrate_threshold(dict(astrometric_precision=precision, pend=500.0, texp=1.0),
                                   nctrl=args.nctrl, workers=args.workers)

    # 2) run the survey with the matched filter and the calibrated cut
    base = SimParams(astrometric_precision=precision)
    cfg = SurveyConfig(a_grid=a_grid, mass_grid=masses, ntrials=ntrials, base=base,
                       nworkers=0, stat="ellipse", chsq_thr=thr)
    ntot = a_grid.size * masses.size * ntrials
    print(f"ellipse survey: {masses.size} masses x {a_grid.size} a x {ntrials} trials = {ntot} "
          f"(cut dchi_ellipse > {thr:.2f}, matched FAR={far:.3f})", flush=True)

    os.makedirs("runs", exist_ok=True)
    out_npz = "runs/survey_50muas_ntrials50_ellipse.npz"
    out_png = "runs/survey_50muas_ntrials50_ellipse.png"
    t0 = time.time()
    res = run_survey(cfg)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    res.save(out_npz)
    plot_survey(res, filename=out_png)
    # stash the calibrated threshold alongside for the manuscript / comparison scripts
    open("runs/ellipse_threshold_50uas.txt", "w").write("%.6f  # FAR=%.4f\n" % (thr, far))
    print("saved ->", out_npz, "and runs/ellipse_threshold_50uas.txt")


if __name__ == "__main__":
    main()
