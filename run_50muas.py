#!/usr/bin/env python3
"""
Production run: duplicate the IDL non-fast survey (exomoons_par.pro) in Python,
at 50 micro-arcsecond astrometric precision.

Grid (matches exomoons_par.pro, non-fast):
    masses : 0.00 .. 1.00 Mearth in 0.01 steps      -> 101 values
    a      : [5,6,7,8,9,10,15,20,25] Rjup            ->   9 values
    trials : 50 per cell
    => 45,450 trials
5-yr campaign, synodic period search to 500 d, detection thresholds
chi2>5 / period<5% / amplitude<25% (the survey defaults).

Run from anywhere (the package is pip-installed):
    python run_50muas.py

It uses ALL CPU cores and writes results + figure into runs/.
To change trials or precision, edit the four marked lines in main().

NOTE: the real work lives under `if __name__ == "__main__":`.  On macOS Python
uses the "spawn" start method, which re-imports this file in every worker
process; without that guard each worker would re-launch the whole survey (a
fork bomb).  Keep the guard.
"""
import os

# Keep each worker single-threaded so N processes don't oversubscribe the BLAS
# (each trial is dominated by numpy bincount + tiny linear solves).  Must be set
# before numpy is imported -- and this module is re-imported in every spawned
# worker, so setting it here covers the workers too.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import numpy as np

from exomoonsim.sim import SimParams
from exomoonsim.survey import SurveyConfig, run_survey
from exomoonsim.plots import plot_survey


def main():
    # ---- configuration (matches the IDL non-fast run) ---------------------
    masses    = np.round(np.arange(101) * 0.01, 2)          # 0.00 .. 1.00 Mearth
    a_grid    = np.array([5, 6, 7, 8, 9, 10, 15, 20, 25], float)   # Rjup
    ntrials   = 50                                          # trials per cell
    precision = 5e-5                                        # arcsec = 50 muas
    # -----------------------------------------------------------------------

    base = SimParams(astrometric_precision=precision)       # 5-yr, pend=500 by default
    cfg = SurveyConfig(a_grid=a_grid, mass_grid=masses, ntrials=ntrials,
                       base=base, nworkers=0)               # nworkers=0 -> all cores

    ntot = a_grid.size * masses.size * ntrials
    print(f"exomoonsim {precision*1e6:.0f} muas run: "
          f"{masses.size} masses x {a_grid.size} a x {ntrials} trials = {ntot} trials")
    print(f"  precision = {precision*1e6:.0f} muas   pend = {base.pend:.0f} d   "
          f"duration = {base.duration_days/365.25:.1f} yr   cores = {os.cpu_count()}")

    os.makedirs("runs", exist_ok=True)
    out_npz = "runs/survey_50muas_ntrials50.npz"
    out_png = "runs/survey_50muas_ntrials50.png"

    t0 = time.time()
    res = run_survey(cfg)                                   # progress prints every ~5%
    print(f"done in {(time.time() - t0) / 60:.1f} min")

    res.save(out_npz)
    plot_survey(res, filename=out_png)
    print("saved results ->", out_npz)
    print("saved figure  ->", out_png)


if __name__ == "__main__":
    main()
