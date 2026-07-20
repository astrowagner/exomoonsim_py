"""Command-line interface:  exomoon-survey  (or  python -m exomoonsim.cli)."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .survey import (SurveyConfig, run_survey, DEFAULT_A, DEFAULT_MASS,
                     FAST_A, FAST_MASS)
from .sim import SimParams


def _grid(arg, default):
    if arg is None:
        return default
    return np.array([float(x) for x in arg.split(",")], float)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="exomoon-survey",
        description="Run the exomoon astrometric-detection parameter survey.")
    ap.add_argument("--fast", action="store_true",
                    help="small 3x3 grid for a quick test")
    ap.add_argument("--ntrials", type=int, default=50, help="trials per cell")
    ap.add_argument("--precision", type=float, default=1e-4,
                    help="astrometric precision in arcsec (1e-5 = 10 muas)")
    ap.add_argument("--pend", type=float, default=500.0,
                    help="max synodic search period, days (covers a up to ~120 RJup)")
    ap.add_argument("--a-grid", default=None,
                    help="comma-separated moon semimajor axes (RJup)")
    ap.add_argument("--mass-grid", default=None,
                    help="comma-separated moon masses (MEarth)")
    ap.add_argument("--workers", type=int, default=0, help="0 = all cores")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--output", default="survey.npz", help="results .npz path")
    ap.add_argument("--plot", default=None, help="figure path (e.g. survey.png)")
    args = ap.parse_args(argv)

    base = SimParams(astrometric_precision=args.precision, pend=args.pend)
    if args.fast:
        a_grid, mass_grid = FAST_A, FAST_MASS
    else:
        a_grid = _grid(args.a_grid, DEFAULT_A)
        mass_grid = _grid(args.mass_grid, DEFAULT_MASS)

    cfg = SurveyConfig(a_grid=a_grid, mass_grid=mass_grid, ntrials=args.ntrials,
                       base=base, nworkers=args.workers, base_seed=args.seed)
    ntot = a_grid.size * mass_grid.size * cfg.ntrials
    print(f"grid: {a_grid.size} a x {mass_grid.size} mass x {cfg.ntrials} trials "
          f"= {ntot} trials   (precision={args.precision*1e6:.0f} muas, pend={args.pend:.0f} d)")

    t0 = time.perf_counter()
    res = run_survey(cfg)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min")

    res.save(args.output)
    print("saved results ->", args.output)
    if args.plot:
        from .plots import plot_survey
        plot_survey(res, filename=args.plot)
        print("saved figure  ->", args.plot)


if __name__ == "__main__":
    main()
