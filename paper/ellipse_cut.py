#!/usr/bin/env python3
"""Calibrate the projected-ellipse (matched-filter) detection threshold.

The ellipse statistic is a raw 4-DOF chi^2, on a different scale than the single sinusoid's
reduced chi^2, so its detection cut must be recalibrated to the SAME pure-noise false-alarm rate
as the single-sinusoid Delta chi^2 > 5 cut of Papers I and II.  We do that empirically: run a set
of zero-mass (no-signal) control trials with each statistic in the SAME search configuration, take
the single-sinusoid false-alarm rate at Delta chi^2 > 5, and set the ellipse threshold to the
quantile of the ellipse peak distribution that gives the same rate.

The threshold depends on the search configuration (period grid = pend, ptestwidth, campaign length
and cadence = texp, duration) through the look-elsewhere effect, but is essentially independent of
precision (the statistic is normalized by sigma^2).  Calibrate once per search configuration and
reuse across precisions/masses/separations.

Importable:  from ellipse_cut import calibrate_threshold
CLI (quick check):  python ellipse_cut.py --nctrl 60 --workers 8 --precision 50
"""
import argparse
from multiprocessing import Pool

import numpy as np

from exomoonsim.sim import SimParams, run_trial


def _peak(args):
    seed, stat, cfg = args
    return float(run_trial(SimParams(moon_mass=0.0, **cfg), seed=seed, stat=stat)["sig"])


def calibrate_threshold(cfg, nctrl=300, workers=None, sine_cut=5.0, seed0=0, verbose=True):
    """Ellipse threshold matching the single-sinusoid Delta chi^2 > ``sine_cut`` false-alarm rate.

    cfg      : dict of SimParams kwargs defining the SEARCH configuration (astrometric_precision,
               pend, ptestwidth, texp, ...); ``moon_mass=0`` is forced for the no-signal control.
    nctrl    : zero-mass control trials per statistic.
    Returns (threshold, far).
    """
    ts = [(seed0 + s, "sine", cfg) for s in range(nctrl)]
    te = [(seed0 + 10_000 + s, "ellipse", cfg) for s in range(nctrl)]
    if workers and workers > 1:
        with Pool(workers) as pool:
            sine = np.array(pool.map(_peak, ts))
            ell = np.array(pool.map(_peak, te))
    else:
        sine = np.array([_peak(t) for t in ts])
        ell = np.array([_peak(t) for t in te])
    far = float(np.mean(sine > sine_cut))
    far = min(max(far, 1.0 / nctrl), 0.5)                 # guard against 0 or absurd rates
    thr = float(np.quantile(ell, 1.0 - far))
    if verbose:
        print("calibrate: nctrl=%d  sine FAR(dchi>%.1f)=%.3f  ->  ellipse threshold=%.2f"
              % (nctrl, sine_cut, far, thr), flush=True)
    return thr, far


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nctrl", type=int, default=300)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--precision", type=float, default=50.0, help="per-epoch precision, micro-arcsec")
    ap.add_argument("--pend", type=float, default=500.0)
    ap.add_argument("--ptestwidth", type=float, default=None)
    ap.add_argument("--texp", type=float, default=1.0)
    args = ap.parse_args()
    cfg = dict(moon_a=10.0, moon_inc=50.0, moon_ecc=0.0,
               astrometric_precision=args.precision * 1e-6, pend=args.pend, texp=args.texp)
    if args.ptestwidth is not None:
        cfg["ptestwidth"] = args.ptestwidth
    thr, far = calibrate_threshold(cfg, nctrl=args.nctrl, workers=args.workers)
    print("threshold=%.3f  far=%.3f" % (thr, far))
