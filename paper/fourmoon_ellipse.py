#!/usr/bin/env python3
"""Four-moon blind recovery with the projected-ellipse matched filter (Paper III re-architecture).

Reproduces the four-moon system of make_four_moon_figs.py but runs the iterative prewhitening
recovery with stat="ellipse" and the detection cut CALIBRATED to the single-sinusoid Delta chi^2>5
false-alarm rate.  Prints, at 20 and 50 uas, which moons are recovered and their (raw ellipse)
significances -- i.e. the new Table 3 and the answer to "does the matched filter clear more of the
four moons at 50 uas than the single sinusoid (which cleared only one)?"

Moderately heavy (calibration control trials + two ellipse recoveries with a full period search).
Run on a multicore machine:

    python fourmoon_ellipse.py --nctrl 300 --workers 16
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from exomoonsim.sim import SimParams, Moon, run_trial
from ellipse_cut import calibrate_threshold

MOONS = [(20, 0.30 / 3), (12, 0.20 / 3), (30, 0.15 / 3), (8, 0.25 / 3)]   # a (Rjup), m (Mearth)
FRACS = (0.15, 0.55, 0.80, 0.35)
SEED = 7
SEARCH = dict(texp=1.0, pend=60.0, ptestwidth=0.0015)


def build(prec):
    moons = [Moon(a=a, mass=m, inc=50, ecc=0.05) for a, m in MOONS]
    p = SimParams(astrometric_precision=prec, moons=moons, **SEARCH)
    for mn, fr in zip(p.moons, FRACS):
        mn.t0 = fr * p._period_days(mn) / 365.25
    return p


def is_real(d, r):
    tsyns = [d["true_synodic"]] + list(d["companion_synodics"])
    return any(abs(r["best_period"] / ts - 1) < 0.05 for ts in tsyns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nctrl", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--thr", type=float, default=None,
                    help="use this calibrated ellipse cut and skip recalibration")
    args = ap.parse_args()

    if args.thr is not None:
        thr = args.thr
        print("using supplied ellipse cut = %.2f\n" % thr)
    else:
        thr, far = calibrate_threshold(dict(astrometric_precision=5e-5, **SEARCH),
                                       nctrl=args.nctrl, workers=args.workers)
        print("ellipse cut = %.2f (matched FAR=%.3f)\n" % (thr, far))

    for prec, lab in [(2e-5, "20 uas"), (5e-5, "50 uas")]:
        p = build(prec)
        d = run_trial(p, seed=SEED, return_diagnostics=True, recover_thr=thr, stat="ellipse")["diag"]
        recs = [r for r in d["recoveries"] if r.get("recovered") and r["sig"] > thr]
        real = [r for r in recs if is_real(d, r)]
        print("=== %s : %d/4 moons recovered (ellipse cut %.1f) ===" % (lab, len(real), thr))
        for r in sorted(recs, key=lambda x: -x["sig"]):
            tag = "real" if is_real(d, r) else "SPURIOUS"
            print("   P=%7.2f d   dchi_ellipse=%8.1f   %s" % (r["best_period"], r["sig"], tag))
        print()
