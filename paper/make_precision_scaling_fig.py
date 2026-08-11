#!/usr/bin/env python3
"""Empirical mass-sensitivity floor vs astrometric precision, for several moon separations.

Extends the single-moon precision scaling of Paper~II. For each moon separation and each
per-epoch precision sigma, we run the full blind end-to-end recovery and bisect on moon
mass for the minimum mass whose median blind Delta chi^2 clears the detection threshold
(Delta chi^2 = 5). This is the realized (blind-search) sensitivity floor, including the
period-search penalty -- not an analytic matched-filter estimate.

Physics: the reflex amplitude is A ~ a * m / (M_pl + m), and the matched-filter
significance scales as Delta chi^2 ~ (A/sigma)^2, so in the regime m << M_pl the floor
obeys  m_min ~ sigma / a  (slope 1 in precision; larger separations detect lighter moons).
It steepens toward the planet mass as the barycentric reflex saturates.

This is COMPUTE-HEAVY: (n_separations x n_precisions) cells, each a mass bisection of
~8 x NTRIALS blind trials. It parallelizes across cells -- run it on a multicore machine.
Results are written to data/precision_scaling.csv; re-running with that file present just
replots (fast), so styling can be iterated without recomputing.

Usage:
    python make_precision_scaling_fig.py                 # compute (all cores) then plot
    python make_precision_scaling_fig.py --workers 16    # choose core count
    python make_precision_scaling_fig.py --plot          # replot only, from the saved CSV
    python make_precision_scaling_fig.py --recompute     # ignore CSV and recompute
    python make_precision_scaling_fig.py --quick         # tiny fast smoke test
"""
import os
import sys
import csv
import argparse
import numpy as np
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))       # repo root -> exomoonsim package
from exomoonsim.sim import SimParams, run_trial

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
DATADIR = os.path.join(HERE, "data")
OUT = os.path.join(FIGDIR, "precision_scaling.png")

# ------------------------------- configuration ----------------------------- #
SEPARATIONS = [10.0, 20.0, 40.0]                         # moon semimajor axis, R_Jup
PRECS_UAS = [2, 5, 10, 20, 50, 100, 200, 500, 1000]      # per-epoch precision, micro-arcsec
THR = 5.0                                                # detection threshold in Delta chi^2
NTRIALS = 21                                             # noise realizations per (a, sigma, mass); median
NITER = 6                                                # bisection iterations in log-mass (~1.5%)
MPL = 0.3 * 317.8                                        # planet mass, Earth masses (0.3 M_Jup)
# Fixed synodic-period search window covering ALL separations with margin for their planet-motion
# sidebands: the a=40 R_Jup moon has P_syn ~ 60 d, so pend must sit well above that (pend=60 would
# truncate its peak and spuriously raise its floor). The grid is log-spaced, so widening pend adds
# only ~ln(pend) trial periods -- negligible per-trial cost -- and keeps the look-elsewhere floor
# identical across cells for a fair separation comparison. Hourly cadence, 5-yr campaign (defaults).
SEARCH = dict(moon_inc=50.0, moon_ecc=0.0, pend=120.0, ptestwidth=0.004)


def median_sig(a, prec_arcsec, mass, cfg):
    """Median blind Delta chi^2 over cfg['ntrials'] noise realizations at (a, sigma, mass)."""
    return float(np.median([
        run_trial(SimParams(moon_a=a, moon_mass=mass, astrometric_precision=prec_arcsec,
                            **cfg["search"]), seed=1000 + k, stat=cfg.get("stat", "sine"))["sig"]
        for k in range(cfg["ntrials"])]))


def find_mmin(task):
    """Bisect on moon mass for median Delta chi^2 == threshold at one (separation, precision) cell.

    `task` is (a_rjup, prec_uas, cfg); cfg carries search/ntrials/thr/niter so this is safe
    under both 'fork' and 'spawn' multiprocessing start methods (macOS uses 'spawn').
    """
    a, prec_uas, cfg = task
    prec = prec_uas * 1e-6                                # uas -> arcsec
    thr = cfg["thr"]
    guess = 0.03 * (prec_uas / 10.0) * (10.0 / a)         # linear-regime anchor: m_min ~ sigma/a
    lo, hi = guess / 10.0, guess * 10.0
    while median_sig(a, prec, lo, cfg) >= thr:            # ensure lo below threshold
        lo /= 2.0
    while median_sig(a, prec, hi, cfg) < thr:             # ensure hi above threshold
        hi *= 2.0
    for _ in range(cfg["niter"]):
        mid = np.sqrt(lo * hi)                            # log-mass bisection
        if median_sig(a, prec, mid, cfg) >= thr:
            hi = mid
        else:
            lo = mid
    m = float(np.sqrt(lo * hi))
    print("  a=%5.1f Rjup  sigma=%6.0f uas  ->  m_min=%.4f Mearth" % (a, prec_uas, m), flush=True)
    return (a, prec_uas, m)


def compute(datafile, seps, precs, cfg, workers):
    cells = [(a, p, cfg) for a in seps for p in precs]
    print("computing %d cells (%d separations x %d precisions), %d trials/eval, on %d workers ..."
          % (len(cells), len(seps), len(precs), cfg["ntrials"], workers), flush=True)
    with Pool(workers) as pool:
        results = pool.map(find_mmin, cells)
    os.makedirs(os.path.dirname(datafile), exist_ok=True)
    with open(datafile, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["a_rjup", "precision_uas", "m_min_Mearth"])
        for a, p, m in sorted(results):
            w.writerow([a, p, round(m, 5)])
    print("wrote", datafile, flush=True)
    return results


def load(datafile):
    rows = list(csv.DictReader(open(datafile)))
    return [(float(r["a_rjup"]), float(r["precision_uas"]), float(r["m_min_Mearth"])) for r in rows]


def plot(results, out=OUT, title=r"Mass Sensitivity vs Precision and Separation ($\Delta\chi^2>5$)"):
    VIR = plt.get_cmap("viridis")
    seps = sorted(set(a for a, _, _ in results))
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    # faint slope-1 guide (m_min proportional to sigma), anchored to the set nearest a=10
    aref = min(seps, key=lambda s: abs(s - 10))
    ref = sorted([(p, m) for a, p, m in results if a == aref])
    if len(ref) > 1:
        p0, m0 = ref[0]
        xg = np.array([r[0] for r in ref], float)
        ax.plot(xg, m0 * xg / p0, ls="--", color="0.7", lw=1.1, zorder=1,
                label=r"$m_{\min}\propto\sigma$")
    for i, a in enumerate(seps):
        pts = sorted([(p, m) for aa, p, m in results if aa == a])
        xs = [p for p, _ in pts]; ys = [m for _, m in pts]
        c = VIR(0.10 + 0.78 * i / max(1, len(seps) - 1))
        ax.plot(xs, ys, "o-", color=c, ms=6, lw=1.1, zorder=3,
                label=r"$a=%g\,R_{\rm Jup}$" % a)
    xmin = min(p for _, p, _ in results); xmax = max(p for _, p, _ in results)
    for mref, lab, xpos, ha in [(0.0123, "Earth's moon mass", xmax * 0.97, "right"),
                                (1.0, "Earth mass", xmin * 1.03, "left")]:   # physical reference lines
        ax.axhline(mref, color="0.78", ls=":", lw=0.9, zorder=0)
        ax.text(xpos, mref * 1.2, lab, fontsize=7.5, color="0.5", ha=ha, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 4.0)
    ax.set_xlabel(r"Per-Epoch Astrometric Precision $\sigma$ ($\mu$as)")
    ax.set_ylabel(r"Minimum Detectable Moon Mass ($M_\oplus$)")
    ax.set_title(title, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=1)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true", help="replot only, from the saved CSV")
    ap.add_argument("--recompute", action="store_true", help="ignore any saved CSV and recompute")
    ap.add_argument("--quick", action="store_true", help="tiny fast smoke test")
    ap.add_argument("--stat", choices=["sine", "ellipse"], default="sine",
                    help="detection statistic: 'sine' (Papers I/II) or 'ellipse' (matched filter)")
    ap.add_argument("--nctrl", type=int, default=300, help="control trials for the ellipse calibration")
    args = ap.parse_args()

    seps, precs = SEPARATIONS, PRECS_UAS
    cfg = dict(search=SEARCH, ntrials=NTRIALS, thr=THR, niter=NITER, stat=args.stat)
    suffix = "" if args.stat == "sine" else "_ellipse"
    datafile = os.path.join(DATADIR, "precision_scaling%s.csv" % suffix)
    out_fig = OUT if args.stat == "sine" else OUT.replace(".png", "_ellipse.png")

    if args.stat == "ellipse" and not (args.plot or (os.path.exists(datafile) and not args.recompute)):
        # calibrate the ellipse cut to the single-sinusoid FAR at this search config (once)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ellipse_cut import calibrate_threshold
        thr, far = calibrate_threshold(dict(astrometric_precision=10e-6, **SEARCH),
                                       nctrl=args.nctrl, workers=args.workers)
        cfg["thr"] = thr
        print("ellipse cut = %.2f (matched FAR=%.3f) at pend=%.0f, ptestwidth=%.4f"
              % (thr, far, SEARCH["pend"], SEARCH["ptestwidth"]), flush=True)

    if args.quick:      # small + fast: 1 separation, 2 precisions, few trials, narrow search
        seps, precs = [10.0], [10, 100]
        cfg = dict(search=dict(moon_inc=50.0, moon_ecc=0.0, pend=12.0, ptestwidth=0.01),
                   ntrials=3, thr=THR, niter=5)
        datafile = os.path.join(DATADIR, "_precision_scaling_quick.csv")

    keep = set(seps)                                     # only plot the configured separations
    filt = lambda rs: [r for r in rs if r[0] in keep]
    title = (r"Matched-Filter Mass Sensitivity vs Precision and Separation" if args.stat == "ellipse"
             else r"Mass Sensitivity vs Precision and Separation ($\Delta\chi^2>5$)")
    if args.plot and os.path.exists(datafile):
        plot(filt(load(datafile)), out=out_fig, title=title)
    elif os.path.exists(datafile) and not args.recompute:
        print("found", datafile, "-- replotting (use --recompute to regenerate)")
        plot(filt(load(datafile)), out=out_fig, title=title)
    else:
        plot(filt(compute(datafile, seps, precs, cfg, args.workers)), out=out_fig, title=title)
