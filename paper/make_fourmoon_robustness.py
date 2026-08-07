#!/usr/bin/env python3
"""Seed-robustness of the four-moon recovery (Paper III, referee-proofing).

The headline four-moon result (Section 4.2, Figure/Table) is shown for a single noise
realization (seed 7). This script repeats the identical blind recovery over many independent
noise seeds and asks: how often are all four moons recovered, what is the per-moon detection
rate, how stable are the recovered masses/inclinations, and how often does a spurious
(wrong-period) detection appear? This demonstrates the result is not cherry-picked.

Same system as make_four_moon_figs.py: four low-mass moons (0.10, 0.067, 0.050, 0.083 Mearth at
20, 12, 30, 8 R_Jup; i=50 deg, offset phases, e=0.05) around the alpha Cen A giant-planet
candidate at 20 uas over a 5-yr hourly campaign, blind iterative prewhitening at Delta chi^2 > 5.

COMPUTE-HEAVY (~20 s per seed x NSEEDS); parallel across seeds -- run on a multicore machine.
Writes data/fourmoon_robustness.csv and figs/fourmoon_robustness.png. Re-run with the CSV present
(or --plot) to just replot.

Usage:
    python make_fourmoon_robustness.py                 # compute (all cores) then plot
    python make_fourmoon_robustness.py --nseeds 50 --workers 16
    python make_fourmoon_robustness.py --plot          # replot from the saved CSV
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
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams, Moon, run_trial

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
DATA = os.path.join(HERE, "data", "fourmoon_robustness.csv")
OUT = os.path.join(FIGDIR, "fourmoon_robustness.png")

# ------- the four-moon system (identical to make_four_moon_figs.py) --------- #
MOONS = [(20, 0.30 / 3), (12, 0.20 / 3), (30, 0.15 / 3), (8, 0.25 / 3)]   # a (R_Jup), m (Mearth)
FRACS = (0.15, 0.55, 0.80, 0.35)                                          # orbital-phase offsets
PREC = 2e-5                                                               # 20 uas
A_IN = sorted(a for a, _ in MOONS)                                        # [8, 12, 20, 30]
M_OF = {a: m for a, m in MOONS}


def _build():
    ms = [Moon(a=a, mass=m, inc=50, ecc=0.05) for a, m in MOONS]
    p = SimParams(astrometric_precision=PREC, texp=1.0, pend=60.0, ptestwidth=0.0015, moons=ms)
    for mn, fr in zip(p.moons, FRACS):
        mn.t0 = fr * p._period_days(mn) / 365.25
    return p


def _is_real(d, r):
    ts = [d["true_synodic"]] + list(d["companion_synodics"])
    return any(abs(r["best_period"] / t - 1) < 0.05 for t in ts)


def one_seed(seed):
    """Return per-seed row: seed, n_real, n_spurious, then (det, m_rec, i_rec) per input a."""
    d = run_trial(_build(), seed=int(seed), return_diagnostics=True)["diag"]
    done = [x for x in d["recoveries"] if x["recovered"]]
    real = [r for r in done if _is_real(d, r)]
    nsp = len(done) - len(real)
    row = {"seed": int(seed), "n_real": len(real), "n_spurious": nsp}
    for a in A_IN:                                   # match each input moon to nearest recovered a
        near = [r for r in real if abs(r["a_rjup"] - a) < 0.15 * a]
        if near:
            r = min(near, key=lambda r: abs(r["a_rjup"] - a))
            row[f"det_{a}"], row[f"m_{a}"], row[f"i_{a}"] = 1, r["mass"], r["inclination"]
        else:
            row[f"det_{a}"], row[f"m_{a}"], row[f"i_{a}"] = 0, np.nan, np.nan
    print("  seed %3d: %d/4 real, %d spurious" % (seed, len(real), nsp), flush=True)
    return row


def compute(nseeds, workers):
    seeds = list(range(1, nseeds + 1))
    print("recovering the four-moon system over %d seeds on %d workers ..." % (nseeds, workers),
          flush=True)
    with Pool(workers) as pool:
        rows = pool.map(one_seed, seeds)
    cols = ["seed", "n_real", "n_spurious"] + sum([[f"det_{a}", f"m_{a}", f"i_{a}"] for a in A_IN], [])
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    with open(DATA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(rows, key=lambda r: r["seed"]):
            w.writerow(r)
    print("wrote", DATA, flush=True)
    return rows


def load():
    out = []
    for r in csv.DictReader(open(DATA)):
        out.append({k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items()})
    return out


def plot(rows):
    n = len(rows)
    nreal = np.array([r["n_real"] for r in rows])
    nsp = np.array([r["n_spurious"] for r in rows])
    allfour = 100.0 * np.mean(nreal == 4)
    VIR = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # left: recovered mass per moon across seeds (points) over the input value
    for k, a in enumerate(A_IN):
        mm = np.array([r[f"m_{a}"] for r in rows]); det = np.array([r[f"det_{a}"] for r in rows])
        good = np.isfinite(mm)
        x = np.full(good.sum(), k) + np.random.uniform(-0.12, 0.12, good.sum())
        ax[0].scatter(x, mm[good], s=14, color=VIR(0.2 + 0.6 * k / 3), alpha=0.6, zorder=3)
        ax[0].plot([k - 0.25, k + 0.25], [M_OF[a]] * 2, color="0.2", lw=2, zorder=4)
        ax[0].text(k, M_OF[a] * 1.14, "%.0f%%" % (100 * det.mean()), ha="center", fontsize=8,
                   color="0.3")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels([r"$%d$" % a for a in A_IN])
    ax[0].set_xlabel(r"Input Moon Separation ($R_{\rm Jup}$)")
    ax[0].set_ylabel(r"Recovered Moon Mass ($M_\oplus$)")
    ax[0].set_title("Recovered Mass per Moon (bars: input; %: detection rate)", fontsize=9)
    ax[0].grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)

    # right: recovered inclination per moon across seeds (input 50 deg)
    for k, a in enumerate(A_IN):
        ii = np.array([r[f"i_{a}"] for r in rows]); good = np.isfinite(ii)
        x = np.full(good.sum(), k) + np.random.uniform(-0.12, 0.12, good.sum())
        ax[1].scatter(x, ii[good], s=14, color=VIR(0.2 + 0.6 * k / 3), alpha=0.6, zorder=3)
    ax[1].axhline(50.0, color="0.2", lw=2, zorder=4)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([r"$%d$" % a for a in A_IN])
    ax[1].set_xlabel(r"Input Moon Separation ($R_{\rm Jup}$)")
    ax[1].set_ylabel(r"Recovered Inclination (deg)")
    ax[1].set_title(r"Recovered Inclination per Moon (line: input $50^\circ$)", fontsize=9)
    ax[1].grid(True, axis="y", ls=":", lw=0.4, alpha=0.5)
    fig.suptitle("Four-Moon Recovery Over %d Noise Seeds: All Four Recovered in %.0f%%, "
                 "%.2f Spurious per Seed" % (n, allfour, nsp.mean()), fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote %s  (all-four %.0f%%, mean spurious %.2f/seed, N=%d)" % (OUT, allfour, nsp.mean(), n),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot and os.path.exists(DATA):
        plot(load())
    elif os.path.exists(DATA):
        print("found", DATA, "-- replotting (delete it to recompute)")
        plot(load())
    else:
        plot(compute(args.nseeds, args.workers))
