#!/usr/bin/env python3
"""Multiple-moon robustness experiment (Paper III): does adding moons manufacture
false positives?

For N = 1..5 nested moons (all i=50 deg, e=0.05, offset phases) around the alpha Cen A
giant-planet candidate at 10 uas over 5 yr, we run the blind prewhitening recovery to
completion and record, in the terminating (null) round, two things:

  * the largest leftover periodogram peak at the already-claimed moon periods
    (residual power that imperfect subtraction leaves behind), and
  * the largest peak OUTSIDE those excluded windows --- the significance the search
    would actually act on, i.e. the spurious-detection risk.

Every one of the N moons is recovered at each N. The leftover at claimed periods can
climb well above the Delta chi^2 = 5 detection threshold, but because the search
excludes the claimed-period windows, the out-of-window peak stays below threshold, so
additional moons do not produce spurious detections. This is the multiple-moon analog
of the false-positive result of Section 4 (sec:fp).

Produces figs/confusion_vs_n.png and prints the table. Config matches the four-moon
figure (hourly cadence, synodic search < 60 d). Per-N results are cached in
data/_confusion.npz so a long run can resume; delete it to recompute. Run:
    python make_confusion_fig.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))       # repo root -> exomoonsim package
from exomoonsim.sim import SimParams, Moon, run_trial

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
_ELL = "--ellipse" in sys.argv                          # matched-filter statistic + calibrated cut
STAT = "ellipse" if _ELL else "sine"
THR = 28.95 if _ELL else 5.0                            # calibrated ellipse cut (pend=60 search config)
CACHE = os.path.join(HERE, "data", "_confusion%s.npz" % ("_ellipse" if _ELL else ""))
OUT = os.path.join(FIGDIR, "confusion_vs_n%s.png" % ("_ellipse" if _ELL else ""))

SUPER = [(20, 0.30), (12, 0.20), (30, 0.15), (8, 0.25), (16, 0.18)]   # nested moon set
FRACS = [0.15, 0.55, 0.80, 0.35, 0.65]                                 # offset phases
NS = [1, 2, 3, 4, 5]
BLANK = 0.25                                            # run_trial recover_blank default


def run_N(n):
    moons = [Moon(a=a, mass=m, inc=50, ecc=0.05) for a, m in SUPER[:n]]
    p = SimParams(astrometric_precision=1e-5, texp=1.0, pend=60.0, ptestwidth=0.0015, moons=moons)
    for mn, fr in zip(p.moons, FRACS[:n]):
        mn.t0 = fr * p._period_days(mn) / 365.25
    d = run_trial(p, seed=7, return_diagnostics=True, recover_thr=THR, stat=STAT)["diag"]
    done = [x for x in d["recoveries"] if x["recovered"]]
    claimed = [r["best_period"] for r in done]
    nr = [x for x in d["recoveries"] if not x["recovered"]]
    if nr:
        pg, sg = nr[0]["period_grid"], np.asarray(nr[0]["periodogram"], float)
        leftover = float(np.nanmax(sg))
        keep = np.ones(sg.size, bool)
        for cp in claimed:
            keep &= np.abs(pg / cp - 1.0) >= BLANK
        oow = float(np.nanmax(sg[keep])) if keep.any() else np.nan
    else:
        leftover = oow = np.nan
    return leftover, oow, len(done)


cache = {}
if os.path.exists(CACHE):
    z = np.load(CACHE)
    cache = {int(k): tuple(v) for k, v in zip(z["N"], z["res"])}
for n in NS:
    if n not in cache:
        print("computing N=%d ..." % n, flush=True)
        cache[n] = run_N(n)
        Narr = np.array(sorted(cache)); res = np.array([cache[k] for k in Narr])
        np.savez(CACHE, N=Narr, res=res)
if not all(n in cache for n in NS):
    print("partial results cached; re-run to continue"); sys.exit(0)

leftover = np.array([cache[n][0] for n in NS])
oow = np.array([cache[n][1] for n in NS])
nrec = np.array([int(cache[n][2]) for n in NS])
print(" N  recovered  leftover_peak  out_of_window_peak")
for i, n in enumerate(NS):
    print("  %d     %d/%d       %6.1f          %6.1f" % (n, nrec[i], n, leftover[i], oow[i]))

VIR = plt.get_cmap("viridis")
fig, ax = plt.subplots(figsize=(6.0, 4.3))
ax.plot(NS, leftover, "o-", color=VIR(0.70), lw=1.6, ms=6,
        label="Leftover at Claimed Periods (Excluded)")
ax.plot(NS, oow, "s-", color=VIR(0.28), lw=1.6, ms=6,
        label="Largest Out-of-Window Peak")
ax.axhline(THR, color=VIR(0.05), ls=":", lw=1.1)
ax.text(NS[-1], THR * 1.06, (r"$\Delta\chi^2_{\rm mf}=%.0f$ cut" % THR) if _ELL else r"$\Delta\chi^2=5$ threshold",
        fontsize=8, color="0.4", ha="right", va="bottom")
ax.set_xticks(NS)
ax.set_xlabel("Number of Moons in the System")
ax.set_ylabel(r"Null-Round Peak  $\Delta\chi^2$")
ax.set_title(("No Out-of-Window False Positives Between Distinct Moons" if _ELL
              else "All $N$ Moons Recovered; No Out-of-Window False Positives"), fontsize=9.5)
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
