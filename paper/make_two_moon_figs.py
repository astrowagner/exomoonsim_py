#!/usr/bin/env python3
"""Figures 2-3 (Paper III): the blind two-moon recovery, split by plot type.

Runs one fiducial two-moon trial (0.3 Mearth at 20 Rjup, 0.2 Mearth at 12 Rjup,
both i=50 deg) around the alpha Cen A giant-planet candidate at 10 uas over 5 yr,
then draws two focused figures from the diagnostics dict:

  two_moon_residuals.png     90-day separation residual, cleaned round by round
  two_moon_periodograms.png  the prewhitening period search (2 rounds + null)

(The phase-folded detections are shown for the richer four-moon system in
make_four_moon_figs.py.) Also prints the recovered parameters of Table 2.
Reproducible: fixed seed,
imports the repo's exomoonsim package. Run:  python make_two_moon_figs.py
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

FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

VIR = plt.get_cmap("viridis")
C_OBS, C_TRUE, C_FIT, C_COM = VIR(0.28), VIR(0.55), VIR(0.82), VIR(0.05)
MAS = 1e3      # arcsec -> mas
UAS = 1e6      # arcsec -> uas

# --------------------------------------------------------------------------- #
SEED = 42
p = SimParams(astrometric_precision=1e-5,               # 10 uas, 5-yr, 1-hr cadence
              moons=[Moon(a=20.0, mass=0.3, inc=50.0, ecc=0.05),
                     Moon(a=12.0, mass=0.2, inc=50.0, ecc=0.05)])
d = run_trial(p, seed=SEED, return_diagnostics=True)["diag"]
td = d["tdays"]
sig1 = d["precision_uas"]                                # per-epoch precision (uas)
recs = d["recoveries"]
done = [r for r in recs if r["recovered"]]
nullr = next((r for r in recs if not r["recovered"]), None)

# ---------------- Figure 2: 90-day residual progression -------------------- #
z = td <= 90.0
band = sig1
stages = [(td, d["rres"] * UAS, "(a) Planet Subtracted (Full Campaign)", False),
          (td[z], d["rres"][z] * UAS, "(b) First 90 d: Both Moons", True),
          (td[z], d["rres_after"][z] * UAS, "(c) First 90 d: Primary Removed", True),
          (td[z], d["rres_allremoved"][z] * UAS, "(d) First 90 d: All Removed", True)]
fig, ax = plt.subplots(1, 4, figsize=(14.0, 3.4), sharey=True)
for a, (x, y, title, zoom) in zip(ax, stages):
    a.axhspan(-band, band, color="0.85", zorder=0)
    a.scatter(x, y, s=3, color=C_OBS, alpha=0.35)
    a.set_title(title, fontsize=9)
    a.set_xlabel("Time (days)")
ax[0].set_ylabel(r"Residual ($\mu$as)")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "two_moon_residuals.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 4: prewhitening periodograms ---------------------- #
marks = [d["true_synodic"]] + list(d["companion_synodics"])
claimed = [r["best_period"] for r in done]
BLANK = 0.25                                             # matches run_trial recover_blank default
fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.6))
for k, r in enumerate(recs):
    a = ax[k]
    if not r["recovered"]:                              # shade the already-claimed (excluded) windows
        for cp in claimed:
            a.axvspan(cp * (1 - BLANK), cp * (1 + BLANK), color=C_FIT, alpha=0.20, zorder=0)
        a.text(0.03, 0.94, "shaded: claimed periods (excluded)", transform=a.transAxes,
               fontsize=7, color="0.4", va="top")
    a.semilogx(r["period_grid"], r["periodogram"], color=C_OBS, lw=0.9)
    for mk in marks:
        a.axvline(mk, color="0.6", ls="--", lw=0.7)
    a.axhline(d["recover_thr"], color=C_COM, ls=":", lw=0.9)
    a.set_xlabel("Trial Synodic Period (days)")
    if r["recovered"]:
        a.set_title(r"(%s) Round %d:  $P$=%.2f d,  $\Delta\chi^2$=%.0f"
                    % ("abc"[k], k + 1, r["best_period"], r["sig"]), fontsize=9)
    else:
        a.set_title(r"(%s) No Further Signal ($\Delta\chi^2<%.0f$)"
                    % ("abc"[k], d["recover_thr"]), fontsize=9)
ax[0].set_ylabel(r"$\chi^2_{\rm flat}-\chi^2_{\rm sine}$")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "two_moon_periodograms.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Table 2 numbers ------------------------------------------ #
print("Table 2 (blind two-moon recovery, primary first):")
print("  moon   a_rec(Rjup)   m_rec(Mearth)   i_rec(deg)")
for k, r in enumerate(done, 1):
    print(f"  {k}       {r['a_rjup']:6.1f}        {r['mass']:7.3f}        {r['inclination']:5.1f}")
print("wrote 2 figures to", FIGDIR)
