#!/usr/bin/env python3
"""Figures 2-5 (Paper III): the blind two-moon recovery, split by plot type.

Runs one fiducial two-moon trial (0.3 Mearth at 20 Rjup, 0.2 Mearth at 12 Rjup,
both i=50 deg) around the alpha Cen A giant-planet candidate at 10 uas over 5 yr,
then draws four focused figures from the diagnostics dict:

  two_moon_context.png       on-sky track + observed separation
  two_moon_residuals.png     90-day separation residual, cleaned round by round
  two_moon_periodograms.png  the prewhitening period search (2 rounds + null)
  two_moon_phasefold.png     the phase-folded detection of each moon (fold + residual)
  two_moon_system.png        the recovered moons in the mass-semimajor-axis plane

Also prints the recovered parameters quoted in Table 2. Reproducible: fixed seed,
imports the repo's exomoonsim package. Run:  python make_two_moon_figs.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

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

# ---------------- Figure 2: context (track + separation) ------------------- #
fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.0))
a0 = ax[0]
a0.plot(d["x_true"], d["y_true"], "-", color=C_TRUE, lw=0.8, label="True Path")
a0.scatter(d["x_obs"], d["y_obs"], s=1.5, color=C_OBS, alpha=0.20, label="Observed")
a0.plot(d["x_fit"], d["y_fit"], "-", color=C_FIT, lw=0.8, label="Best-Fit Orbit")
a0.plot(d["x_com"], d["y_com"], ":", color="0.3", lw=0.8, label="Barycenter")
a0.set_aspect("equal"); a0.set_xlabel("X (arcsec)"); a0.set_ylabel("Y (arcsec)")
a0.set_title("(a) On-Sky Track", fontsize=10); a0.legend(fontsize=7, loc="best")
a1 = ax[1]
obs_r = np.hypot(d["x_obs"], d["y_obs"]) * MAS
true_r = np.hypot(d["x_true"], d["y_true"]) * MAS
fit_r = np.hypot(d["x_fit"], d["y_fit"]) * MAS
a1.scatter(td, obs_r, s=1.5, color=C_OBS, alpha=0.20, label="Observed")
a1.plot(td, true_r, "-", color=C_TRUE, lw=0.6, label="True")
a1.plot(td, fit_r, "-", color=C_FIT, lw=0.6, label="Fit")
a1.set_xlabel("Time (days)"); a1.set_ylabel("Separation (mas)")
a1.set_title("(b) Observed Separation", fontsize=10); a1.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "two_moon_context.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 3: 90-day residual progression -------------------- #
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

# ---------------- Figure 5: phase-folded detections ------------------------ #
fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.0))
for row, r in enumerate(done[:2]):
    ph, am, er, mo = r["fold_phase"], r["fold_amp"] * UAS, r["fold_err"] * UAS, r["fold_model"] * UAS
    o = np.argsort(ph)
    fa = ax[row, 0]
    fa.errorbar(ph[o], am[o], yerr=er[o], fmt="o", ms=2.5, color=C_OBS,
                alpha=0.7, lw=0.6, label="Binned")
    fa.plot(ph[o], mo[o], "-", color=C_FIT, lw=1.5, label="Sine Fit")
    fa.set_ylabel(r"Amplitude ($\mu$as)")
    fa.set_title(r"Moon %d: Phase-Folded at %.2f d" % (row + 1, r["best_period"]), fontsize=9)
    fa.legend(fontsize=7)
    fb = ax[row, 1]
    fb.axhline(0, color="0.6", ls=":", lw=0.8)
    fb.errorbar(ph[o], am[o] - mo[o], yerr=er[o], fmt="o", ms=2.5, color=C_OBS, alpha=0.7, lw=0.6)
    fb.set_ylabel(r"Residual ($\mu$as)")
    fb.set_title("Moon %d: Phase-Fold Minus Sine" % (row + 1), fontsize=9)
for a in ax[1, :]:
    a.set_xlabel("Phase-Folded Day")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "two_moon_phasefold.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 6: recovered system (mass-semimajor-axis plane) --- #
fig, a = plt.subplots(figsize=(5.4, 4.4))
inp = list(d["input_moons"])
if inp:
    ia, im = zip(*inp)
    a.scatter(ia, im, s=95, facecolors="none", edgecolors=C_TRUE, lw=1.6, zorder=5)
for r in done:
    aa, mm = r["a_rjup"], r["mass"]
    a.errorbar(aa, mm, yerr=r.get("mass_err", 0.0), fmt="o", ms=6, color=C_FIT,
               ecolor=C_FIT, capsize=3, lw=1.2, zorder=4)
    a.annotate(r"$i=%.0f^\circ$" % r["inclination"], (aa, mm), textcoords="offset points",
               xytext=(8, 4), fontsize=8, color=C_FIT)
a.set_xscale("log"); a.set_yscale("log")
allx = [v[0] for v in inp] + [r["a_rjup"] for r in done]
ally = [v[1] for v in inp] + [r["mass"] for r in done]
xt = [c for c in (5, 7, 10, 15, 20, 30) if min(allx) * 0.7 <= c <= max(allx) * 1.4]
yt = [c for c in (0.1, 0.15, 0.2, 0.3, 0.5) if min(ally) * 0.6 <= c <= max(ally) * 1.6]
a.set_xticks(xt); a.set_xticklabels(["%g" % c for c in xt])
a.set_yticks(yt); a.set_yticklabels(["%g" % c for c in yt])
a.xaxis.set_minor_formatter(ticker.NullFormatter())
a.yaxis.set_minor_formatter(ticker.NullFormatter())
a.set_xlim(min(allx) * 0.8, max(allx) * 1.30)          # room for the i-labels
a.set_ylim(min(ally) * 0.7, max(ally) * 1.30)
a.set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
a.set_ylabel(r"Moon Mass ($M_\oplus$)")
a.legend(handles=[
    Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
           markeredgecolor=C_TRUE, markersize=8, label="Input"),
    Line2D([0], [0], marker="o", ls="none", color=C_FIT, markersize=6,
           label="Recovered ($\\pm$Stat)")], fontsize=8, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "two_moon_system.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------- Table 2 numbers ------------------------------------------ #
print("Table 2 (blind two-moon recovery, primary first):")
print("  moon   a_rec(Rjup)   m_rec(Mearth)   i_rec(deg)")
for k, r in enumerate(done, 1):
    print(f"  {k}       {r['a_rjup']:6.1f}        {r['mass']:7.3f}        {r['inclination']:5.1f}")
print("wrote 5 figures to", FIGDIR)
