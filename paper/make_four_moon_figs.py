#!/usr/bin/env python3
"""Four-moon demonstration figures (Paper III) + the four-moon recovery table.

Runs one blind recovery of a four-moon system around the alpha Cen A giant-planet
candidate at 10 uas over 5 yr, with all four moons on offset orbital phases:

  0.30 Mearth @ 20 Rjup,  0.20 @ 12,  0.15 @ 30,  0.25 @ 8  (R_Jup),  all i=50 deg.

Produces:
  four_moon_phasefold.png  the phase-folded detection of each of the four moons
  four_moon_summary.png    dual panel: the initial (round-1) periodogram and the
                           recovered system in the mass-semimajor-axis plane

and prints the recovered parameters (the four-moon table). The synodic-period search
is restricted to <60 d (all four moons lie below 40 d) to keep the run fast; the
cadence, precision, baseline, and host match the fiducial system (Table 1).

Reproducible: fixed seed, imports the repo's exomoonsim package. Run:
    python make_four_moon_figs.py
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
UAS = 1e6

SEED = 7
moons = [Moon(a=20, mass=0.30, inc=50, ecc=0.05),
         Moon(a=12, mass=0.20, inc=50, ecc=0.05),
         Moon(a=30, mass=0.15, inc=50, ecc=0.05),
         Moon(a=8,  mass=0.25, inc=50, ecc=0.05)]
p = SimParams(astrometric_precision=1e-5, texp=1.0, pend=60.0, ptestwidth=0.0015, moons=moons)
for mn, fr in zip(p.moons, (0.15, 0.55, 0.80, 0.35)):   # offset all orbital phases
    mn.t0 = fr * p._period_days(mn) / 365.25
d = run_trial(p, seed=SEED, return_diagnostics=True)["diag"]
done = [r for r in d["recoveries"] if r["recovered"]]

# --------------- Figure A: four phase-folded detections (2x2) -------------- #
order = sorted(range(len(done)), key=lambda k: done[k]["best_period"])
fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.4))
for axi, k in zip(ax.ravel(), order):
    r = done[k]
    ph, am = r["fold_phase"], r["fold_amp"] * UAS
    er, mo = r["fold_err"] * UAS, r["fold_model"] * UAS
    o = np.argsort(ph)
    axi.errorbar(ph[o], am[o], yerr=er[o], fmt="o", ms=2.3, color=C_OBS,
                 alpha=0.7, lw=0.5, label="Binned")
    axi.plot(ph[o], mo[o], "-", color=C_FIT, lw=1.4, label="Sine Fit")
    axi.set_ylabel(r"Amplitude ($\mu$as)")
    axi.set_title(r"$a=%.0f\,R_{\rm Jup}$,  $P_{\rm syn}=%.2f$ d" % (r["a_rjup"], r["best_period"]),
                  fontsize=9)
for axi in ax[1, :]:
    axi.set_xlabel("Phase-Folded Day")
ax[0, 0].legend(fontsize=7, loc="best")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "four_moon_phasefold.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------- Figure B: initial periodogram + recovered system ---------- #
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
# (a) round-1 periodogram (the four-tooth comb)
r0 = d["recoveries"][0]
ax[0].semilogx(r0["period_grid"], r0["periodogram"], color=C_OBS, lw=0.9)
for mk in [d["true_synodic"]] + list(d["companion_synodics"]):
    ax[0].axvline(mk, color="0.6", ls="--", lw=0.7)
ax[0].axhline(d["recover_thr"], color=C_COM, ls=":", lw=0.9)
ax[0].set_xlabel("Trial Synodic Period (days)")
ax[0].set_ylabel(r"$\chi^2_{\rm flat}-\chi^2_{\rm sine}$")
ax[0].set_title("(a) Initial Periodogram", fontsize=10)
# (b) recovered system in the mass-semimajor-axis plane
inp = list(d["input_moons"])
ia, im = zip(*inp)
ax[1].scatter(ia, im, s=95, facecolors="none", edgecolors=C_TRUE, lw=1.6, zorder=5)
for r in done:
    aa, mm = r["a_rjup"], r["mass"]
    ax[1].errorbar(aa, mm, yerr=r.get("mass_err", 0.0), fmt="o", ms=6, color=C_FIT,
                   ecolor=C_FIT, capsize=3, lw=1.2, zorder=4)
    ax[1].annotate(r"$i=%.0f^\circ$" % r["inclination"], (aa, mm), textcoords="offset points",
                   xytext=(7, 4), fontsize=7.5, color=C_FIT)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
allx = [v[0] for v in inp] + [r["a_rjup"] for r in done]
ally = [v[1] for v in inp] + [r["mass"] for r in done]
xt = [c for c in (5, 8, 10, 15, 20, 30) if min(allx) * 0.7 <= c <= max(allx) * 1.4]
yt = [c for c in (0.1, 0.15, 0.2, 0.3, 0.5) if min(ally) * 0.6 <= c <= max(ally) * 1.6]
ax[1].set_xticks(xt); ax[1].set_xticklabels(["%g" % c for c in xt])
ax[1].set_yticks(yt); ax[1].set_yticklabels(["%g" % c for c in yt])
ax[1].xaxis.set_minor_formatter(ticker.NullFormatter())
ax[1].yaxis.set_minor_formatter(ticker.NullFormatter())
ax[1].set_xlim(min(allx) * 0.8, max(allx) * 1.32)
ax[1].set_ylim(min(ally) * 0.7, max(ally) * 1.32)
ax[1].set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
ax[1].set_ylabel(r"Moon Mass ($M_\oplus$)")
ax[1].set_title("(b) Recovered System", fontsize=10)
ax[1].legend(handles=[
    Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
           markeredgecolor=C_TRUE, markersize=8, label="Input"),
    Line2D([0], [0], marker="o", ls="none", color=C_FIT, markersize=6,
           label="Recovered ($\\pm$Stat)")], fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "four_moon_summary.png"),
                                dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------- four-moon table ------------------------------------------- #
nr = [x for x in d["recoveries"] if not x["recovered"]]
print("Four-moon recovery (by round):")
print("  a_in  m_in  |  a_rec  m_rec   i_rec   P_syn(d)   dchi")
inmap = {round(a): m for a, m in inp}
for r in done:
    a = r["a_rjup"]
    print("  %4.0f  %4.2f  |  %5.1f  %5.3f  %5.1f   %7.2f   %5.0f"
          % (a, inmap.get(round(a), float("nan")), a, r["mass"], r["inclination"],
             r["best_period"], r["sig"]))
print("null-round leftover peak dchi = %.1f (excluded; at a claimed period)"
      % (nr[0]["peak_sig"] if nr else float("nan")))
print("wrote 2 figures to", FIGDIR)
