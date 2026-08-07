#!/usr/bin/env python3
"""Figure: false_positive.png  (Paper III, false-positive fraction).

Two panels, from the 50 uas production survey (9 a x 101 mass, 50 trials/cell;
data/survey_50muas_ntrials50.npz):
  (a) false-positive fraction across the (mass, semimajor-axis) grid, with the
      50% detection-fraction contour overlaid.
  (b) false-positive and detection fraction vs the input astrometric signal
      amplitude: spurious detections peak on the detection threshold and vanish
      once the signal is comfortably above the noise.

A false positive is a trial whose significance clears the chi^2 cut but whose
recovered period is wrong (period error >= 5%): the search locked onto a noise
peak. Reproducible: uses the committed survey cube + the frozen exomoonsim
snapshot for the (purely geometric) input-amplitude grid.

Run:  python make_false_positive_fig.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams
from exomoonsim.survey import input_amplitude_grid

DATA = os.path.join(HERE, "data", "survey_50muas_ntrials50.npz")
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "false_positive.png")

d = np.load(DATA, allow_pickle=True)
a, mass = d["a_grid"], d["mass_grid"]
fp, det = np.asarray(d["fpcube"], float), np.asarray(d["detfrac"], float)  # [na, nm]
prec_uas = 50.0

# input signal amplitude (uas), purely geometric
base = SimParams(astrometric_precision=prec_uas * 1e-6)
amp = np.asarray(input_amplitude_grid(base, a, mass), float) * 1000.0  # mas->uas; [na, nm]

# ---- summary numbers for the manuscript text ----
peak = np.nanmax(fp)
pj = np.unravel_index(np.nanargmax(fp), fp.shape)
noise_row = fp[:, 0]                       # mass = 0: pure-noise false-alarm rate
print(f"peak false-positive fraction = {peak:.2f} at a={a[pj[0]]:.0f} Rjup, "
      f"mass={mass[pj[1]]:.2f} Mearth (input amp {amp[pj]:.1f} uas)")
print(f"zero-signal (mass=0) false-alarm rate: mean={np.nanmean(noise_row):.3f}, "
      f"max={np.nanmax(noise_row):.3f}")
det_edge = amp[(det >= 0.4) & (det <= 0.6)]
if det_edge.size:
    print(f"input amplitude at the 50% detection edge: "
          f"{np.nanmedian(det_edge):.1f} uas (~{np.nanmedian(det_edge)/prec_uas:.1f}x precision)")

# --------------------------------------------------------------------------- #
VIR = plt.get_cmap("viridis")
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))

# (a) FP map with detection contour
A, M = np.meshgrid(a, mass, indexing="ij")
lev = np.linspace(0, max(peak, 1e-3), 11)
cf = ax[0].contourf(A, M, fp, levels=lev, cmap="viridis", extend="max")
cs = ax[0].contour(A, M, det, levels=[0.5], colors="white", linewidths=1.4)
ax[0].clabel(cs, fmt={0.5: "50% Detection"}, fontsize=7, manual=[(7.0, 0.5)])
cb = fig.colorbar(cf, ax=ax[0]); cb.set_label("False-Positive Fraction")
ax[0].set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
ax[0].set_ylabel(r"Moon Mass ($M_\oplus$)")
ax[0].set_title(r"(a) False Positives Across the Grid ($50\,\mu$as)", fontsize=10)

# (b) FP & detection vs input amplitude (binned)
af, ff, df = amp.ravel(), fp.ravel(), det.ravel()
g = np.isfinite(af) & (af > 0)
af, ff, df = af[g], ff[g], df[g]
edges = np.logspace(np.log10(af.min()), np.log10(af.max()), 16)
ctr = np.sqrt(edges[:-1] * edges[1:])
idx = np.digitize(af, edges) - 1
fpb = np.array([np.nanmean(ff[idx == k]) if np.any(idx == k) else np.nan for k in range(len(ctr))])
dtb = np.array([np.nanmean(df[idx == k]) if np.any(idx == k) else np.nan for k in range(len(ctr))])
ax[1].plot(ctr, dtb, "s-", color=VIR(0.28), lw=1.4, ms=4, label="Detection Fraction")
ax[1].plot(ctr, fpb, "o-", color=VIR(0.70), lw=1.4, ms=4, label="False-Positive Fraction")
ax[1].axvline(prec_uas, color="0.6", ls=":", lw=1.0)
ax[1].text(prec_uas * 1.05, 0.9, r"$\sigma_{\rm pos}$", color="0.5", fontsize=8)
ax[1].set_xscale("log")
ax[1].set_xlabel(r"Input Signal Amplitude ($\mu$as)")
ax[1].set_ylabel("Fraction of Trials")
ax[1].set_ylim(-0.02, 1.02)
ax[1].set_title("(b) False Positives Fade Above Threshold", fontsize=10)
ax[1].legend(frameon=False, fontsize=8, loc="center left")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
