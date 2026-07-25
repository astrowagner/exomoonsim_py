#!/usr/bin/env python3
"""Figure: fp_vs_threshold.png  -- false positives as a function of the Delta chi^2
detection cut.

Post-processes the committed per-trial cube (data/survey_50muas_ntrials50.npz;
trials = [na, nm, ntrials, (sig, perr, amperr)]) by sweeping the significance
threshold c and recomputing, at each c:
  (a) the pure-noise false-alarm rate P(sig>c) on the zero-mass (no-signal) control,
      and the grid-averaged false-positive fraction (fires with a wrong period,
      perr>=5%);
  (b) the reliability (purity) of claimed detections, TP/(TP+FP), and the
      completeness retained relative to the fiducial c=5 cut.

Nothing is re-simulated: this is exact reprocessing of the survey trials, so it is
fully reproducible. Run:  python make_fp_vs_threshold.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "survey_50muas_ntrials50.npz")
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "fp_vs_threshold.png")

d = np.load(DATA, allow_pickle=True)
mass = np.asarray(d["mass_grid"], float)
tr = np.asarray(d["trials"], float)            # [na, nm, ntrials, 3]
perr_thr = float(d["perr_thr"])                # 5% period tolerance
c_fid = float(d["chsq_thr"])                   # fiducial Delta chi^2 = 5
sig, perr = tr[..., 0], tr[..., 1]

# zero-mass (no-signal) control trials: pure noise
j0 = int(np.argmin(np.abs(mass)))              # mass = 0 column
sig_noise = sig[:, j0, :].ravel()              # [na*ntrials]
sig_noise = sig_noise[np.isfinite(sig_noise)]

# signal-present cells (mass > 0)
sig_s = sig[:, mass > 0, :]
perr_s = perr[:, mass > 0, :]
finite = np.isfinite(sig_s) & np.isfinite(perr_s)

cuts = np.linspace(1.0, 40.0, 200)
noise_far = np.array([np.mean(sig_noise > c) for c in cuts])
# grid false-positive fraction: mean over ALL cells of (sig>c & wrong period)
fpg = np.array([np.mean((sig > c) & (perr >= perr_thr) & np.isfinite(perr)) for c in cuts])
# reliability & completeness among signal-present cells
tp = np.array([np.sum(finite & (sig_s > c) & (perr_s < perr_thr)) for c in cuts], float)
fp = np.array([np.sum(finite & (sig_s > c) & (perr_s >= perr_thr)) for c in cuts], float)
with np.errstate(invalid="ignore", divide="ignore"):
    reliability = tp / (tp + fp)
tp5 = np.sum(finite & (sig_s > c_fid) & (perr_s < perr_thr))
completeness = tp / tp5                          # retained true detections relative to c=5

def at(c):
    k = int(np.argmin(np.abs(cuts - c)))
    return noise_far[k], fpg[k], reliability[k], completeness[k]

print(f"perr tolerance = {perr_thr:.0f}%   fiducial cut c = {c_fid:.0f}")
print(" cut   noiseFAR   gridFP    purity   completeness(rel c=5)")
for c in (3, 5, 8, 12, 20, 30):
    nf, fg, rl, cp = at(c)
    print(f"{c:4.0f}   {nf:7.3f}   {fg:6.3f}   {rl:6.3f}   {cp:7.3f}")
# cut where noise false-alarm rate drops below 1%
below = cuts[noise_far < 0.01]
if below.size:
    print(f"noise false-alarm rate < 1% for c > {below.min():.1f}")

# --------------------------------------------------------------------------- #
VIR = plt.get_cmap("viridis")
C1, C2, C3 = VIR(0.28), VIR(0.70), VIR(0.50)
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.2))

# (a) false-positive rates vs cut (log-y)
ax[0].semilogy(cuts, np.clip(noise_far, 1e-4, None), color=C2, lw=1.6,
               label="Noise False-Alarm Rate (No Signal)")
ax[0].semilogy(cuts, np.clip(fpg, 1e-4, None), color=C1, lw=1.6,
               label="Grid False-Positive Fraction")
ax[0].axvline(c_fid, color="0.6", ls=":", lw=1.0)
ax[0].text(c_fid * 1.05, 0.5, r"$\Delta\chi^2=5$", color="0.5", fontsize=8, rotation=90, va="top")
ax[0].set_xlabel(r"Detection Cut  $\Delta\chi^2 > c$")
ax[0].set_ylabel("False-Positive Fraction")
ax[0].set_ylim(3e-4, 1.0)
ax[0].set_title("(a) False Positives vs Selection", fontsize=10)
ax[0].legend(frameon=False, fontsize=8, loc="upper right")

# (b) reliability & completeness vs cut
ax[1].plot(cuts, reliability, color=C1, lw=1.6, label="Reliability  TP/(TP+FP)")
ax[1].plot(cuts, completeness, color=C3, lw=1.6, ls="--",
           label=r"Completeness (Rel. $\Delta\chi^2=5$)")
ax[1].axvline(c_fid, color="0.6", ls=":", lw=1.0)
ax[1].set_xlabel(r"Detection Cut  $\Delta\chi^2 > c$")
ax[1].set_ylabel("Fraction")
ax[1].set_ylim(0.0, 1.02)
ax[1].set_title("(b) Reliability vs Completeness Trade-Off", fontsize=10)
ax[1].legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT)
