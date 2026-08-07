#!/usr/bin/env python3
"""Detection-fraction map at 10 uas: single sinusoid vs projected-ellipse matched filter (Paper III).

Both statistics run on the same (moon mass x separation) grid at 10 uas, with the matched-filter
threshold calibrated to the SAME pure-noise false-alarm rate as the single-sinusoid Delta chi^2 > 5
cut.  The matched filter fills in the low-mass, wide-separation corner the single sinusoid misses
(e.g. the 0.01 Mearth row, undetected with the single sinusoid, is recovered for a >~ 20 R_Jup).

Reads committed data/matched_filter_map.npz (produced by analysis/matched_filter_map.py).
Run:  python make_matched_filter_map_fig.py
"""
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
DATA = os.path.join(HERE, "data", "matched_filter_map.npz")
OUT = os.path.join(FIGDIR, "matched_filter_map.png")


def _log_edges(v):
    lv = np.log10(np.asarray(v, float)); mid = (lv[1:] + lv[:-1]) / 2.0
    return 10.0 ** np.concatenate([[2 * lv[0] - mid[0]], mid, [2 * lv[-1] - mid[-1]]])


def _draw(ax, aa, mm, det, title):
    pcm = ax.pcolormesh(_log_edges(aa), _log_edges(mm), det.T, cmap="viridis",
                        vmin=0, vmax=1, shading="flat")
    for i, av in enumerate(aa):
        for j, mv in enumerate(mm):
            ax.text(av, mv, "%.0f" % (det[i, j] * 100), ha="center", va="center",
                    fontsize=6.5, color="white" if det[i, j] < 0.55 else "black")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(aa); ax.set_xticklabels(["%g" % x for x in aa])
    ax.set_yticks(mm); ax.set_yticklabels(["%g" % x for x in mm])
    ax.set_xlabel(r"Moon Semimajor Axis ($R_{\rm Jup}$)")
    ax.set_title(title, fontsize=10)
    return pcm


if __name__ == "__main__":
    z = np.load(DATA)
    aa, mm = z["a_grid"], z["mass_grid"]
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
    pcm = _draw(ax[0], aa, mm, z["det_sine"], r"(a) Single Sinusoid ($10\,\mu$as)")
    _draw(ax[1], aa, mm, z["det_ellipse"], r"(b) Projected-Ellipse Matched Filter ($10\,\mu$as)")
    ax[0].set_ylabel(r"Moon Mass ($M_\oplus$)")
    fig.colorbar(pcm, ax=ax.tolist(), label="Detection Fraction", fraction=0.046, pad=0.02)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
