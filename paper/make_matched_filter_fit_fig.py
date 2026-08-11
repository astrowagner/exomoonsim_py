#!/usr/bin/env python3
"""Fig. 2: projected-ellipse matched filter vs single sinusoid, for two moon periods (Paper III).

Thin wrapper around analysis/matched_filter_fit_demo.py that writes the figure into paper/figs/.
Runs the simulation directly (fixed seed), so it reproduces the published figure exactly.

Run:  python make_matched_filter_fit_fig.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from analysis.matched_filter_fit_demo import build_figure

FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "matched_filter_fit.png")

if __name__ == "__main__":
    r1, r2 = build_figure(OUT)
    print("wrote", OUT)
    print("a=20 R_J : sine RMS=%.1f  ellipse RMS=%.1f  (gain %.2fx)" % (r1[0], r1[1], r1[0] / r1[1]))
    print("a=50 R_J : sine RMS=%.1f  ellipse RMS=%.1f  (gain %.2fx)" % (r2[0], r2[1], r2[0] / r2[1]))
