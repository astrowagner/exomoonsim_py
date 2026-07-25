#!/usr/bin/env python3
"""Figure: two_moon_recovery.png  (Paper III, two-moon diagnostic) + Table 2 numbers.

Blind iterative-prewhitening recovery of the fiducial two-moon system
(0.3 Mearth at 20 Rjup, 0.2 Mearth at 12 Rjup, both i=50 deg) around the alpha Cen A
giant-planet candidate at 10 uas over 5 yr. Renders the full multi-panel diagnostic
via exomoonsim.plots.plot_trial and prints the recovered parameters quoted in Table 2.

Reproducible: fixed seed, imports the repo's exomoonsim package. Run:
    python make_two_moon_fig.py
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))       # repo root -> exomoonsim package
from exomoonsim.sim import SimParams, Moon, run_trial
from exomoonsim.plots import plot_trial

FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "two_moon_recovery.png")

SEED = 42
p = SimParams(astrometric_precision=1e-5,               # 10 uas, 5-yr, 1-hr cadence (defaults)
              moons=[Moon(a=20.0, mass=0.3, inc=50.0, ecc=0.05),
                     Moon(a=12.0, mass=0.2, inc=50.0, ecc=0.05)])
r = run_trial(p, seed=SEED, return_diagnostics=True)
d = r["diag"]
plot_trial(d, OUT)

print("Table 2 (blind two-moon recovery, primary first):")
print("  moon   a_rec(Rjup)   m_rec(Mearth)   i_rec(deg)")
for k, rc in enumerate([x for x in d["recoveries"] if x["recovered"]], 1):
    print(f"  {k}       {rc['a_rjup']:6.1f}        {rc['mass']:7.3f}        {rc['inclination']:5.1f}")
print("wrote", OUT)
