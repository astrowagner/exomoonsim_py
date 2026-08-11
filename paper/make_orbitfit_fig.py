#!/usr/bin/env python3
"""Fig: old (rho,theta) vs new Cartesian planet-orbit fit, and the residual improvement (Paper III).

Left  -- mechanism: the legacy fit minimizes chi^2 in (separation, position-angle) with a CONSTANT
position-angle error (sigma_theta = 0.1 deg in exomoonsim.pro, ~50x the correctly propagated
sigma/rho), which nearly removes the tangential constraint and, with a hand-rolled Newton-Raphson,
leaves orbit residual behind.  The new fit minimizes the Cartesian (dx, dy) residual -- the space
in which the astrometric noise is actually Gaussian -- with a robust Levenberg-Marquardt solver.

Right -- consequence, from the committed IDL vs Python survey cubes (9 a x 101 mass, 50 uas, 50
trials/cell): where both codes detect, the Python detection significance is a uniform ~1.50x higher,
i.e. the cleaner residual is a ~1.22x lower effective noise floor at the same signal.

Run:  python make_orbitfit_fig.py
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "validation"))
from compare_maps import parse_idl_dump

FIGDIR = os.path.join(HERE, "figs"); os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "orbit_fit.png")
IDL = os.path.join(HERE, "data", "idl_cubes_dump.txt")
NPZ = os.path.join(HERE, "data", "survey_50muas_ntrials50.npz")

VIR = plt.get_cmap("viridis")
C_NEW, C_OLD, C_TAN = VIR(0.12), VIR(0.55), "crimson"


def _arrow(ax, p0, p1, color, lw=2.2, ls="-", alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, ls=ls, alpha=alpha, shrinkA=0, shrinkB=0))


def panel_mechanism(ax):
    # local zoom near the planet: star lies off to the LEFT (radial = horizontal), so the
    # observed point is offset from the model by a small radial (Delta-rho) and a larger
    # tangential (rho*Delta-theta) misfit.  Offsets exaggerated for clarity.
    P = np.array([0.0, 0.0])                 # planet (model) position
    Q = np.array([0.22, 0.0])                # after the radial step (Delta-rho, along star line)
    O = np.array([0.22, 0.60])              # observed = P + radial + tangential

    # direction to the star (radial), drawn as a dotted arrow leaving the frame
    _arrow(ax, np.array([-0.15, 0.0]), np.array([-0.72, 0.0]), "0.6", lw=1.2, ls=(0, (2, 2)))
    ax.annotate(r"to star $\bigstar$", (-0.72, 0.0), textcoords="offset points", xytext=(2, 8),
                fontsize=9, color="0.45")

    ax.plot(*P, "o", color="0.25", ms=9, zorder=5)
    ax.annotate("planet\n(model)", P, textcoords="offset points", xytext=(-40, -6), fontsize=9,
                color="0.25", ha="center")
    ax.plot(*O, "o", mfc="white", mec="0.25", mew=1.6, ms=9, zorder=5)
    ax.annotate("observed", O, textcoords="offset points", xytext=(9, -11), fontsize=9, color="0.25")

    # OLD: radial (kept) + tangential (down-weighted) decomposition
    _arrow(ax, P, Q, C_OLD, lw=2.6)
    _arrow(ax, Q, O, C_TAN, lw=2.2, ls=(0, (4, 3)), alpha=0.95)
    ax.annotate(r"$\Delta\rho$", (Q + P) / 2, textcoords="offset points", xytext=(0, -16),
                fontsize=10, color=C_OLD, ha="center")
    ax.annotate(r"$\rho\,\Delta\theta$" + "\n" + r"($\sigma_\theta{=}0.1^\circ$," + "\ndown-weighted)",
                Q + 0.5 * (O - Q), textcoords="offset points", xytext=(12, -4), fontsize=8.5,
                color=C_TAN, va="center")
    # NEW: Cartesian residual P->O
    _arrow(ax, P, O, C_NEW, lw=2.8)
    ax.annotate(r"$(\Delta x,\Delta y)$", (P + O) / 2, textcoords="offset points", xytext=(-52, 2),
                fontsize=10, color=C_NEW, ha="center")

    ax.annotate(r"new fit: minimize $(\Delta x,\Delta y)$  —  Cartesian",
                (0.02, 0.95), xycoords="axes fraction", color=C_NEW, fontsize=9.5)
    ax.annotate(r"old fit: minimize $(\Delta\rho/\sigma_\rho)^2+(\Delta\theta/\sigma_\theta)^2$  —  $(\rho,\theta)$",
                (0.02, 0.88), xycoords="axes fraction", color=C_OLD, fontsize=9.5)

    ax.set_xlim(-0.85, 0.95); ax.set_ylim(-0.35, 0.92); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) Orbit-fit residual: Cartesian vs. Separation/Position-Angle", fontsize=10)


def panel_data(ax):
    idl = parse_idl_dump(IDL); z = np.load(NPZ)
    si, sp = idl["sigcube"], z["sigcube"]
    both = (si > 5) & (sp > 5)
    x, y = si[both], sp[both]
    r = np.median(y / x)
    ax.scatter(x, y, s=10, color=VIR(0.45), alpha=0.45, edgecolors="none", zorder=2,
               label="Survey cells (both detect)")
    lo, hi = 4.0, 1100.0
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1.3, zorder=1, label="1:1 (equal)")
    ax.plot([lo, hi], [r * lo, r * hi], "-", color=C_NEW, lw=1.8, zorder=3,
            label=r"%.2f:1 (median)" % r)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"IDL Detection Significance $\Delta\chi^2$ (old fit)")
    ax.set_ylabel(r"Python $\Delta\chi^2$ (new fit)")
    ax.axhline(5, color="0.8", lw=0.8, zorder=0); ax.axvline(5, color="0.8", lw=0.8, zorder=0)
    ax.set_title(r"(b) $50\,\mu$as survey: new fit is a uniform $\sim\!1.5\times$ more significant",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.97, 0.05, r"$\Rightarrow$ residual noise $\sqrt{%.2f}\approx%.2f\times$ lower"
            % (r, np.sqrt(r)), transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color=C_NEW)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0))
    panel_mechanism(ax[0])
    panel_data(ax[1])
    ax[1].grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
