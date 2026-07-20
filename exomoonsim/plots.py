"""
Four-panel survey figure (matplotlib): significance, sidereal-period error,
amplitude error, and detection fraction, versus moon semimajor axis and mass.

Uses filled contours with colorbars -- so the detection-fraction panel is clean by
construction (no degenerate-contour artifacts to fight, unlike the IDL/Coyote
version).
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib import ticker


def _panel(ax, a, m, cube, title, *, levels, cmap, norm=None, cbar_label=None,
           logcbar=False):
    A, M = np.meshgrid(a, m, indexing="ij")
    cube = np.array(cube, float)
    if cube.shape[0] >= 2 and cube.shape[1] >= 2:
        cf = ax.contourf(A, M, cube, levels=levels, cmap=cmap, norm=norm, extend="both")
        cs = ax.contour(A, M, cube, levels=levels, colors="white", linewidths=0.4, norm=norm)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
    else:                                    # grid too small to contour
        cf = ax.pcolormesh(A, M, cube, cmap=cmap, norm=norm, shading="nearest")
    cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Moon Semimajor Axis (R$_{\\rm Jup}$)", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_survey(result, filename=None, figsize=(15, 3.4), dpi=150):
    """Render the four-panel survey figure from a SurveyResult.

    Returns the matplotlib Figure.  If ``filename`` is given, also saves it.
    """
    a = np.asarray(result.a_grid, float)
    m = np.asarray(result.mass_grid, float)
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)

    # --- significance (spans orders of magnitude -> log color scale) ---
    sig = np.clip(result.sigcube, 1e-2, None)
    smax = max(10.0, np.nanmax(sig))
    slev = np.geomspace(1.0, smax, 12)
    _panel(axes[0], a, m, sig,
           r"$\chi^2_{\nu,\rm flat}-\chi^2_{\nu,\rm sine}$",
           levels=slev, cmap="viridis", norm=LogNorm(vmin=1.0, vmax=smax),
           cbar_label="significance")

    # --- period / amplitude error (%) ---
    err_levels = [0, 1, 2, 5, 10, 25, 50, 100]
    _panel(axes[1], a, m, np.clip(result.perrcube, 0, 100),
           "Sidereal Period Error (%)", levels=err_levels, cmap="cividis_r")
    _panel(axes[2], a, m, np.clip(result.amperrcube, 0, 100),
           "Amplitude Error (%)", levels=err_levels, cmap="cividis_r")

    # --- detection fraction (clean 0..1 sequential map) ---
    dlev = np.linspace(0.0, 1.0, 11)
    _panel(axes[3], a, m, result.detfrac, "Detection Fraction",
           levels=dlev, cmap="Blues", cbar_label="fraction")

    axes[0].set_ylabel("Moon Mass (M$_\\oplus$)", fontsize=8)
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return fig
