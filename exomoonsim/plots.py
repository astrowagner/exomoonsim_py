"""
Survey figure and single-trial diagnostics (matplotlib).

- ``plot_survey``  : the four-panel grid figure (significance, sidereal-period
  error, amplitude error, detection fraction) vs moon semimajor axis and mass.
- ``plot_trial``   : a nine-panel per-trial diagnostic figure (the Python analogue
  of the IDL ``exomoonsim.pdf`` doplot pages).

The survey panels use filled contours with colorbars -- so the detection-fraction
panel is clean by construction (no degenerate-contour artifacts to fight, unlike the
IDL/Coyote version).
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib import ticker

# viridis-sampled palette for the single-trial diagnostics (plot_trial) -- distinct
# but harmonious series colors instead of the default categorical ones.
_VIR = plt.get_cmap("viridis")
_C_OBS = _VIR(0.28)     # observations / data
_C_TRUE = _VIR(0.55)    # true signal / true period
_C_FIT = _VIR(0.82)     # best-fit / recovered / sine model
_C_COMP = _VIR(0.05)    # companion marker (dark, distinct)


def _fmt_3sig(v, _pos=None):
    """Colorbar tick label: 3 significant figures, no scientific notation."""
    if not np.isfinite(v) or v == 0:
        return "0"
    return f"{float(f'{v:.3g}'):g}"


def _fmt_int(v, _pos=None):
    return f"{v:.0f}"


def _panel(ax, a, m, cube, title, *, levels, cmap, norm=None, cbar_label=None,
           cbar_ticks=None, cbar_fmt=None, outline=True, outline_labels=True):
    A, M = np.meshgrid(a, m, indexing="ij")
    cube = np.array(cube, float)
    if cube.shape[0] >= 2 and cube.shape[1] >= 2:
        cf = ax.contourf(A, M, cube, levels=levels, cmap=cmap, norm=norm, extend="both")
        if outline:
            cs = ax.contour(A, M, cube, levels=levels, colors="white",
                            linewidths=0.4, norm=norm)
            if outline_labels:
                ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
    else:                                    # grid too small to contour
        cf = ax.pcolormesh(A, M, cube, cmap=cmap, norm=norm, shading="nearest")
    cb = ax.figure.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, ticks=cbar_ticks)
    if cbar_fmt is not None:
        cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(cbar_fmt))
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Moon Semimajor Axis (R$_{\\rm Jup}$)", fontsize=8)
    ax.tick_params(labelsize=7)
    return cb


def plot_survey(result, filename=None, figsize=(13.5, 6.6), dpi=150, ylog=True,
                precision_arcsec=None):
    """Render the six-panel survey figure (3 columns x 2 rows) from a SurveyResult.

    Panels: significance, sidereal-period error, amplitude error, detection fraction,
    false-positive fraction, and the input astrometric signal amplitude (the physical
    driver, with the 50% detection contour and the per-epoch precision overlaid).

    ``ylog`` (default) puts the moon-mass axis on a log scale and drops the mass=0
    control row.  ``precision_arcsec`` overrides the per-epoch precision used for the
    reference contour (default: ``result.config.base.astrometric_precision``).
    Returns the matplotlib Figure; saves it if ``filename`` is given.
    """
    from .survey import input_amplitude_grid

    a = np.asarray(result.a_grid, float)
    m = np.asarray(result.mass_grid, float)
    sigc = np.asarray(result.sigcube, float)
    perrc = np.asarray(result.perrcube, float)
    ampc = np.asarray(result.amperrcube, float)
    detc = np.asarray(result.detfrac, float)
    fpc = np.asarray(result.fpcube, float)
    signal = np.asarray(input_amplitude_grid(result.config.base, a, result.mass_grid,
                                             getattr(result.config, "companions", ())), float)
    if ylog:                                    # log mass axis -> positive masses only
        keep = m > 0
        m = m[keep]
        sigc, perrc, ampc = sigc[:, keep], perrc[:, keep], ampc[:, keep]
        detc, fpc, signal = detc[:, keep], fpc[:, keep], signal[:, keep]

    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)

    # --- significance (spans orders of magnitude -> log color scale) ---
    sig = np.clip(sigc, 1e-2, None)
    smax = max(10.0, np.nanmax(sig))
    slev = np.geomspace(1.0, smax, 12)
    sticks = [10.0 ** k for k in range(0, int(np.floor(np.log10(smax))) + 1)]
    _panel(axes[0, 0], a, m, sig,
           r"$\chi^2_{\nu,\rm flat}-\chi^2_{\nu,\rm sine}$",
           levels=slev, cmap="viridis", norm=LogNorm(vmin=1.0, vmax=smax),
           cbar_label="Significance", cbar_ticks=sticks, cbar_fmt=_fmt_3sig)

    # --- period / amplitude error (%) ---
    err_levels = [0, 1, 2, 5, 10, 25, 50, 100]
    _panel(axes[0, 1], a, m, np.clip(perrc, 0, 100),
           "Sidereal Period Error (%)", levels=err_levels, cmap="viridis")
    _panel(axes[0, 2], a, m, np.clip(ampc, 0, 100),
           "Amplitude Error (%)", levels=err_levels, cmap="viridis")

    # --- detection fraction ---
    dlev = np.linspace(0.0, 1.0, 11)
    _panel(axes[1, 0], a, m, detc, "Detection Fraction",
           levels=dlev, cmap="viridis", cbar_label="Fraction")

    # --- false-positive fraction (own [0, max] scale) ---
    fpmax = float(np.nanmax(fpc))
    fplev = (np.linspace(0.0, fpmax, 11) if np.isfinite(fpmax) and fpmax > 0
             else np.linspace(0.0, 1.0, 11))
    _panel(axes[1, 1], a, m, fpc, "False-Positive Fraction",
           levels=fplev, cmap="viridis", cbar_label="Fraction")

    # --- input signal amplitude (µas), log scale: the driver behind detection ---
    ax6 = axes[1, 2]
    sig_uas = signal * 1000.0                    # mas -> µas
    pos = sig_uas[np.isfinite(sig_uas) & (sig_uas > 0)]
    amax = float(np.nanmax(sig_uas)) if pos.size else 10.0
    amin = 1.0                                   # floor at 1 µas -> clean integer ticks
    atop = max(amax, amin * 10)
    alev = np.geomspace(amin, atop, 12)
    uticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if amin <= t <= amax]
    _panel(ax6, a, m, np.clip(sig_uas, amin, None), "Input Signal Amplitude (µas)",
           levels=alev, cmap="viridis", norm=LogNorm(vmin=amin, vmax=atop),
           cbar_label="Amplitude (µas)", cbar_ticks=uticks, cbar_fmt=_fmt_int,
           outline_labels=False)
    A, M = np.meshgrid(a, m, indexing="ij")
    _dstyle = {0.1: ":", 0.5: "--", 0.9: "-"}          # dotted / dashed / solid
    det_levels = [lv for lv in (0.1, 0.5, 0.9)
                  if np.nanmin(detc) <= lv <= np.nanmax(detc)]
    det_styles = [_dstyle[lv] for lv in det_levels]
    if det_levels:
        ax6.contour(A, M, detc, levels=det_levels, colors="black", linewidths=1.2,
                    linestyles=det_styles)
    handles = [Line2D([0], [0], color="black", ls=s, lw=1.2) for s in det_styles]
    labels = ["%d%% detection" % int(lv * 100) for lv in det_levels]
    prec = (precision_arcsec if precision_arcsec is not None
            else getattr(result.config.base, "astrometric_precision", None))
    if prec:
        pm = prec * 1e6                          # arcsec -> µas
        if amin <= pm <= amax:
            ax6.contour(A, M, sig_uas, levels=[pm], colors="magenta",
                        linestyles="--", linewidths=1.2)
            handles.append(Line2D([0], [0], color="magenta", ls="--", lw=1.2))
            labels.append(r"1$\sigma$/epoch")
    ax6.legend(handles, labels, fontsize=6, loc="lower right")

    if ylog:
        for ax in axes.flat:
            ax.set_yscale("log")
            ax.set_ylim(m.min(), m.max())
    for ax in axes.flat:
        ax.set_ylabel("Moon Mass (M$_\\oplus$)", fontsize=8)
    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return fig


def plot_trial(diag, filename=None, figsize=(14, 11), dpi=140):
    """Stacked diagnostic figure for a SINGLE trial (dynamic number of rows).

    ``diag`` is the dict from ``run_trial(..., return_diagnostics=True)["diag"]``.

    Row 1   -- on-sky track; observed separation vs time; separation vs obs number.
    Row 2   -- separation residual vs time (full; first 90 d; first 90 d with the
               recovered primary removed -- blind, so its sideband leftover remains).
    Rows 3+ -- iterative prewhitening recovery, ONE row per round: the period search
               (periodogram), the phase-folded residual + sine fit, and the leftover.
               The final (null) round shows its flat periodogram alongside two summary
               panels -- the recovered moon system in the (semimajor axis, mass) plane
               (recovered vs input), and the convergence ledger (peak chi^2 per round
               vs the recovery threshold).

    Returns the matplotlib Figure; saves it if ``filename`` is given.
    """
    # ---- pull arrays and derive separations (mas) ----
    xo, yo = np.asarray(diag["x_obs"]), np.asarray(diag["y_obs"])
    xt, yt = np.asarray(diag["x_true"]), np.asarray(diag["y_true"])
    xf, yf = np.asarray(diag["x_fit"]), np.asarray(diag["y_fit"])
    xc, yc = np.asarray(diag["x_com"]), np.asarray(diag["y_com"])
    td = np.asarray(diag["tdays"])
    rres = np.asarray(diag["rres"])
    obs_r, true_r = np.hypot(xo, yo), np.hypot(xt, yt)
    pred_r, com_r = np.hypot(xf, yf), np.hypot(xc, yc)
    MAS = 1e3
    bp = diag["best_period"]

    # smoothed residual (rolling mean, like IDL smooth())
    k = max(3, min(21, rres.size // 50))
    srres = np.convolve(rres, np.ones(k) / k, mode="same")

    # zoom windows
    zoom = (td - td[0]) <= 3.0 * bp          # ~3 moon periods
    if zoom.sum() < 5:
        zoom = np.arange(td.size) < min(td.size, 200)
    z90 = (td - td[0]) <= 90.0

    rec_list = diag.get("recoveries", [])
    thr = diag.get("recover_thr", 5.0)
    true_synods = [diag["true_synodic"]] + list(diag.get("companion_synodics", []))
    nrows = 2 + max(1, len(rec_list))
    fig, ax = plt.subplots(nrows, 3, figsize=(figsize[0], 3.3 * nrows),
                           constrained_layout=True)

    # (1) on-sky track, full
    a = ax[0, 0]
    a.plot(xt, yt, "-", color=_C_TRUE, lw=0.8, label="True Path")
    a.scatter(xo, yo, s=1.5, color=_C_OBS, alpha=0.20, label="Observed")
    a.plot(xf, yf, "-", color=_C_FIT, lw=0.8, label="Best-Fit Orbit")
    a.plot(xc, yc, ":", color="0.3", lw=0.8, label="Barycenter")
    a.set_aspect("equal", "datalim")
    a.set_xlabel("X (arcsec)"); a.set_ylabel("Y (arcsec)")
    a.set_title("On-Sky Track (Full)"); a.legend(fontsize=6, loc="best")

    # (2) observed separation vs time
    a = ax[0, 1]
    a.scatter(td, obs_r * MAS, s=1.5, color=_C_OBS, alpha=0.20, label="Observed")
    a.plot(td, true_r * MAS, "-", color=_C_TRUE, lw=0.6, label="True")
    a.plot(td, pred_r * MAS, "-", color=_C_FIT, lw=0.6, label="Fit")
    a.set_xlabel("Time (days)"); a.set_ylabel("Separation (mas)")
    a.set_title("Observed Separation"); a.legend(fontsize=6)

    # (3) observed separation vs observation number (zoom) + barycenter
    a = ax[0, 2]
    idx = np.arange(int(zoom.sum()))
    a.scatter(idx, obs_r[zoom] * MAS, s=8, color=_C_OBS, alpha=0.5, label="Observed")
    a.plot(idx, true_r[zoom] * MAS, "-", color=_C_TRUE, lw=0.8, label="True")
    a.plot(idx, pred_r[zoom] * MAS, "-", color=_C_FIT, lw=0.8, label="Fit")
    a.plot(idx, com_r[zoom] * MAS, ":", color="0.3", lw=0.8, label="Barycenter")
    a.set_xlabel("Observation Number"); a.set_ylabel("Separation (mas)")
    a.set_title("Separation (First %d d)" % int(3 * bp)); a.legend(fontsize=6)

    # (4) residual vs time, full
    a = ax[1, 0]
    prec_mas = diag.get("precision_uas", 0.0) / 1000.0
    if prec_mas:
        a.axhspan(-prec_mas, prec_mas, color="0.88", zorder=0)
    a.scatter(td, rres * MAS, s=1.5, color=_C_OBS, alpha=0.20)
    a.plot(td, srres * MAS, "-", color=_C_TRUE, lw=0.6)
    a.axhline(0, color="0.6", lw=0.5, ls="--")
    a.set_xlabel("Time (days)"); a.set_ylabel("Residual (mas)")
    a.set_title("Residual (Planet Subtracted)")

    # (5) residual vs time, first 90 days (moon signal present)
    a = ax[1, 1]
    if prec_mas:
        a.axhspan(-prec_mas, prec_mas, color="0.88", zorder=0,
                  label="±1$\\sigma$/epoch (%.0f µas)" % diag.get("precision_uas", 0.0))
    a.scatter(td[z90], rres[z90] * MAS, s=5, color=_C_OBS, alpha=0.4)
    a.plot(td[z90], srres[z90] * MAS, "-", color=_C_TRUE, lw=0.8)
    a.axhline(0, color="0.6", lw=0.5, ls="--")
    a.set_xlabel("Time (days)"); a.set_ylabel("Residual (mas)")
    a.set_title("Residual (First 90 d)"); a.legend(fontsize=6, loc="upper right")
    _ylim90 = a.get_ylim()

    # (6) residual vs time, first 90 days AFTER subtracting the recovered sine
    a = ax[1, 2]
    ra = np.asarray(diag.get("rres_after", rres))
    sra = np.convolve(ra, np.ones(k) / k, mode="same")
    if prec_mas:
        a.axhspan(-prec_mas, prec_mas, color="0.88", zorder=0)
    a.scatter(td[z90], ra[z90] * MAS, s=5, color=_C_OBS, alpha=0.4)
    a.plot(td[z90], sra[z90] * MAS, "-", color=_C_TRUE, lw=0.8)
    a.axhline(0, color="0.6", lw=0.5, ls="--")
    a.set_ylim(_ylim90)
    a.set_xlabel("Time (days)"); a.set_ylabel("Residual (mas)")
    a.set_title("Residual (First 90 d, Primary Removed)")

    # ===== rows 2+: iterative recovery, one row per prewhitening round =====
    for ri, rc in enumerate(rec_list):
        pa = ax[2 + ri, 0]
        pa.plot(rc["period_grid"], rc["periodogram"], "-", color="0.3", lw=0.8)
        pa.axhline(thr, color=_C_COMP, ls=":", lw=0.8)
        for ts in true_synods:                       # true signals, for reference
            pa.axvline(ts, color=_C_TRUE, ls="--", lw=0.6, alpha=0.5)
        pa.set_xscale("log")
        pa.set_xlabel("Trial Synodic Period (days)")
        pa.set_ylabel(r"$\chi^2_{\rm flat}-\chi^2_{\rm sine}$")

        if rc["recovered"]:
            pa.axvline(rc["best_period"], color=_C_FIT, lw=1.2)
            pa.set_title(r"Recovery %d:  $P$=%.2f d,  $\chi^2$=%.0f"
                         % (ri + 1, rc["sidereal"], rc["sig"]))
            ph2 = np.asarray(rc["fold_phase"]); o2 = np.argsort(ph2)
            A2 = np.asarray(rc["fold_amp"]); E2 = np.asarray(rc["fold_err"])
            M2 = np.asarray(rc["fold_model"])
            fx = ax[2 + ri, 1]
            fx.errorbar(ph2, A2 * MAS, yerr=E2 * MAS, fmt="o", ms=3, color=_C_OBS,
                        alpha=0.7, lw=0.6, label="Binned")
            fx.plot(ph2[o2], M2[o2] * MAS, "-", color=_C_FIT, lw=1.5, label="Sine Fit")
            fx.axhline(0, color="0.6", lw=0.5, ls="--")
            fx.set_xlabel("Phase-Folded Day"); fx.set_ylabel("Amplitude (mas)")
            fx.set_title("Phase-Folded at %.2f d" % rc["best_period"]); fx.legend(fontsize=6)
            mx = ax[2 + ri, 2]
            mx.errorbar(ph2, (A2 - M2) * MAS, yerr=E2 * MAS, fmt="o", ms=3, color=_C_OBS,
                        alpha=0.7, lw=0.6)
            mx.axhline(0, color="0.6", lw=0.5, ls="--")
            mx.set_xlabel("Phase-Folded Day"); mx.set_ylabel("Residual (mas)")
            mx.set_title("Phase-Fold Minus Sine")
        else:
            pa.set_title(r"No Further Signal ($\chi^2$ < %.0f)" % thr)
            # ---- summary A: recovered system in (a, mass), with mass uncertainties ----
            sa = ax[2 + ri, 1]
            inp = diag.get("input_moons", [])
            recs = [x for x in rec_list if x["recovered"]]
            if inp:
                ia_, im_ = zip(*inp)
                sa.scatter(ia_, im_, s=90, facecolors="none", edgecolors=_C_TRUE,
                           lw=1.5, zorder=5)
            for x in recs:
                aa, mm = x["a_rjup"], x["mass"]
                sa.errorbar(aa, mm, yerr=x.get("mass_err", 0.0), fmt="o", ms=6,
                            color=_C_FIT, ecolor=_C_FIT, capsize=3, lw=1.2, zorder=4)
                if "inclination" in x:
                    sa.annotate("i=%.0f°" % x["inclination"], (aa, mm),
                                textcoords="offset points", xytext=(7, 3),
                                fontsize=6, color=_C_FIT)
            sa.set_xscale("log"); sa.set_yscale("log")
            allx = [v[0] for v in inp] + [x["a_rjup"] for x in recs]
            ally = [v[1] for v in inp] + [x["mass"] for x in recs]
            if allx and ally:
                xt = [c for c in (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100)
                      if min(allx) * 0.7 <= c <= max(allx) * 1.4]
                yt = [c for c in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5)
                      if min(ally) * 0.6 <= c <= max(ally) * 1.6]
                sa.set_xticks(xt); sa.set_xticklabels(["%g" % c for c in xt])
                sa.set_yticks(yt); sa.set_yticklabels(["%g" % c for c in yt])
                sa.xaxis.set_minor_formatter(ticker.NullFormatter())
                sa.yaxis.set_minor_formatter(ticker.NullFormatter())
            sa.set_xlabel("Moon Semimajor Axis (R$_{\\rm Jup}$)")
            sa.set_ylabel("Moon Mass (M$_\\oplus$)")
            sa.set_title("Recovered System ($i$ Labeled)")
            sa.legend(handles=[
                Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
                       markeredgecolor=_C_TRUE, markersize=8, label="Input"),
                Line2D([0], [0], marker="o", ls="none", color=_C_FIT, markersize=6,
                       label="Recovered ±Stat")],
                fontsize=6, loc="best")
            # ---- summary B: 90-day residual, FULL recovered system removed ----
            sb = ax[2 + ri, 2]
            rall = np.asarray(diag.get("rres_allremoved", rres))
            srall = np.convolve(rall, np.ones(k) / k, mode="same")
            if prec_mas:
                sb.axhspan(-prec_mas, prec_mas, color="0.88", zorder=0,
                           label="±1$\\sigma$/epoch")
            sb.scatter(td[z90], rall[z90] * MAS, s=5, color=_C_OBS, alpha=0.4)
            sb.plot(td[z90], srall[z90] * MAS, "-", color=_C_TRUE, lw=0.8)
            sb.axhline(0, color="0.6", lw=0.5, ls="--")
            sb.set_ylim(_ylim90)
            sb.set_xlabel("Time (days)"); sb.set_ylabel("Residual (mas)")
            sb.set_title("Residual (First 90 d, All Recovered Removed)")
            sb.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        "Single Trial   |   %d Moon(s) In, %d Recovered   |   %.0f µas/Epoch   |   "
        "Primary %.2f d (Rec %.2f d, %.2f%%),  Amp %.1f/%.1f µas"
        % (len(diag.get("input_moons", [])),
           sum(1 for x in rec_list if x["recovered"]),
           diag.get("precision_uas", float("nan")),
           diag["moon_period"], diag["sidereal_period"], diag["perr"],
           diag["retrieved_amp"] * 1000, diag["input_amp"] * 1000),
        fontsize=11)

    if filename:
        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    return fig
