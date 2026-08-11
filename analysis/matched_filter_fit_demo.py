#!/usr/bin/env python3
"""What the projected-ellipse matched filter fits, vs the single sinusoid (for Sumin).

Single sinusoid (the default search): fits  A cos(2*pi t/P_syn - varphi)  -- two free
coefficients at ONE frequency (the synodic carrier), i.e. a CONSTANT-amplitude tone.

Matched filter (the ellipse): fits, at the sidereal frequency w = 2*pi/P_sid,

    rho(t) = a_x cos(w t)cos(phi) + a_y cos(w t)sin(phi)
           + b_x sin(w t)cos(phi) + b_y sin(w t)sin(phi) + const,

with phi(t) the KNOWN planet position angle.  This is the moon's 2-D reflex projected onto the
star->planet direction.  The observable is the RADIAL separation residual, rho(t) = w(t) . rhat(t).
Because the planet's projected orbit is inclined and eccentric, its on-sky angle phi(t) sweeps
NON-uniformly, so the projection modulates the moon's amplitude and phase over the planet period.
In frequency space the moon's power is therefore spread into a COMB -- the sidereal carrier plus
planet-motion sidebands at w_s +- k w_p -- rather than a single clean tone.  The single sinusoid
fits only the tallest line of that comb (the synodic carrier) and discards the rest; the matched
filter uses the known phi(t) to fit the whole comb coherently in one linear model, so it recovers
the amplitude the sinusoid leaves behind (here RMS ~10 uas ~ the noise floor, vs ~25 for the sine).

Two rows compare a nominal moon (a=20 R_Jup, P_syn~21 d) with a wider, ~4x-longer-period moon
(a=50 R_Jup, P_syn~87 d), both at i=60 deg.  Each row has three views: (left) the time series
with the single-sinusoid and matched-filter fits; (middle) the per-cycle amplitude vs the
matched-filter envelope and the flat sinusoid; (right) the synodic phase-fold, where the sideband
power shows up as the vertical scatter the ellipse reproduces and the sinusoid discards.  The
matched-filter advantage GROWS for the wider moon (sine RMS ~55 vs ellipse ~11 uas, a ~5x gap,
versus ~2.4x at a=20): a longer-period moon sweeps more planet motion per orbit, so more of its
power leaks into the sidebands.  (Each model is fit at its true period; the blind single-sinusoid
search additionally aliases for the wide moon, which only widens the real-world gap.)

Run:  python analysis/matched_filter_fit_demo.py
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams, run_trial

OUT = os.path.join(HERE, "matched_filter_fit_demo.png")
UAS = 1e6


def _refine(objfun, p0, frac=0.01, n=401):
    grid = p0 * np.linspace(1 - frac, 1 + frac, n)
    vals = [objfun(p) for p in grid]
    return grid[int(np.argmax(vals))]


def fit_sine(td, rres, psyn):
    """Least-squares constant-amplitude sinusoid; period refined around psyn."""
    def varred(P):
        w = 2 * np.pi / P
        G = np.column_stack([np.cos(w * td), np.sin(w * td), np.ones(td.size)])
        c, *_ = np.linalg.lstsq(G, rres, rcond=None)
        return -np.sum((rres - G @ c) ** 2)
    P = _refine(varred, psyn)
    w = 2 * np.pi / P
    G = np.column_stack([np.cos(w * td), np.sin(w * td), np.ones(td.size)])
    c = np.linalg.lstsq(G, rres, rcond=None)[0]
    model = G @ c
    amp = np.hypot(c[0], c[1])
    rms = np.sqrt(np.mean((rres - model) ** 2))
    return model, amp, rms, P, c


def fit_ellipse(td, rres, phi, psid):
    """Least-squares projected ellipse; sidereal period refined around psid."""
    def basis(P):
        w = 2 * np.pi / P
        return np.column_stack([np.cos(w * td) * np.cos(phi), np.cos(w * td) * np.sin(phi),
                                np.sin(w * td) * np.cos(phi), np.sin(w * td) * np.sin(phi),
                                np.ones(td.size)])
    def varred(P):
        G = basis(P)
        c, *_ = np.linalg.lstsq(G, rres, rcond=None)
        return -np.sum((rres - G @ c) ** 2)
    P = _refine(varred, psid)
    G = basis(P)
    c = np.linalg.lstsq(G, rres, rcond=None)[0]
    model = G @ c
    ax_, ay_, bx_, by_ = c[:4]
    # local amplitude envelope sqrt(C^2 + S^2), C = a_x cosphi + a_y sinphi, S = b_x cosphi + b_y sinphi
    C = ax_ * np.cos(phi) + ay_ * np.sin(phi)
    S = bx_ * np.cos(phi) + by_ * np.sin(phi)
    env = np.hypot(C, S)
    rms = np.sqrt(np.mean((rres - model) ** 2))
    return model, env, rms


def per_cycle_amp(td, rres, psyn):
    """Local sinusoid amplitude per synodic cycle (noise-unbiased data envelope points).

    Within each one-cycle window fit [cos, sin, const] at the carrier and take sqrt(c0^2+c1^2).
    Unlike (max-min)/2 this is not inflated by the ~3 sigma extremes of the in-cycle noise.
    """
    w = 2 * np.pi / psyn
    edges = np.arange(td.min(), td.max() + psyn, psyn)
    # keep only well-sampled cycles: a fully in-season cycle has a few hundred 1-hr samples;
    # cycles straddling the annual gap are sparse and give unreliable local amplitudes.
    nmin = 0.6 * (psyn * 24.0)                   # >~60% of one cycle's hourly samples
    tc, ac = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (td >= lo) & (td < hi)
        if m.sum() >= nmin:
            t = td[m]
            G = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones(t.size)])
            c = np.linalg.lstsq(G, rres[m], rcond=None)[0]
            tc.append(0.5 * (lo + hi)); ac.append(np.hypot(c[0], c[1]))
    return np.array(tc), np.array(ac)


def load(inc, moon_a=20.0):
    p = SimParams(moon_a=moon_a, moon_mass=1.0, moon_inc=inc, moon_ecc=0.0,
                  astrometric_precision=1e-5, pend=60.0, ptestwidth=0.002)
    d = run_trial(p, seed=3, return_diagnostics=True, recover_thr=1e12)["diag"]
    d["moon_a"] = moon_a
    return d


def draw_row(axr, VIR, inc, moon_a, tags):
    """Draw the three-view comparison for one moon semimajor axis onto the axes row ``axr``."""
    c_data, c_sine, c_ell = VIR(0.62), "crimson", VIR(0.12)
    d = load(inc, moon_a)
    td, rres = d["tdays"], np.asarray(d["rres"])
    phi = np.arctan2(d["y_fit"], d["x_fit"])
    # anchor each model at its true period (the blind search aliases for wide, long-period moons):
    # sine at the synodic carrier, ellipse at the moon's orbital (sidereal) period.
    ms, amps, rs, Psyn, csine = fit_sine(td, rres, d["true_synodic"])
    me, enve, re = fit_ellipse(td, rres, phi, d["moon_period"])
    ta, tb, tc = tags
    lbl = r"$a=%g\,R_{\rm J}$ ($P_{\rm syn}=%.0f$ d)" % (moon_a, Psyn)

    # (a) time series over the first observing season (before the annual gap)
    tmax = 300.0
    seg = td < tmax; o = np.argsort(td[seg])
    axr[0].scatter(td[seg][o], rres[seg][o] * UAS, s=5, color=c_data, alpha=0.30, label="Separation residual", zorder=1)
    axr[0].plot(td[seg][o], ms[seg][o] * UAS, "--", color=c_sine, lw=1.3,
                label=r"Single sinusoid (RMS %.1f)" % (rs * UAS), zorder=2)
    axr[0].plot(td[seg][o], me[seg][o] * UAS, "-", color=c_ell, lw=1.3,
                label=r"Matched filter (RMS %.1f)" % (re * UAS), zorder=3)
    axr[0].set_xlabel("Time (days)"); axr[0].set_ylabel(r"Separation Residual ($\mu$as)")
    axr[0].set_title(r"(%s) %s: time series" % (ta, lbl), fontsize=10)
    axr[0].legend(frameon=False, fontsize=8, loc="upper right")

    # (b) amplitude envelope over the full baseline
    tcyc, acyc = per_cycle_amp(td, rres, Psyn)
    oo = np.argsort(td)
    axr[1].scatter(tcyc, acyc * UAS, s=15, color=c_data, alpha=0.7, label="Per-cycle amplitude (data)", zorder=2)
    axr[1].axhline(amps * UAS, ls="--", color=c_sine, lw=1.6, label="Single sinusoid (constant)", zorder=1)
    axr[1].plot(td[oo], enve[oo] * UAS, "-", color=c_ell, lw=1.8, label="Matched-filter envelope", zorder=3)
    axr[1].set_xlabel("Time (days)"); axr[1].set_ylabel(r"Signal Amplitude ($\mu$as)")
    axr[1].set_title(r"(%s) %s: amplitude beat" % (tb, lbl), fontsize=10)
    axr[1].legend(frameon=False, fontsize=8, loc="lower right")
    lo = min(enve.min(), acyc.min()) * UAS; hi = max(enve.max(), acyc.max()) * UAS
    axr[1].set_ylim(lo - 0.28 * (hi - lo), hi + 0.10 * (hi - lo))

    # (c) phase-fold at the synodic period
    ph = (td % Psyn) / Psyn
    axr[2].scatter(ph, rres * UAS, s=4, color=c_data, alpha=0.12, zorder=1, label="Folded data")
    axr[2].scatter(ph, me * UAS, s=4, color=c_ell, alpha=0.18, zorder=2, label="Matched filter (folded)")
    pg = np.linspace(0, 1, 300)
    sine_curve = (csine[0] * np.cos(2 * np.pi * pg) + csine[1] * np.sin(2 * np.pi * pg) + csine[2]) * UAS
    axr[2].plot(pg, sine_curve, "--", color=c_sine, lw=1.8, zorder=3, label="Single sinusoid")
    nb = 24; edges = np.linspace(0, 1, nb + 1); cen = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(ph, edges) - 1, 0, nb - 1)
    mean = np.array([rres[idx == k].mean() * UAS if np.any(idx == k) else np.nan for k in range(nb)])
    axr[2].plot(cen, mean, "o", color="white", mec=VIR(0.35), mew=1.2, ms=5, zorder=4)
    axr[2].set_xlabel("Synodic Phase"); axr[2].set_ylabel(r"Separation Residual ($\mu$as)")
    axr[2].set_title(r"(%s) %s: folded at $P_{\rm syn}$" % (tc, lbl), fontsize=10)
    axr[2].legend(frameon=False, fontsize=8, loc="upper right", markerscale=2)
    return rs * UAS, re * UAS


def build_figure(out=OUT):
    """Build the two-row (a=20, a=50 R_Jup) three-view comparison and save it to ``out``."""
    VIR = plt.get_cmap("viridis")
    INC = 60.0
    fig, ax = plt.subplots(2, 3, figsize=(17.4, 9.2))
    r1 = draw_row(ax[0], VIR, INC, 20.0, ("a", "b", "c"))   # nominal moon (P_syn ~ 21 d)
    r2 = draw_row(ax[1], VIR, INC, 50.0, ("d", "e", "f"))   # wider, ~4x-period moon (P_syn ~ 87 d)
    for row in ax:
        for x in row:
            x.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    return r1, r2


if __name__ == "__main__":
    r1, r2 = build_figure()
    print("wrote", OUT)
    print("a=20 R_J : sine RMS=%.1f  ellipse RMS=%.1f  (gain %.2fx)" % (r1[0], r1[1], r1[0] / r1[1]))
    print("a=50 R_J : sine RMS=%.1f  ellipse RMS=%.1f  (gain %.2fx)" % (r2[0], r2[1], r2[0] / r2[1]))
