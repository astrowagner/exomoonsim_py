#!/usr/bin/env python3
"""Figure: recovery_summary.png  (Paper III, Figure 1) + Table 3 numbers.

Three panels for a single 1 M_earth moon at a=20 R_Jup around the alpha Cen A
giant-planet candidate, observed at 10 uas over 5 yr (circular orbit):
  (a) separation-residual periodogram: synodic peak + planet-orbit sidebands (+-1/P_pl)
  (b) the recovered sky-projected reflex ellipse (semi-major -> mass, axis ratio -> cos i)
  (c) recovered mass vs true inclination: amplitude-only (biased) vs ellipse fit (unbiased)

Also prints the rows of Table 3 (inclination dependence of the recovered mass) and
the fiducial residual-RMS numbers quoted in Section 4.2.

Reproducible: uses the frozen exomoonsim snapshot in scripts/exomoonsim and fixed
seeds. Run:  python make_recovery_fig.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))                       # repo root -> exomoonsim package
from exomoonsim.sim import SimParams, run_trial

FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)
OUT = os.path.join(FIGDIR, "recovery_summary.png")

SEED = 11                                       # fixed for reproducibility

def cfg(inc):
    """Fiducial single-moon configuration (matches Table 1 host/observing)."""
    return SimParams(astrometric_precision=1e-5, duration_days=5 * 365.25,
                     texp=1.0, pstart=1.0, pend=60.0, ptestwidth=0.002,
                     moon_a=20.0, moon_mass=1.0, moon_inc=float(inc), moon_ecc=0.0)

# --------------------------------------------------------------------------- #
# Inclination scan (Table 3 + panel c) and the fiducial i=55 trial (panels a,b)
# --------------------------------------------------------------------------- #
incs = np.array([10., 30., 50., 55., 70., 80.])
FID = 55.0
m_amp, m_ell, i_ell = [], [], []
fid = None
for inc in incs:
    p = cfg(inc)
    r = run_trial(p, seed=SEED, return_diagnostics=True)
    d = r["diag"]; rc = d["recoveries"][0]
    a_rj = rc["a_rjup"]
    m_amp.append(p.mass_from_amp(d["retrieved_amp"], a_rj))   # amplitude-only (Papers I-II)
    m_ell.append(rc["mass"]); i_ell.append(rc["inclination"])  # ellipse fit
    if inc == FID:
        td = d["tdays"]; phi = np.arctan2(d["y_fit"], d["x_fit"]); w = 2 * np.pi / rc["sidereal"]
        G = np.column_stack([np.cos(w * td) * np.cos(phi), np.cos(w * td) * np.sin(phi),
                             np.sin(w * td) * np.cos(phi), np.sin(w * td) * np.sin(phi),
                             np.ones(td.size)])
        c = np.linalg.lstsq(G, d["rres"], rcond=None)[0]
        fid = dict(pg=rc["period_grid"], sg=rc["periodogram"], best=d["best_period"],
                   Ppl=p.planet_period_yr() * 365.25, coef=c[:4], i_rec=rc["inclination"],
                   smaj_uas=rc["amp_mas"] * 1000.0, mass=rc["mass"], mass_err=rc["mass_err"],
                   rms_raw=np.std(d["rres"]) * 1e6, rms_clean=np.std(d["rres_allremoved"]) * 1e6,
                   prec=d["precision_uas"])
m_amp = np.array(m_amp); m_ell = np.array(m_ell); i_ell = np.array(i_ell)

print("Table 3 (inclination dependence, single 1.0 Mearth moon):")
print(" i_true   m(amp-only)   m(ellipse)   i(ellipse)")
for k in range(len(incs)):
    print(f"  {incs[k]:4.0f}      {m_amp[k]:6.3f}       {m_ell[k]:6.3f}      {i_ell[k]:5.1f}")
print(f"\nFiducial (i={FID:.0f}): mass={fid['mass']:.3f}+/-{fid['mass_err']:.3f}, "
      f"i_rec={fid['i_rec']:.1f}")
print(f"Residual RMS: raw={fid['rms_raw']:.1f} uas -> after model={fid['rms_clean']:.1f} uas "
      f"(precision={fid['prec']:.1f} uas)")

# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
VIR = plt.get_cmap("viridis")
C_ELL, C_AMP, C_SIDE = VIR(0.28), VIR(0.70), VIR(0.62)
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))

# (a) periodogram + sidebands
pg, sg, best, Ppl = fid["pg"], fid["sg"], fid["best"], fid["Ppl"]
ax[0].plot(pg, sg, color=C_ELL, lw=1.1)
sb_lo = 1.0 / (1.0 / best + 1.0 / Ppl); sb_hi = 1.0 / (1.0 / best - 1.0 / Ppl)
ax[0].axvline(best, color="0.6", ls="-", lw=0.8)
for sb in (sb_lo, sb_hi):
    ax[0].axvline(sb, color=C_SIDE, ls="--", lw=0.9)
ax[0].set_xlim(best * 0.6, best * 1.6)
ax[0].annotate("Synodic", (best, sg.max()), xytext=(0, -3), textcoords="offset points",
               ha="center", va="top", fontsize=8, color="0.35")
ax[0].annotate(r"$\pm\,1/P_{\rm pl}$", (sb_hi, sg.max() * 0.5), xytext=(5, 0),
               textcoords="offset points", ha="left", fontsize=8, color=C_SIDE)
ax[0].set_xlabel("Period (days)"); ax[0].set_ylabel(r"Significance  $\Delta\chi^2$")
ax[0].set_title("(a) Periodogram + Sidebands", fontsize=10)

# (b) recovered reflex ellipse
axc, ayc, bxc, byc = fid["coef"]; i_rec = fid["i_rec"]; smaj = fid["smaj_uas"]
u = np.linspace(0, 2 * np.pi, 400)
Xe = (axc * np.cos(u) + bxc * np.sin(u)) * 1e6
Ye = (ayc * np.cos(u) + byc * np.sin(u)) * 1e6
ax[1].plot(Xe, Ye, color=C_ELL, lw=1.6)
th = 0.5 * np.arctan2(2 * (axc * bxc + ayc * byc),
                      (axc**2 + ayc**2) - (bxc**2 + byc**2))
smin = smaj * np.cos(np.radians(i_rec))
for L, cth, sth in ((smaj, np.cos(th), np.sin(th)), (smin, -np.sin(th), np.cos(th))):
    ax[1].plot([-L * cth, L * cth], [-L * sth, L * sth], color=C_AMP, lw=1.0)
ax[1].add_patch(plt.Circle((0, 0), smaj, fill=False, ec="0.7", ls=":", lw=0.8))
ax[1].set_aspect("equal"); lim = smaj * 1.25
ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
ax[1].set_xlabel(r"$\Delta$RA Reflex ($\mu$as)"); ax[1].set_ylabel(r"$\Delta$Dec Reflex ($\mu$as)")
ax[1].set_title("(b) Recovered Reflex Ellipse", fontsize=10)
ax[1].text(0.04, 0.96, f"$i_{{\\rm true}}={FID:.0f}^\\circ$\n$i_{{\\rm rec}}={i_rec:.0f}^\\circ$",
           transform=ax[1].transAxes, va="top", ha="left", fontsize=9)

# (c) mass vs inclination
ax[2].axhline(1.0, color="0.6", ls=":", lw=1.0, zorder=0)
ax[2].plot(incs, m_amp, "o-", color=C_AMP, lw=1.3, ms=5, label="Amplitude Only")
ax[2].plot(incs, m_ell, "s-", color=C_ELL, lw=1.3, ms=5, label="Ellipse Fit")
ax[2].set_xlabel(r"True Inclination $i$ (deg)"); ax[2].set_ylabel(r"Recovered Mass ($M_\oplus$)")
ax[2].set_title("(c) Breaking the Degeneracy", fontsize=10)
ax[2].set_ylim(0.5, 1.35); ax[2].legend(frameon=False, fontsize=8, loc="lower left")
ax[2].text(0.96, 0.965, "Truth", transform=ax[2].transAxes, ha="right", va="top",
           fontsize=8, color="0.5")

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("\nwrote", OUT)
