#!/usr/bin/env python3
"""Why the detection maps are flat in semimajor axis a (for Sumin/Wagner).

Single moon (m = 0.1 Mearth, i = 50 deg) around the fiducial alpha Cen A planet
(P_pl ~ 882 d), 1 hr cadence, 5 yr, noiseless signal so we can read off the intrinsic
scalings.  Three panels:

  (1) reflex amplitude A ~ a.
  (2) LEFT axis: moon/planet period ratio P_moon/P_pl (~a^3/2), which drives the
      planet-motion sidebands.  RIGHT axis: the periodogram peak-to-sidelobe ratio.
      The sidelobes stay a small fraction of the peak (ratio >> 1) all the way to
      a ~ 50 R_Jup, so sideband splitting is a MINOR effect -- until the ~1-yr annual
      alias takes over at large a.
  (3) intrinsic detectability: the ideal matched filter (total signal power / sigma^2)
      scales as a^2 exactly, while the blind single-sinusoid statistic the survey uses
      is flat -- it peaks near a ~ 10 R_Jup and rolls over.  Shown normalized to a=10
      (the two statistics have different absolute normalizations; only shapes compare).
      Past a ~ 63 R_Jup the recovered period sticks near ~1 yr (annual-gap alias).

Together: the near-flat SUCCESS-FRACTION map is mostly detection saturation
(Delta chi^2 >> threshold -> detfrac = 1) plus the blind detector's flat/aliased
response, not the intrinsic a-scaling (which is steep, ~a^2).

Run:  python analysis/ab_scaling.py   (cached; --plot-only style: rerun replots)
"""
import os
import sys
import numpy as np
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from exomoonsim.sim import SimParams, run_trial

OUT = os.path.join(HERE, "ab_scaling.png")
CACHE = os.path.join(HERE, "_ab_scaling.npy")
AS = [3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 120]     # R_Jup
SIG = 1e-5                                            # 10 uas reference for the matched filter
P_PL = 1.8 ** 1.5 * 365.25                            # planet period (days), a_pl=1.8 au, M=1


def one(a):
    d = run_trial(SimParams(moon_a=a, moon_mass=0.1, moon_inc=50, moon_ecc=0.0,
                            astrometric_precision=1e-7, pend=500.0),
                  seed=1, return_diagnostics=True, recover_thr=1e12)["diag"]
    rres = np.asarray(d["rres"])
    mf = float(np.sum(rres ** 2)) / SIG ** 2          # ideal matched-filter dchi at 10 uas
    pg = np.asarray(d["period_grid"]); sg = np.asarray(d["periodogram"])
    Psyn = d["true_synodic"]; Psid = float(d.get("sidereal_period", np.nan))
    Plo = 1.0 / (1.0 / Psyn + 1.0 / P_PL)             # planet-motion sidebands at +-1/P_pl
    Phi = 1.0 / (1.0 / Psyn - 1.0 / P_PL)

    def near(P, f=0.015):
        m = np.abs(pg / P - 1.0) < f
        return float(np.max(sg[m])) if m.any() else np.nan
    peak = near(Psyn)
    lobe = np.nanmax([near(Plo), near(Phi)])
    ratio = peak / lobe if (lobe and lobe > 0) else np.nan
    return a, d["input_amp"] * 1e3, d["best_period"], mf, d["sig"], Psid, ratio


if __name__ == "__main__":
    if os.path.exists(CACHE):
        R = np.load(CACHE)
    else:
        with Pool(min(len(AS), os.cpu_count())) as p:
            R = np.array(p.map(one, AS), float)
        np.save(CACHE, R)
    a, amp, Pbest, mf, blind, Psid, ratio = R.T
    clean = Pbest < 250.0                              # recovered period not stuck near ~1 yr
    a_alias = float(np.sqrt(a[clean][-1] * a[~clean][0]))

    VIR = plt.get_cmap("viridis")
    C_L, C_R = VIR(0.28), VIR(0.72)
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.3))

    # (1) amplitude ------------------------------------------------------------
    ax[0].plot(a, amp, "o-", color=VIR(0.28), ms=6, lw=1.4)
    ax[0].plot(a, amp[0] * (a / a[0]), "--", color="0.6", lw=1.1, label=r"$\propto a$")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    ax[0].set_ylabel(r"Reflex Amplitude ($\mu$as)")
    ax[0].set_title(r"(1) Signal grows $\propto a$", fontsize=10)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")

    # (2) period ratio (left) and peak/sidelobe (right) ------------------------
    axL = ax[1]; axR = axL.twinx()
    Psid_kep = Psid[3] * (a / a[3]) ** 1.5             # clean Kepler moon period (a[3]=10); avoids alias
    l1, = axL.plot(a, Psid_kep / P_PL, "o-", color=C_L, ms=6, lw=1.5)
    l2, = axR.plot(a[clean], ratio[clean], "s-", color=C_R, ms=6, lw=1.5)
    axR.plot(a[~clean], ratio[~clean], "s", mfc="none", mec=C_R, ms=6, mew=1.4)
    axL.set_xscale("log"); axL.set_yscale("log"); axR.set_yscale("log")
    axL.set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    axL.set_ylabel(r"Moon/Planet Period $P_{\rm moon}/P_{\rm pl}$", color=C_L)
    axR.set_ylabel(r"Periodogram Peak / Sidelobe", color=C_R)
    axL.tick_params(axis="y", colors=C_L); axR.tick_params(axis="y", colors=C_R)
    axL.set_title("(2) Sidebands stay weak (peak $\\gg$ sidelobe)", fontsize=10)
    axL.legend([l1, l2], [r"$P_{\rm moon}/P_{\rm pl}$", "Peak/sidelobe"],
               frameon=False, fontsize=8.5, loc="center left")

    # (3) intrinsic a^2 vs flat blind detector (normalized to a=10) ------------
    i10 = 3                                            # a[3] = 10 R_Jup
    ax[2].axvspan(a_alias, a[-1] * 1.15, color="0.92", zorder=0)
    ax[2].plot(a, (a / a[i10]) ** 2, "--", color="0.55", lw=1.1, zorder=1, label=r"$\propto a^{2}$")
    ax[2].plot(a, mf / mf[i10], "o-", color=VIR(0.28), ms=6, lw=1.5, zorder=3,
               label="Matched filter (ideal)")
    ax[2].plot(a[clean], (blind / blind[i10])[clean], "D-", color=VIR(0.75), ms=6, lw=1.5,
               zorder=3, label="Blind single-sinusoid")
    ax[2].plot(a[~clean], (blind / blind[i10])[~clean], "D", mfc="none", mec=VIR(0.75),
               ms=6, mew=1.4, zorder=3, label=r"Blind (aliased to $\sim$1 yr)")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_ylim(6e-2, 4e2)
    ax[2].text(a_alias * 1.06, 0.14, "annual-gap\nalias", fontsize=7.5, color="0.45", va="bottom")
    ax[2].set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
    ax[2].set_ylabel(r"Significance (rel. to $a=10\,R_{\rm Jup}$)")
    ax[2].set_title(r"(3) Ideal $\propto a^2$; blind detector is flat", fontsize=10)
    ax[2].legend(frameon=False, fontsize=7.6, loc="upper left")

    ax[0].grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax[2].grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
    print("a_alias ~ %.0f Rjup ; P_pl = %.0f d" % (a_alias, P_PL))
