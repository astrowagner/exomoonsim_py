#!/usr/bin/env python3
"""The observing window and the ~1-yr aliasing of long-period (wide-a) moons (for the paper).

The campaign observes ~10 months of each year and skips ~2 (year_gap_frac = 0.83), a pattern that
repeats every 12 months.  That annual repetition stamps a comb at 1/yr and its harmonics into the
SPECTRAL WINDOW of the sampling -- so a periodicity search for a long-period moon can lock onto the
~1-yr window instead of the moon.  Two panels:

  (a) spectral window |W(f)|^2 = |sum_j exp(-2 pi i f t_j)|^2 of the observing epochs, vs period;
      the 1-yr peak and its harmonics (6 mo, 4 mo, ...) are the alias-driving structure.
  (b) recovered vs true synodic period as the moon separation grows: the blind single-sinusoid
      search tracks Kepler up to a ~ 50 R_Jup, then departs and pins near ~1 yr -- the alias.

Panel (b) reads the recovered periods from analysis/_ab_scaling.npy (see analysis/ab_scaling.py).
Run:  python analysis/window_function.py
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
OUT = os.path.join(HERE, "window_function.png")
AB = os.path.join(HERE, "_ab_scaling.npy")

# --- fiducial sampling (matches run_trial): 5 yr, 1 hr cadence, 10-month observing season -------
DUR, TEXP, GAP = 1826.25, 1.0, 0.83
P_PL = 1.8 ** 1.5 * 365.25                              # planet period (days)
P_SID10 = 7.132                                         # moon sidereal period at a=10 R_Jup (days)


def observing_epochs():
    n = int(DUR * 24.0 / TEXP)
    t = np.arange(n) * TEXP / 24.0 / 365.25             # years
    return (t[(t % 1.0) <= GAP]) * 365.25               # observed epochs, days


def spectral_window(tdays, periods):
    f = 1.0 / periods                                   # cycles/day
    W = np.abs(np.exp(-2j * np.pi * np.outer(f, tdays)).sum(axis=1)) ** 2
    return W / len(tdays) ** 2


def true_synodic(a):
    psid = P_SID10 * (a / 10.0) ** 1.5
    return 1.0 / (1.0 / psid - 1.0 / P_PL)


if __name__ == "__main__":
    tdays = observing_epochs()
    per = np.logspace(np.log10(2), np.log10(1000), 1400)
    W = spectral_window(tdays, per)

    VIR = plt.get_cmap("viridis")
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.4))

    # (a) spectral window
    ax[0].plot(per, W, color=VIR(0.28), lw=1.0)
    for k in range(1, 6):
        ax[0].axvline(365.25 / k, color="crimson", ls=":", lw=0.9, zorder=0)
    ax[0].text(365.25, W.max() * 1.05, "1 yr", color="crimson", fontsize=8, ha="center", va="bottom")
    ax[0].text(365.25 / 2, W[np.argmin(np.abs(per - 365.25 / 2))] * 1.4, "6 mo", color="crimson",
               fontsize=7.5, ha="center", va="bottom")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylim(max(W.min(), 1e-6), 2.0)
    ax[0].set_xlabel("Period (days)")
    ax[0].set_ylabel(r"Spectral Window $|W|^2$")
    ax[0].set_title("(a) Observing window: 1-yr comb", fontsize=10)

    # (b) recovered vs true period (alias plateau)
    if os.path.exists(AB):
        R = np.load(AB); a, prec = R[:, 0], R[:, 2]
        ag = np.logspace(np.log10(a.min()), np.log10(a.max()), 200)
        ax[1].plot(ag, true_synodic(ag), "-", color="0.55", lw=1.4, label="True synodic period")
        clean = prec < 250
        ax[1].plot(a[clean], prec[clean], "o", color=VIR(0.28), ms=7, label="Recovered (blind search)")
        ax[1].plot(a[~clean], prec[~clean], "o", mfc="none", mec=VIR(0.28), ms=7, mew=1.5,
                   label=r"Recovered, aliased to $\sim$1 yr")
        ax[1].axhline(365.25, color="crimson", ls=":", lw=1.0)
        ax[1].text(a.min() * 1.05, 365.25 * 1.05, "1 yr", color="crimson", fontsize=8, va="bottom")
        ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel(r"Moon Semimajor Axis $a$ ($R_{\rm Jup}$)")
        ax[1].set_ylabel("Period (days)")
        ax[1].set_title("(b) Search aliases at wide separation", fontsize=10)
        ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    else:
        ax[1].text(0.5, 0.5, "run analysis/ab_scaling.py first\n(for _ab_scaling.npy)",
                   ha="center", va="center", transform=ax[1].transAxes)

    for x in ax:
        x.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
