"""
Single-trial exomoon astrometric detection simulator.

Port of the per-trial pipeline in exomoonsim.pro:
  1. build a planet that wobbles about the planet-moon barycenter,
  2. observe its on-sky track over a campaign (with astrometric noise + an annual
     observing gap),
  3. fit and subtract the planet's orbit about the star,
  4. search the separation residual for the moon's periodic signal (phase-fold +
     fixed-period sine fit over a grid of trial periods),
  5. report the detection significance and the recovered period / amplitude error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from . import constants as C
from .orbits import orbit_xy
from .fit import fit_planet_orbit, sine_fit_fixed_period


@dataclass
class SimParams:
    """System + observing configuration (defaults match exomoonsim.pro)."""
    # star / distance
    system_distance: float = 1.3      # pc
    star_mass: float = 1.0            # Msun
    # planet
    planet_a: float = 1.8            # au
    planet_mass: float = 0.3         # MJup
    planet_inc: float = 45.0         # deg
    planet_ecc: float = 0.3
    planet_omega: float = 150.0
    planet_bigomega: float = 150.0
    planet_t0: float = 3.0           # yr (within the campaign window)
    # moon
    moon_a: float = 18.0             # Jupiter radii
    moon_mass: float = 1.0           # Earth masses
    moon_inc: float = 50.0
    moon_ecc: float = 0.05
    moon_omega: float = 0.0
    moon_bigomega: float = 0.0
    moon_t0: float = 0.0
    retrograde: bool = False
    # observing
    astrometric_precision: float = 1e-4   # arcsec (1e-4 = 100 muas; 1e-5 = 10 muas)
    duration_days: float = 5.0 * 365.25   # campaign length (matches current IDL; set 10*365.25 for a 10-yr run)
    texp: float = 1.0                # hr
    spacing: float = 0.0             # hr overhead
    year_gap_frac: float = 0.83      # keep only fractional-year phase < this
    # period search
    pstart: float = 1.0              # days (synodic)
    pend: float = 500.0
    ptestwidth: float = 0.001        # geometric step (fractional)
    binwidth: float = 0.1            # days
    window: float = 1.0              # days (sliding phase-fold window)

    def moon_period_days(self):
        """Sidereal moon period about the planet (days)."""
        a_au = self.moon_a / C.RJUP_PER_AU
        mu = (self.planet_mass * C.MJUP2KG + self.moon_mass * C.MEAR2KG) / C.MSUN2KG
        return a_au ** 1.5 / np.sqrt(mu) * 365.25

    def planet_period_yr(self):
        mu = (self.star_mass + self.planet_mass * C.MJUP2KG / C.MSUN2KG
              + self.moon_mass * C.MEAR2KG / C.MSUN2KG)
        return self.planet_a ** 1.5 / np.sqrt(mu)

    def cm_au(self):
        """Planet's distance from the planet-moon barycenter (au)."""
        return (self.moon_mass * C.MEAR2KG
                / (self.planet_mass * C.MJUP2KG + self.moon_mass * C.MEAR2KG)
                * (self.moon_a * C.RJUP2M / C.AU2M))

    def input_amp_mas(self):
        """True astrometric semi-amplitude of the planet's wobble (mas)."""
        return self.cm_au() / self.system_distance * 1000.0


# --------------------------------------------------------------------------- #
#  Phase-fold binning (overlapping sliding window, matching phasefold_bin.pro)
# --------------------------------------------------------------------------- #
def _phasefold(phase, rres, rres2, binwidth, window, nbin):
    """Mean/count/std of ``rres`` in overlapping windows of width ``window``.

    Bin ii covers [ii*binwidth, ii*binwidth + window).  Vectorized via fine-cell
    cumulative sums (window must be an integer number of fine cells).  ``rres2`` is
    rres**2 (precomputed once by the caller).  Every point falls in [0, nfine) so
    no masking is needed.
    """
    nwin = int(round(window / binwidth))
    nfine = nbin + nwin
    fb = (phase / binwidth).astype(np.intp)          # floor (phase >= 0)
    fcount = np.bincount(fb, minlength=nfine)[:nfine].astype(float)
    fsum = np.bincount(fb, weights=rres, minlength=nfine)[:nfine]
    fsumsq = np.bincount(fb, weights=rres2, minlength=nfine)[:nfine]
    ccnt = np.concatenate([[0.0], np.cumsum(fcount)])
    csum = np.concatenate([[0.0], np.cumsum(fsum)])
    csq = np.concatenate([[0.0], np.cumsum(fsumsq)])
    lo = np.arange(nbin)
    hi = lo + nwin
    n = ccnt[hi] - ccnt[lo]
    s = csum[hi] - csum[lo]
    q = csq[hi] - csq[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(n >= 1, s / n, np.nan)
        var = np.where(n >= 2, (q - s * s / n) / (n - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))
    return (lo + 0.5) * binwidth, amp, n, std


# --------------------------------------------------------------------------- #
#  One trial
# --------------------------------------------------------------------------- #
def run_trial(params: SimParams = None, seed=None):
    """Run a single simulation trial.

    Returns a dict with keys:
      sig     : detection significance  (chi2_flat - chi2_sine, reduced)
      perr    : sidereal-period error (%)
      amperr  : astrometric-amplitude error (%)
      resav   : planet-fit residual metric (for the adaptive threshold)
      best_period, sidereal_period, moon_period, input_amp, retrieved_amp
    """
    p = params or SimParams()
    rng = np.random.default_rng(seed)

    d = p.system_distance
    planet_period = p.planet_period_yr()
    moon_period = p.moon_period_days()           # sidereal, days
    cm = p.cm_au()
    moon_cm = (p.planet_mass * C.MJUP2KG
               / (p.planet_mass * C.MJUP2KG + p.moon_mass * C.MEAR2KG)
               * (p.moon_a * C.RJUP2M / C.AU2M))
    input_amp = p.input_amp_mas()

    # ---- observation times (years since start); annual observing gap ----
    nobs = int(p.duration_days * 24.0 / (p.texp + p.spacing))
    t = np.arange(nobs) * (p.texp + p.spacing) / 24.0 / 365.25
    keep = (t % 1.0) <= p.year_gap_frac
    t = t[keep]

    # ---- planet track: barycenter about star + planet wobble about barycenter ----
    com_x, com_y, _ = orbit_xy(p.planet_a, p.planet_ecc, planet_period,
                               p.planet_inc, p.planet_omega, p.planet_bigomega,
                               p.planet_t0, t)
    wob_x, wob_y, _ = orbit_xy(cm, p.moon_ecc, moon_period / 365.25,
                               p.moon_inc, p.moon_omega, p.moon_bigomega,
                               p.moon_t0, t)
    if p.retrograde:
        wob_x = -wob_x
    planet_x = com_x + wob_x
    planet_y = com_y + wob_y

    # ---- observe (au -> arcsec, plus astrometric noise) ----
    sig_pos = p.astrometric_precision
    x_obs = planet_x / d + rng.normal(0, sig_pos, t.size)
    y_obs = planet_y / d + rng.normal(0, sig_pos, t.size)
    obs_r = np.hypot(x_obs, y_obs)

    # ---- fit & subtract the planet's orbit about the star ----
    guess = np.array([planet_period, p.planet_t0, p.planet_ecc, p.planet_a / d,
                      p.planet_inc, p.planet_bigomega, p.planet_omega])
    guess = guess * (1.0 + rng.normal(0, 0.02, 7))
    fit = fit_planet_orbit(t, x_obs, y_obs, sig_pos, guess)
    if not np.isfinite(fit.period):
        return _nan_result(moon_period, input_amp)
    pred_x, pred_y, _ = orbit_xy(fit.a, fit.ecc, fit.period, fit.inc,
                                 fit.omega, fit.big_omega, fit.t_peri, t)
    pred_r = np.hypot(pred_x, pred_y)

    rres = obs_r - pred_r                      # separation residual = moon signal
    resav = np.sqrt(np.mean(rres ** 2))        # residual metric (for threshold)
    rres2 = rres * rres                         # precomputed once

    tdays = t * 365.25                          # phase-fold time axis (days)

    # ---- period search over synodic period ----
    pps, sigs = [], []
    pp = p.pstart
    while pp < p.pend:
        sig = _period_significance(tdays, rres, rres2, pp, p.binwidth, p.window)
        pps.append(pp)
        sigs.append(sig)
        pp *= (1.0 + p.ptestwidth)
    pps = np.asarray(pps)
    sigs = np.asarray(sigs)
    best = int(np.nanargmax(sigs))
    best_pp = pps[best]
    sig_val = sigs[best]

    # synodic -> sidereal period
    sign = -1.0 if p.retrograde else 1.0
    sidereal = 1.0 / (1.0 / best_pp + sign / (365.25 * fit.period))
    perr = abs(sidereal - moon_period) / moon_period * 100.0

    # ---- amplitude from a final phase-fold + sine fit at the best period ----
    retrieved_amp = _amplitude_at(tdays, rres, rres2, best_pp, p.binwidth, p.window) * 1000.0 * d
    # input_amp == 0 only for a zero-mass moon (the no-signal control cell); IDL's
    # float divide yields Inf there, so mirror that rather than raising.
    amperr = (abs(retrieved_amp - input_amp) / input_amp * 100.0
              if input_amp != 0 else np.inf)

    return dict(sig=float(sig_val), perr=float(perr), amperr=float(amperr),
                resav=float(resav), best_period=float(best_pp),
                sidereal_period=float(sidereal), moon_period=float(moon_period),
                input_amp=float(input_amp), retrieved_amp=float(retrieved_amp),
                fit_success=bool(fit.success))


def _period_significance(tdays, rres, rres2, pday, binwidth, window):
    nbin = int(np.floor(pday) / binwidth)
    if nbin < 6:
        return 0.0
    phase = tdays % pday
    tt, amp, n, s = _phasefold(phase, rres, rres2, binwidth, window, nbin)
    good = (n >= 2) & np.isfinite(amp) & (s > 0)
    ng = int(np.count_nonzero(good))
    if ng < 5:
        return 0.0
    a_g = amp[good]
    e_g = s[good] / np.sqrt(n[good])
    sfit = sine_fit_fixed_period(tt[good], a_g, pday)
    inv_e2 = 1.0 / (e_g * e_g)
    chi_flat = np.sum(a_g * a_g * inv_e2) / nbin
    chi_sine = np.sum((a_g - sfit.model) ** 2 * inv_e2) / (nbin - 4)
    return chi_flat - chi_sine


def _amplitude_at(tdays, rres, rres2, pday, binwidth, window):
    nbin = int(np.floor(pday) / binwidth)
    phase = tdays % pday
    tt, amp, n, s = _phasefold(phase, rres, rres2, binwidth, window, nbin)
    good = (n >= 2) & np.isfinite(amp) & (s > 0)
    if np.count_nonzero(good) < 5:
        return np.nan
    return sine_fit_fixed_period(tt[good], amp[good], pday, want_model=False).amp


def _nan_result(moon_period, input_amp):
    return dict(sig=np.nan, perr=np.nan, amperr=np.nan, resav=np.nan,
                best_period=np.nan, sidereal_period=np.nan,
                moon_period=float(moon_period), input_amp=float(input_amp),
                retrieved_amp=np.nan, fit_success=False)
