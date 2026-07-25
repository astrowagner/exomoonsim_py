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
class Moon:
    """One moon's orbit about the planet (defaults match the single-moon values)."""
    a: float = 18.0            # Jupiter radii
    mass: float = 1.0          # Earth masses
    inc: float = 50.0          # deg
    ecc: float = 0.05
    omega: float = 0.0
    bigomega: float = 0.0
    t0: float = 0.0
    retrograde: bool = False


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
    # optional explicit list of Moon; when set it OVERRIDES the scalar moon_* fields
    # above (which then just describe a default single moon).  The "primary" moon --
    # the one a blind single-period search recovers first -- is the one with the
    # largest reflex (m*a); recovery is scored against it.
    moons: list = None
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

    # ---- moon bookkeeping (single scalar moon, or the explicit `moons` list) ----
    def moon_specs(self):
        """Effective list of moons: the explicit ``moons`` if given, else one Moon
        built from the scalar ``moon_*`` fields."""
        if self.moons:
            return list(self.moons)
        return [Moon(a=self.moon_a, mass=self.moon_mass, inc=self.moon_inc,
                     ecc=self.moon_ecc, omega=self.moon_omega,
                     bigomega=self.moon_bigomega, t0=self.moon_t0,
                     retrograde=self.retrograde)]

    def _cm_au(self, moon):
        """One moon's reflex: planet's distance from the planet-moon barycenter (au);
        proportional to moon mass x semimajor axis."""
        return (moon.mass * C.MEAR2KG
                / (self.planet_mass * C.MJUP2KG + moon.mass * C.MEAR2KG)
                * (moon.a * C.RJUP2M / C.AU2M))

    def _period_days(self, moon):
        """One moon's sidereal period about the planet (days)."""
        a_au = moon.a / C.RJUP_PER_AU
        mu = (self.planet_mass * C.MJUP2KG + moon.mass * C.MEAR2KG) / C.MSUN2KG
        return a_au ** 1.5 / np.sqrt(mu) * 365.25

    def primary_index(self):
        """Index of the primary moon (largest reflex, i.e. largest m*a)."""
        moons = self.moon_specs()
        return int(np.argmax([self._cm_au(mn) for mn in moons]))

    def moon_period_days(self):
        """Sidereal period (days) of the primary moon."""
        moons = self.moon_specs()
        return self._period_days(moons[self.primary_index()])

    def planet_period_yr(self):
        mtot = sum(mn.mass for mn in self.moon_specs())
        mu = (self.star_mass + self.planet_mass * C.MJUP2KG / C.MSUN2KG
              + mtot * C.MEAR2KG / C.MSUN2KG)
        return self.planet_a ** 1.5 / np.sqrt(mu)

    def cm_au(self):
        """Reflex (au) of the primary moon."""
        moons = self.moon_specs()
        return self._cm_au(moons[self.primary_index()])

    def input_amp_mas(self):
        """Astrometric semi-amplitude (mas) of the primary moon's wobble."""
        return self.cm_au() / self.system_distance * 1000.0

    # ---- inverse model: recovered (period, amplitude) -> (a, mass) ----
    def a_from_period(self, period_days):
        """Sidereal period (days) -> moon semimajor axis (R_Jup), via Kepler,
        neglecting the moon mass (m << M_planet)."""
        mu = self.planet_mass * C.MJUP2KG / C.MSUN2KG
        a_au = (period_days / 365.25) ** (2.0 / 3.0) * mu ** (1.0 / 3.0)
        return a_au * C.RJUP_PER_AU

    def mass_from_amp(self, amp_mas, a_rjup):
        """Wobble semi-amplitude (mas) + semimajor axis (R_Jup) -> moon mass
        (M_Earth), inverting the reflex relation."""
        a_au = a_rjup / C.RJUP_PER_AU
        cm_au = amp_mas * self.system_distance / 1000.0
        f = min(cm_au / a_au, 0.999)               # = m/(M_p+m)
        return f * self.planet_mass * C.MJUP2KG / (C.MEAR2KG * (1.0 - f))


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
def run_trial(params: SimParams = None, seed=None, return_diagnostics=False,
              recover_thr=5.0, recover_blank=0.25):
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

    # ---- moons: reflex + period for each; primary = largest reflex (m*a) ----
    moons = p.moon_specs()
    cms = [p._cm_au(mn) for mn in moons]              # au
    periods = [p._period_days(mn) for mn in moons]    # sidereal, days
    ip = int(np.argmax(cms))                          # primary (most discernible)
    moon_period = periods[ip]                         # scored against the primary
    input_amp = cms[ip] / d * 1000.0                  # primary wobble amplitude (mas)

    # ---- observation times (years since start); annual observing gap ----
    nobs = int(p.duration_days * 24.0 / (p.texp + p.spacing))
    t = np.arange(nobs) * (p.texp + p.spacing) / 24.0 / 365.25
    keep = (t % 1.0) <= p.year_gap_frac
    t = t[keep]

    # ---- planet track: barycenter about star + superposed moon wobbles ----
    com_x, com_y, _ = orbit_xy(p.planet_a, p.planet_ecc, planet_period,
                               p.planet_inc, p.planet_omega, p.planet_bigomega,
                               p.planet_t0, t)
    wob_x = np.zeros_like(com_x)
    wob_y = np.zeros_like(com_y)
    for mn, cm, per in zip(moons, cms, periods):
        wx, wy, _ = orbit_xy(cm, mn.ecc, per / 365.25, mn.inc, mn.omega,
                             mn.bigomega, mn.t0, t)
        if mn.retrograde:
            wx = -wx
        wob_x = wob_x + wx
        wob_y = wob_y + wy
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

    # synodic -> sidereal period (using the primary moon's sense of rotation)
    sign = -1.0 if moons[ip].retrograde else 1.0
    sidereal = 1.0 / (1.0 / best_pp + sign / (365.25 * fit.period))
    perr = abs(sidereal - moon_period) / moon_period * 100.0

    # ---- amplitude from a final phase-fold + sine fit at the best period ----
    retrieved_amp = _amplitude_at(tdays, rres, rres2, best_pp, p.binwidth, p.window) * 1000.0 * d
    # input_amp == 0 only for a zero-mass moon (the no-signal control cell); IDL's
    # float divide yields Inf there, so mirror that rather than raising.
    amperr = (abs(retrieved_amp - input_amp) / input_amp * 100.0
              if input_amp != 0 else np.inf)

    out = dict(sig=float(sig_val), perr=float(perr), amperr=float(amperr),
               resav=float(resav), best_period=float(best_pp),
               sidereal_period=float(sidereal), moon_period=float(moon_period),
               input_amp=float(input_amp), retrieved_amp=float(retrieved_amp),
               fit_success=bool(fit.success))

    if return_diagnostics:
        # Re-fold the residual at the best (synodic) period and keep the
        # intermediate arrays so a single trial can be plotted (plots.plot_trial).
        nbin = int(np.floor(best_pp) / p.binwidth)
        phase = tdays % best_pp
        tt, amp_b, nph, sph = _phasefold(phase, rres, rres2, p.binwidth, p.window, nbin)
        good = (nph >= 2) & np.isfinite(amp_b) & (sph > 0)
        sfit = sine_fit_fixed_period(tt[good], amp_b[good], best_pp)
        # The "recovered removed" residual panels are built BLIND in the prewhitening loop
        # below by subtracting the RECOVERED phase-folded templates (never the truth), so
        # they show what is actually achievable without prior knowledge -- including the
        # sideband leftover the templates cannot fully capture.
        rres_after = rres            # residual after the primary is removed (set in loop)
        true_synodic = 1.0 / (1.0 / moon_period - sign / (planet_period * 365.25))
        comp_synodics = []
        for j, (mn, per) in enumerate(zip(moons, periods)):
            if j == ip:
                continue
            sgn = -1.0 if mn.retrograde else 1.0
            comp_synodics.append(1.0 / (1.0 / per - sgn / (planet_period * 365.25)))

        # --- iterative prewhitening: recover each moon down to the chi2 threshold ---
        def _search(res):
            r2 = res * res
            ppv, sgv = [], []
            pp = p.pstart
            while pp < p.pend:
                ppv.append(pp)
                sgv.append(_period_significance(tdays, res, r2, pp, p.binwidth, p.window))
                pp *= (1.0 + p.ptestwidth)
            return np.asarray(ppv), np.asarray(sgv)

        phi = np.arctan2(pred_y, pred_x)              # known planet position angle vs time
        recoveries = []
        res_pw = rres
        claimed = []
        cur = (pps, sigs)                       # round 1 reuses the main search on rres
        for _round in range(len(moons) + 2):
            ppv, sgv = cur if cur is not None else _search(res_pw)
            sgm = sgv.astype(float).copy()
            # blank a window around already-claimed periods so the search doesn't
            # re-lock on the (broad, two-sided) leftover of imperfect subtraction.
            # recover_blank is the half-width as a fraction of period; well-separated
            # moons are safe, closely-spaced ones need a smaller value + care.
            for cp in claimed:
                sgm[np.abs(ppv / cp - 1.0) < recover_blank] = -np.inf
            bi = int(np.nanargmax(sgm))
            speak, ppk = float(sgm[bi]), float(ppv[bi])
            if (not np.isfinite(speak)) or speak <= recover_thr:
                recoveries.append(dict(recovered=False, period_grid=ppv, periodogram=sgv,
                                       peak_sig=float(np.nanmax(sgv))))
                break
            # phase-fold at the synodic period (for the recovery-row plots)
            nb2 = int(np.floor(ppk) / p.binwidth)
            ph2 = tdays % ppk
            t2, a2, n2, s2 = _phasefold(ph2, res_pw, res_pw * res_pw, p.binwidth, p.window, nb2)
            g2 = (n2 >= 2) & np.isfinite(a2) & (s2 > 0)
            sf2 = sine_fit_fixed_period(t2[g2], a2[g2], ppk)
            sid = 1.0 / (1.0 / ppk + 1.0 / (365.25 * fit.period))     # prograde
            a_rj = p.a_from_period(sid)

            # physics model: the moon at its SIDEREAL period x the known planet direction
            # phi(t).  Fitting this 2-D-projected model (a) subtracts the signal including
            # its planet-orbit sidebands, and (b) yields the projected reflex ellipse ->
            # inclination and a mass largely free of the inclination degeneracy.
            wsid = 2.0 * np.pi / sid
            G = np.column_stack([np.cos(wsid * tdays) * np.cos(phi),
                                 np.cos(wsid * tdays) * np.sin(phi),
                                 np.sin(wsid * tdays) * np.cos(phi),
                                 np.sin(wsid * tdays) * np.sin(phi),
                                 np.ones(tdays.size)])
            coef = np.linalg.lstsq(G, res_pw, rcond=None)[0]
            model = G @ coef
            axc, ayc, bxc, byc = coef[:4]
            Cc, Ss, CS = axc**2 + ayc**2, bxc**2 + byc**2, axc * bxc + ayc * byc
            Rr = np.sqrt(((Cc - Ss) / 2.0) ** 2 + CS ** 2)
            smaj = np.sqrt(max((Cc + Ss) / 2.0 + Rr, 0.0))
            smin = np.sqrt(max((Cc + Ss) / 2.0 - Rr, 0.0))
            incl = float(np.degrees(np.arccos(np.clip(smin / max(smaj, 1e-30), 0.0, 1.0))))
            amp_mas = smaj * 1000.0                   # sky reflex semi-major -> mas
            mass = p.mass_from_amp(amp_mas, a_rj)
            resid = res_pw - model                    # statistical amplitude -> mass error
            dof = max(res_pw.size - G.shape[1], 1)
            gcov = np.linalg.inv(G.T @ G) * (float(np.sum(resid ** 2)) / dof)
            amp_err_mas = np.sqrt(max(float(np.mean(np.diag(gcov)[:4])), 0.0)) * 1000.0
            mass_err = mass * amp_err_mas / max(amp_mas, 1e-30)

            recoveries.append(dict(recovered=True, period_grid=ppv, periodogram=sgv,
                                   best_period=ppk, sidereal=sid, sig=speak, amp_mas=amp_mas,
                                   a_rjup=a_rj, mass=mass, inclination=incl, mass_err=mass_err,
                                   fold_phase=t2[g2], fold_amp=a2[g2],
                                   fold_err=s2[g2] / np.sqrt(n2[g2]), fold_model=sf2.model))
            res_pw = res_pw - model                   # subtract the physics model
            claimed.append(ppk)
            cur = None
            if len(claimed) == 1:                     # residual just after the primary
                rres_after = res_pw

        # BLIND residual after removing all recovered signals (physics model, never truth).
        rres_allremoved = res_pw

        out["diag"] = dict(
            t_years=t, tdays=tdays,
            x_obs=x_obs, y_obs=y_obs,
            x_true=planet_x / d, y_true=planet_y / d,
            x_fit=pred_x, y_fit=pred_y,
            x_com=com_x / d, y_com=com_y / d,
            rres=rres, rres_after=rres_after, rres_allremoved=rres_allremoved,
            period_grid=pps, periodogram=sigs,
            best_period=float(best_pp), true_synodic=float(true_synodic),
            companion_synodics=comp_synodics, precision_uas=float(sig_pos * 1e6),
            recoveries=recoveries, input_moons=[(mn.a, mn.mass) for mn in moons],
            recover_thr=float(recover_thr),
            fold_phase=tt[good], fold_amp=amp_b[good],
            fold_err=sph[good] / np.sqrt(nph[good]), fold_model=sfit.model,
            input_amp=float(input_amp), retrieved_amp=float(retrieved_amp),
            sig=float(sig_val), perr=float(perr), amperr=float(amperr),
            sidereal_period=float(sidereal), moon_period=float(moon_period))
    return out


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
