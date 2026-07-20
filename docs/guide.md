# The science and the algorithm

This guide explains **what `exomoonsim` simulates** and **how it decides whether a
moon is detectable**. It assumes no prior familiarity with the code — read it once
and the modules will make sense.

---

## 1. The idea in one paragraph

A planet with a moon does not sit still: the planet and moon both orbit their
common center of mass (the *barycenter*), while that barycenter orbits the star.
So the planet traces its stellar orbit **plus a small extra wobble** at the moon's
orbital period. If we measure the planet's position on the sky precisely enough
(*astrometry*), that wobble is a signature of the moon. `exomoonsim` asks: **given
a telescope's astrometric precision, which moons (what mass, what orbital
separation) leave a wobble we can actually detect?**

---

## 2. What a single trial does

One "trial" (`exomoonsim.sim.run_trial`) is one simulated observing campaign:

1. **Build the true motion.**
   - The planet–moon **barycenter** orbits the star on a Keplerian orbit.
   - The **planet** orbits the barycenter on its own small Keplerian orbit; its
     semi-major axis is
     `a_planet_about_bary = a_moon · M_moon / (M_planet + M_moon)`
     — the lever-arm of the moon. This is the wobble we hope to detect.
   - Planet sky position = barycenter position + wobble. (`orbits.orbit_xy` does the
     Kepler solve and the projection onto the sky for both.)

2. **Observe it.** Sample the track over the campaign at a fixed cadence, convert
   from AU to arcseconds using the system distance, and add Gaussian **astrometric
   noise** of size `astrometric_precision` to each x and y measurement. An annual
   observing gap is applied (part of each year the target isn't visible).

3. **Remove the planet's stellar orbit.** Fit a 7-parameter Keplerian orbit
   (period, time of periastron, eccentricity, semi-major axis, inclination, node,
   argument of periapsis) to the noisy positions and subtract it
   (`fit.fit_planet_orbit`). What's left — the **residual separation** `rres` —
   contains the moon wobble plus noise.

4. **Search for the moon.** The wobble shows up in the residual at the moon's
   **synodic** period (its period *as seen against the rotating planet–star
   direction*, not its true/sidereal period). For each trial period on a fine grid:
   - **phase-fold** the residual (wrap time modulo the trial period) and average it
     in bins (`sim._phasefold`),
   - **fit a sine** at that fixed period (`fit.sine_fit_fixed_period` — exact,
     because with the period fixed the model is linear),
   - compute how much better the sine fits than a flat line, as a reduced-χ²
     difference: `significance = χ²_flat − χ²_sine`.
   The trial period with the highest significance is the detection.

5. **Report three numbers.**
   - **significance** — how strongly a periodic signal stands out.
   - **period error (%)** — how far the recovered (sidereal) moon period is from the
     truth. The synodic period is converted to sidereal using the fitted planet
     period.
   - **amplitude error (%)** — how far the recovered wobble amplitude is from the
     truth.

---

## 3. From one trial to a survey

Because there's noise, one trial is a single random draw. `survey.run_survey` runs
**many trials per grid cell** across a grid of moon mass × moon semimajor axis, in
parallel, and aggregates each cell by the **median** of the three numbers plus a
**detection fraction** — the fraction of trials that passed all three cuts
(`sig > chsq_thr`, `perr < perr_thr`, `amperr < amp_thr`). The detection-fraction
map is the headline science result: where in (mass, separation) space is a moon
detectable at this precision?

---

## 4. Reading the maps (`plots.plot_survey`)

Four panels vs. moon semimajor axis (x) and moon mass (y):

- **Significance** — grows with moon mass and separation (bigger lever-arm → bigger
  wobble). Shown on a log scale.
- **Period error** and **amplitude error** — small where the moon is detected.
- **Detection fraction** — 0 (not detected) to 1 (always detected). The boundary is
  the science takeaway.

### Features you will see
- A **short-separation cutoff** (small `a`): the moon's period gets so short that
  the fixed phase-fold window smears it, hurting amplitude recovery.
- A **long-separation cutoff**: if the moon's *synodic* period exceeds `pend` (the
  top of the search grid), it can't be found. Raising `pend` extends the reach.
- **Resonance gaps**: near a mean-motion commensurability with the host (e.g. the
  4:1, where the moon's synodic period ≈ a harmonic of the planet's orbital
  period), the moon signal blends with imperfectly-removed planet-orbit power and
  detection drops in a narrow stripe. This is real physics, not an artifact.

---

## 5. Parameters that matter most

| parameter | what it controls |
|---|---|
| `astrometric_precision` | per-measurement noise (arcsec). 1e-5 = 10 µas. The single biggest lever on detectability. |
| `duration_days` | campaign length. Longer baseline → tighter orbit fit and more moon cycles. |
| `moon_a`, `moon_mass` | the grid axes — the moon we're testing. |
| `planet_a`, `planet_mass`, `star_mass`, `system_distance` | the host system. |
| `pstart`, `pend` | synodic-period search range (days). `pend` sets the largest detectable separation. |
| `ptestwidth` | period-grid fineness (fractional step). Smaller = finer + slower. |
| `binwidth`, `window` | phase-fold bin size and averaging window (days). |

---

## 6. A note on fidelity to the original IDL

This is a modernized port. It reproduces the IDL's **detection maps** and
**period recovery** (validated cell-by-cell). Two deliberate improvements: the
orbit fit is done in Cartesian (x, y) with a compact Levenberg–Marquardt (smoother
and tighter-converging than the original ρ,θ Newton–Raphson), and the fixed-period
sine fit is solved exactly by linear least squares. Because the fit is tighter, the
absolute significance runs a few times higher than the IDL for the same signal —
the port is simply more sensitive. Results are not bit-for-bit identical to the IDL
(different RNG and fitter); validate statistically, by distribution and map shape.
