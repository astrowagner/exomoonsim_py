#!/usr/bin/env python3
"""
Single-trial diagnostics -- a walkthrough of ONE simulated observing campaign.

The survey tools (run_survey / plot_survey) show *detectability across a grid*.
This example opens up a *single* trial so you can see the machinery that produces
each grid cell:

  1. the planet's noisy on-sky track and the stellar orbit we fit to it;
  2. the separation residual left after subtracting that orbit -- this is the
     moon's signal;
  3. the period search (periodogram) that locates the moon's synodic period;
  4. the phase-folded residual at that period, with the recovered sine.

Run it (from anywhere, once the package is installed):

    python single_trial_diagnostics.py

It prints the key recovered quantities and saves single_trial.png.
"""
import numpy as np

from exomoonsim.sim import SimParams, run_trial
from exomoonsim.plots import plot_trial


# A clearly detectable moon: a few Earth masses, well separated, at 10 muas.
# (Try nudging moon_mass down or astrometric_precision up to watch the signal
#  sink into the noise -- the periodogram peak shrinks and the phase-fold blurs.
#  Once nothing clears the detection cut the figure says so explicitly: the
#  residual panels are labeled "Nothing Recovered, Unchanged".)
#
# We set moon_ecc=0 here so the figure shows one clean recovery. The default
# (moon_ecc=0.05) is worth trying too: the fitted model is a CIRCULAR reflex, so
# an eccentric moon leaves its second harmonic in the residual, and for a signal
# this strong the search claims that harmonic -- at exactly half the moon's
# period -- as an extra "moon". You will see "1 Moon(s) In, 3 Recovered", with
# the spurious claims visible in the "Recovered System" panel. That is a real
# effect, not a bug; see the eccentricity discussion in Paper III (and
# tutorials/03_characterization.py).
params = SimParams(
    moon_a=20.0,                 # moon semimajor axis, Jupiter radii
    moon_mass=3.0,               # moon mass, Earth masses
    moon_ecc=0.0,                # circular orbit (default 0.05 -- see note above)
    astrometric_precision=1e-5,  # 10 micro-arcseconds per epoch
)

# return_diagnostics=True keeps the intermediate arrays (track, residual,
# periodogram, phase-fold) in r["diag"] alongside the usual scalar results.
r = run_trial(params, seed=1, return_diagnostics=True)

print("true sidereal moon period : %8.3f d" % r["moon_period"])
print("recovered sidereal period : %8.3f d   (error %.3f%%)"
      % (r["sidereal_period"], r["perr"]))
print("input wobble amplitude    : %8.4f mas" % r["input_amp"])
print("recovered amplitude       : %8.4f mas   (error %.1f%%)"
      % (r["retrieved_amp"], r["amperr"]))
print("detection significance    : %8.1f" % r["sig"])

# The four-panel figure. plot_trial takes the r["diag"] dict.
plot_trial(r["diag"], filename="single_trial.png")
print("\nsaved single_trial.png")
