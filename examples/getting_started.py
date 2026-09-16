"""
Getting started with exomoonsim — a commented, end-to-end walkthrough.

Run it from the repository root (after `pip install -e .`):

    python examples/getting_started.py

It (1) runs a single detection trial and prints the result, (2) runs a small
parallel survey over a grid of moon mass and separation, and (3) saves the
four-panel figure. Everything here is also available in notebooks/example.ipynb.

NOTE — the ``if __name__ == "__main__":`` guard below is REQUIRED, not optional.
Step (2) runs trials in parallel with multiprocessing, and on macOS and Windows
Python starts worker processes with the "spawn" method, which re-imports this
file in every worker. Without the guard, each worker would re-run the survey and
try to spawn its own workers, raising:

    RuntimeError: An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.

The guard makes the top-level code run only in the parent process. Any script of
your own that calls run_survey() needs the same guard.
"""
import numpy as np

from exomoonsim.sim import SimParams, run_trial
from exomoonsim.survey import SurveyConfig, run_survey
from exomoonsim.plots import plot_survey


def main():
    # ----------------------------------------------------------------------- #
    # 1) A SINGLE TRIAL
    # ----------------------------------------------------------------------- #
    # SimParams holds the whole configuration: the host system, the moon we're
    # testing, the observing setup, and the period-search settings. Anything you
    # don't set keeps its default (see exomoonsim/sim.py or docs/guide.md).
    params = SimParams(
        moon_a=10.0,                 # moon semimajor axis, in Jupiter radii
        moon_mass=1.0,               # moon mass, in Earth masses
        astrometric_precision=1e-5,  # 1e-5 arcsec = 10 micro-arcseconds per measurement
    )

    # run_trial simulates one observing campaign and tries to detect the moon.
    # `seed` makes it reproducible.
    result = run_trial(params, seed=1)

    print("Single trial (Earth-mass moon at 10 R_Jup, 10 muas):")
    print(f"  significance      = {result['sig']:.1f}")
    print(f"  period error (%)  = {result['perr']:.4f}")
    print(f"  amplitude err (%) = {result['amperr']:.1f}")
    print(f"  recovered period  = {result['sidereal_period']:.3f} d "
          f"(true {result['moon_period']:.3f} d)")

    # ----------------------------------------------------------------------- #
    # 2) A SMALL SURVEY  (runs in parallel — needs the __main__ guard, see above)
    # ----------------------------------------------------------------------- #
    # A survey runs many trials per grid cell (in parallel) and aggregates them.
    # Here we use a small 3x3 grid and few trials so it finishes quickly; drop the
    # a_grid/mass_grid arguments (and raise ntrials) for the full default survey.
    config = SurveyConfig(
        a_grid=np.array([5.0, 10.0, 20.0]),     # moon semimajor axes (R_Jup)
        mass_grid=np.array([0.1, 1.0, 5.0]),    # moon masses (M_Earth)
        ntrials=6,                              # trials per cell
        base=SimParams(astrometric_precision=1e-5),
    )
    survey = run_survey(config)                 # uses all CPU cores by default
    survey.save("example_survey.npz")

    print("\nDetection fraction (rows = semimajor axis, cols = mass):")
    print(survey.detfrac)

    # ----------------------------------------------------------------------- #
    # 3) THE FIGURE
    # ----------------------------------------------------------------------- #
    # plot_survey returns a matplotlib Figure and (optionally) saves it.
    plot_survey(survey, filename="example_survey.png")
    print("\nSaved example_survey.npz and example_survey.png")


if __name__ == "__main__":
    main()
