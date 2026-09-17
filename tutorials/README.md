# exomoonsim tutorials

Four hands-on walkthroughs built from the Paper III figures and tests, in
recommended order. Each exists as a **Jupyter notebook** (`.ipynb`, cell-by-cell
with inline plots) and an identical **Python script** (`.py`, runnable from the
terminal and easy to adapt for your own runs).

| # | Tutorial | What you learn | Paper figures | FAST runtime |
|---|---|---|---|---|
| 1 | `01_signal_and_matched_filter` | What the moon signal looks like, the planet-motion sidebands, and why the projected-ellipse **matched filter** beats a single sinusoid | Fig. 2 | ~1 min |
| 2 | `02_multi_moon_recovery` | Calibrating the matched-filter cut, iterative prewhitening on a two-moon system, and the four-moon sinusoid-vs-matched-filter comparison | Figs. 3–6, Table 3 | ~5 min |
| 3 | `03_characterization` | The mass–inclination degeneracy, how the ellipse fit breaks it (mass **and** inclination), the recovered reflex ellipse, and the eccentricity limit | Fig. 1, Table 4, Fig. 12 | ~2 min |
| 4 | `04_surveys_and_detection_maps` | Running a parallel survey, detection ("success") maps, false positives, the mass floor, and switching a survey to the matched filter | Figs. 8–9, 13 | ~3 min |

## Setup

From the repository root:

```bash
pip install -e .            # or: export PYTHONPATH=$PWD
python tutorials/01_signal_and_matched_filter.py        # script
jupyter notebook tutorials/01_signal_and_matched_filter.ipynb   # notebook
```

Only `numpy` and `matplotlib` are needed (plus Jupyter for the notebooks).

## The `FAST` switch

Every tutorial starts with `FAST = True`. That selects a laptop-friendly
configuration (coarser cadence and period grid, small grids, fewer trials) so the
whole tutorial runs in minutes and every result is qualitatively the paper's.
Set `FAST = False` to use the **exact configuration of the corresponding paper
figure**; each tutorial notes how long that takes (Tutorial 4's full survey grid
runs for hours and is normally run via `run_50muas.py` instead).

A few FAST results differ from the paper in the honest way you'd expect from
coarser sampling — e.g. in Tutorial 2 the marginal fourth moon (Δχ²_mf ≈ 29 against a
cut of 28.9 at the paper's hourly cadence) falls just below the cut at 6-hour cadence.
The tutorials point these out rather than hide them.

## Multiprocessing on macOS (read this once)

Tutorial 4 (and any survey script) runs trials in parallel worker processes. On
macOS and Windows, Python starts workers with the "spawn" method, which
**re-imports the running script** in every worker. A script that calls
`run_survey()` at the top level therefore has each worker relaunch the survey and
fails with:

```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

The fix is the standard guard — put the executable code under

```python
if __name__ == "__main__":
    ...
```

The `.py` tutorials do this; in the notebooks it isn't needed (the kernel is
already `__main__`) and is omitted. Tutorials 1–3 use no worker processes.

## Regenerating the notebooks

The notebooks are generated from the scripts, so edit the `.py` and rebuild:

```bash
python tutorials/_make_notebooks.py
```

Scripts use light cell markers — `# %%` for a code cell and `# %% [markdown]`
followed by `# ` comment lines for prose. The module docstring becomes the title cell.

## Where to go next

* `examples/getting_started.py` — the shortest end-to-end run (single trial + tiny survey).
* `run_sumin_survey.py` — the student-project driver: a mass × separation success map at a chosen precision.
* `paper/` — the scripts that reproduce every computed figure and table of Paper III (`paper/README.md`).
* `docs/guide.md` — the physics and algorithm guide.
