# Reproducing the Paper III figures and tables

Everything needed to regenerate the computed figures and tables of Paper III lives in
this directory. The scripts import the `exomoonsim` package from this repository
(one level up), so the simulation code is never duplicated. See `provenance.txt` for
the commit and library versions the published figures were made with.

## Setup

```bash
pip install -r requirements.txt          # numpy + matplotlib only
```

`exomoonsim` itself does not need to be installed: each script puts the repository
root on `sys.path`, so `import exomoonsim` resolves to the package in this repo.
(Equivalently, `pip install -e .` from the repo root also works.)

## Scripts

Run from this directory with plain `python`. Figures are written to `paper/figs/`;
copy them into the manuscript's `figs/` directory to rebuild the PDF.

| script | produces | inputs | runtime |
|---|---|---|---|
| `make_recovery_fig.py` | `figs/recovery_summary.png` (Fig. 1) **and** the Table 3 rows + Sec. 4.2 residual-RMS numbers | `exomoonsim`, `seed=11` | ~30 s |
| `make_two_moon_figs.py` | `figs/two_moon_{context,residuals,periodograms,phasefold}.png` (Figs. 2-5) **and** the Table 2 rows | `exomoonsim`, `seed=42` | ~30 s |
| `make_four_moon_figs.py` | `figs/four_moon_{summary,phasefold}.png` (Figs. 6-7) **and** the Table 3 rows | `exomoonsim`, `seed=7` | ~20 s |
| `make_confusion_fig.py` | `figs/confusion_vs_n.png` (Fig. 10, multiple-moon robustness) | `exomoonsim`, `seed=7` | ~1-2 min |
| `make_false_positive_fig.py` | `figs/false_positive.png` (Fig. 3) | `data/survey_50muas_ntrials50.npz` | ~2 s |
| `make_fp_vs_threshold.py` | `figs/fp_vs_threshold.png` (Fig. 4) + threshold-sweep table | `data/survey_50muas_ntrials50.npz` | ~2 s |
| `make_validation_fig.py` | `figs/idl_vs_python.png` (Appendix) + agreement stats | `data/idl_cubes_dump.txt`, `data/survey_50muas_ntrials50.npz` | ~3 s |

`make_recovery_fig.py` and `make_two_moon_fig.py` run the simulation directly (fixed
seeds). The three survey-based scripts read only committed data, so they reproduce the
exact published figures.

## Data

- `data/survey_50muas_ntrials50.npz` — Python survey on the 9 (semimajor axis) ×
  101 (mass) grid at 50 µas, 50 trials/cell (detection, significance, period-error,
  amplitude-error, and false-positive cubes plus the raw per-trial results). It is the
  output of the repository's top-level `run_50muas.py` (the slow, 45,450-trial step),
  committed here so the figures do not require re-running it.
- `data/idl_cubes_dump.txt` — the original IDL survey's cubes on the identical grid,
  written by `validation/dump_cubes.pro`. Committed so the IDL vs Python comparison
  reproduces without an IDL license.

To regenerate the survey from scratch: `python ../run_50muas.py` (writes
`runs/survey_50muas_ntrials50.npz` at the repo root), then copy that file into
`paper/data/`.
