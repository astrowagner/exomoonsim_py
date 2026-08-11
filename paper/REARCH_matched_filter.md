# Paper III re-architecture: matched filter as the primary detector

Goal: make the projected-ellipse matched filter the primary detection statistic throughout the
paper (single sinusoid kept as the Papers I/II baseline for comparison). Detection now uses the
same projected-ellipse model that already fits, subtracts, and characterizes each moon.

Every ellipse run needs its detection cut **calibrated** to the single-sinusoid Δχ²>5 false-alarm
rate — this is automatic inside each driver via `paper/ellipse_cut.py` (runs zero-mass control
trials in the exact search config, sets the ellipse threshold to the matched-FAR quantile). The
threshold depends on the period grid (pend, ptestwidth, cadence) but not on precision, so it is
calibrated once per search config.

## Heavy runs (multicore machine, in any order)

| # | Command | Output | Feeds manuscript |
|---|---|---|---|
| 1 | `python run_50muas_ellipse.py` | `runs/survey_50muas_ntrials50_ellipse.npz`, `runs/ellipse_threshold_50uas.txt` | detection maps, false-positive fraction (§4.4), significance vs sine |
| 2 | `python paper/make_precision_scaling_fig.py --stat ellipse` | `paper/data/precision_scaling_ellipse.csv`, `paper/figs/precision_scaling_ellipse.png` | §5 precision floor **and the separation-lever reframe** (weak→strong under matched filter) |
| 3 | `python paper/fourmoon_ellipse.py` | prints the new four-moon recovery table | §4.2 / Table 3 (does the matched filter clear >1 of 4 at 50 µas?) |

Notes
- All three self-calibrate; `--nctrl 300 --workers N` are available (defaults: 300 control trials,
  all cores). Calibration adds 2×nctrl control trials on top of the survey itself.
- Run 1 is the big one (45,450 ellipse trials + control); expect it to be the ellipse cost (~10–50×
  a sine trial) times the sine survey wall-time.
- After run 1, the existing figure scripts regenerate from the ellipse npz by pointing them at it:
  `make_false_positive_fig.py`, `make_fp_vs_threshold.py` (change their input npz to the `_ellipse`
  file), and `make_matched_filter_map_fig.py` already exists for the sine-vs-ellipse map.

## Already in hand (no re-run)
- `figs/matched_filter.png` (Fig 14): a²-scaling + min-mass, matched filter vs sine.
- `figs/matched_filter_fit.png` (Fig 2): the fit mechanism, two moon periods.
- `figs/matched_filter_map.png`: 10 µas sine-vs-ellipse detection map (supplementary).

## Preliminary probe (this session, indicative only — not calibrated-search numbers)
Per-moon significance at each moon's true period on the raw 50 µas four-moon residual:

| a (R_Jup) | P_syn (d) | sine Δχ² | ellipse Δχ² (raw) |
|---|---|---|---|
| 20 | 20.6 | 9.8 | 299 |
| 12 | 9.5 | 3.2 | 53 |
| 30 | 38.7 | 3.8 | 165 |
| 8 | 5.1 | 3.8 | 26 |

Against the ~27 calibrated ellipse cut this suggests ~3 of 4 clear at 50 µas (a=8 borderline),
versus 1 of 4 for the single sinusoid — run 3 pins this down with the proper calibrated search.

## Then (me): manuscript restructure
Once the data files/numbers are back:
- Promote the ellipse statistic to the primary detector in §2.2 (recovery loop) and §3.
- Reframe §4.1/§4.2 results and Table 3 with the matched-filter significances.
- Recompute §4.4 false positives from the ellipse survey.
- Flip the §5 separation-lever finding (weak a^-1/4 → strong ~a^2) using run 2; move fig:matched up.
- Keep the single sinusoid as the labeled Papers I/II baseline; appendix validation stays sine.
