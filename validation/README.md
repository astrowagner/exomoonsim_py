# Validation against the original IDL

This directory documents how the Python port was checked against the original IDL
`exomoonsim`. It is provenance, not part of the package.

## Tooling

- `dump_cubes.pro` — IDL procedure. Restores a survey `moon_mass_a_test.sav` and
  writes its cubes (significance, period error, amplitude error, detection
  fraction) to a plain-text file.
- `compare_idl_py.py` — reads that text dump and a Python `survey.npz` and prints a
  cell-by-cell comparison plus a detection-map agreement tally (best for small grids).
- `compare_maps.py` — same inputs, but for large grids: prints summary agreement
  statistics and writes a side-by-side figure (detection fraction and significance,
  IDL vs Python vs difference). Used for the full 9×101 production grid.

## How it was run

```idl
; in IDL: run the same grid, then dump the cubes to text
exomoons_par, /fast, precision=1d-5, subdir='py_compare'
dump_cubes, '<rundir>/moon_mass_a_test.sav'     ; writes idl_cubes_dump.txt
```

```bash
# in Python: same grid, then compare
exomoon-survey --fast --ntrials 10 --precision 1e-5 --output survey.npz
python validation/compare_idl_py.py idl_cubes_dump.txt survey.npz
```

## Result

On the shared 3×3 grid (matched configuration):

- **Detection maps agree cell-for-cell.**
- **Period recovery matches** (e.g. 0.0147% for a=10 R_Jup, mass=1 M_Earth —
  identical across trials; it's set by the period-grid resolution).
- **Amplitude error matches** once the campaign duration is aligned.
- **Absolute significance runs a few times higher in Python**, traced to the
  tighter-converging Cartesian orbit fit (it leaves a cleaner residual). Same
  signal, same period — the port is simply more sensitive. See `docs/guide.md` §6.

### Full-grid 50 µas run (9 a × 101 mass, 50 trials/cell)

Re-run on the full production grid at 50 µas (`exomoons_par, precision=5d-5` in
IDL; `run_50muas.py` in Python), then compared with `compare_maps.py`:

- **Detection maps agree on 95.6%** of cells (869/909). The disagreements lie on
  the detection boundary (mean detfrac ≈ 0.5 on both sides); only 15 cells differ
  by more than 0.5 — the expected 1/ntrials quantization plus the significance
  offset flipping marginal cells along the threshold contour.
- **Period recovery essentially identical:** 0.0252% (IDL) vs 0.0249% (Python)
  median, where both detect.
- **Amplitude error agrees to < 1%:** 11.0% vs 11.7%.
- **Significance is a uniform ~1.49× higher in Python** (16–84% spread
  [1.44, 1.53]) — the constant tighter-fit offset, now pinned down across the whole
  grid (the 3×3 spot-check gave ~1.37).
