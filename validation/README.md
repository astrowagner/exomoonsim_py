# Validation against the original IDL

This directory documents how the Python port was checked against the original IDL
`exomoonsim`. It is provenance, not part of the package.

## Tooling

- `dump_cubes.pro` — IDL procedure. Restores a survey `moon_mass_a_test.sav` and
  writes its cubes (significance, period error, amplitude error, detection
  fraction) to a plain-text file.
- `compare_idl_py.py` — reads that text dump and a Python `survey.npz` and prints a
  cell-by-cell comparison plus a detection-map agreement tally.

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
