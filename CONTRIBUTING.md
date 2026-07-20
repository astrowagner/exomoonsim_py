# Developing exomoonsim

A short orientation for anyone extending the code.

## Setup

```bash
git clone <repo-url>
cd exomoonsim
pip install -e ".[test]"     # editable install + test deps
pytest -q                    # should pass in a few seconds
```

## How the code is organized

The pipeline flows bottom-up; read the modules in this order:

1. `constants.py` — units and physical constants.
2. `orbits.py` — Kepler solver + turning orbital elements into on-sky (x, y).
3. `fit.py` — the orbit fit (Levenberg–Marquardt) and the fixed-period sine fit.
4. `sim.py` — `SimParams` (all configuration) and `run_trial` (one campaign).
5. `survey.py` — `run_survey` runs many trials over a grid, in parallel.
6. `plots.py` / `cli.py` — the figure and the command-line entry point.

`docs/guide.md` explains the science and the algorithm; start there.

## Conventions

- **Pure numpy + matplotlib** — no other runtime dependencies. Keep it that way
  unless there's a strong reason (it keeps install trivial and the code portable).
- Angles are in **degrees** in the public API (elements, inclinations), converted to
  radians internally. Distances are in the units noted per field (au, R_Jup, arcsec).
- New physics goes in `orbits.py`/`fit.py`; new configuration is a field on the
  `SimParams` dataclass and flows through the survey automatically.
- Add a test in `tests/` for anything non-trivial. Keep tests fast (see
  `tests/conftest.py:fast_params`).

## Testing

```bash
pytest -q
```

CI runs the suite on Python 3.9–3.12 (see `.github/workflows/tests.yml`).

## Validating against the original IDL

`validation/` holds the tooling used to check the port against the original IDL
code (detection maps and period recovery match; see `validation/README.md`).
