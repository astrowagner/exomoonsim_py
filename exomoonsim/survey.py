"""
Parameter-grid survey driver (parallel).

Runs the single-trial simulator over a grid of (moon mass, moon semimajor axis),
distributing independent trials across a multiprocessing pool, and aggregates each
cell by the median (significance, period error, amplitude error) and the detection
fraction.  This is the Python equivalent of exomoons_par.pro -- but with ordinary
multiprocessing instead of IDL_IDLBridge, so there is no pool-teardown or bridge
bookkeeping to worry about.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import numpy as np
from multiprocessing import Pool

from .sim import SimParams, run_trial

# default grids (match exomoons.pro)
DEFAULT_MASS = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                         0.1, 0.125, 0.15, 0.2])
DEFAULT_A = np.array([5, 6, 7, 8, 9, 10, 20, 50, 80, 90, 100, 110, 120], float)
FAST_MASS = np.array([0.1, 1.0, 5.0])
FAST_A = np.array([5, 10, 20], float)


@dataclass
class SurveyConfig:
    mass_grid: np.ndarray = field(default_factory=lambda: DEFAULT_MASS.copy())
    a_grid: np.ndarray = field(default_factory=lambda: DEFAULT_A.copy())
    ntrials: int = 50
    base: SimParams = field(default_factory=SimParams)
    chsq_thr: float = 5.0
    perr_thr: float = 5.0
    amp_thr: float = 25.0
    nworkers: int = 0            # 0 -> os.cpu_count()
    base_seed: int = 12345


@dataclass
class SurveyResult:
    a_grid: np.ndarray
    mass_grid: np.ndarray
    sigcube: np.ndarray         # [na, nm] median significance
    perrcube: np.ndarray        # [na, nm] median period error (%)
    amperrcube: np.ndarray      # [na, nm] median amplitude error (%)
    detfrac: np.ndarray         # [na, nm] detection fraction
    trials: np.ndarray          # [na, nm, ntrials, 3] raw (sig, perr, amperr)
    ntrials: int
    config: SurveyConfig

    def save(self, path):
        np.savez_compressed(
            path, a_grid=self.a_grid, mass_grid=self.mass_grid,
            sigcube=self.sigcube, perrcube=self.perrcube,
            amperrcube=self.amperrcube, detfrac=self.detfrac,
            trials=self.trials, ntrials=self.ntrials,
            chsq_thr=self.config.chsq_thr, perr_thr=self.config.perr_thr,
            amp_thr=self.config.amp_thr,
        )

    @staticmethod
    def load(path):
        d = np.load(path, allow_pickle=True)
        cfg = SurveyConfig(chsq_thr=float(d["chsq_thr"]),
                           perr_thr=float(d["perr_thr"]),
                           amp_thr=float(d["amp_thr"]))
        return SurveyResult(
            a_grid=d["a_grid"], mass_grid=d["mass_grid"], sigcube=d["sigcube"],
            perrcube=d["perrcube"], amperrcube=d["amperrcube"], detfrac=d["detfrac"],
            trials=d["trials"], ntrials=int(d["ntrials"]), config=cfg)


def _run_one(args):
    """Worker: run one trial for a given cell.  Top-level for pickling."""
    base_dict, moon_a, moon_mass, seed = args
    p = SimParams(**base_dict)
    p.moon_a = float(moon_a)
    p.moon_mass = float(moon_mass)
    r = run_trial(p, seed=seed)
    return r["sig"], r["perr"], r["amperr"]


def run_survey(config: SurveyConfig = None, progress=True):
    """Run the full grid survey in parallel; return a SurveyResult."""
    cfg = config or SurveyConfig()
    na, nm = cfg.a_grid.size, cfg.mass_grid.size
    base_dict = dataclasses.asdict(cfg.base)

    # build flat work list
    work = []
    idx = []
    k = 0
    for im, mass in enumerate(cfg.mass_grid):
        for ia, a in enumerate(cfg.a_grid):
            for it in range(cfg.ntrials):
                work.append((base_dict, a, mass, cfg.base_seed + k))
                idx.append((ia, im, it))
                k += 1

    nproc = cfg.nworkers or None
    results = np.empty((len(work), 3))
    done = 0
    with Pool(processes=nproc) as pool:
        for i, res in enumerate(pool.imap(_run_one, work, chunksize=1)):
            results[i] = res
            done += 1
            if progress and (done % max(1, len(work) // 20) == 0 or done == len(work)):
                print(f"  {done}/{len(work)} trials", flush=True)

    trials = np.full((na, nm, cfg.ntrials, 3), np.nan)
    for (ia, im, it), res in zip(idx, results):
        trials[ia, im, it] = res

    sig = np.nanmedian(trials[..., 0], axis=2)
    perr = np.nanmedian(trials[..., 1], axis=2)
    amperr = np.nanmedian(trials[..., 2], axis=2)
    passed = ((trials[..., 0] > cfg.chsq_thr)
              & (trials[..., 1] < cfg.perr_thr)
              & (trials[..., 2] < cfg.amp_thr))
    detfrac = np.mean(passed, axis=2)

    return SurveyResult(a_grid=cfg.a_grid, mass_grid=cfg.mass_grid, sigcube=sig,
                        perrcube=perr, amperrcube=amperr, detfrac=detfrac,
                        trials=trials, ntrials=cfg.ntrials, config=cfg)
