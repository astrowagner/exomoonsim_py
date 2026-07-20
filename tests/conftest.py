"""Shared test fixtures/helpers.

A ``fast_params`` factory builds a SimParams that runs a trial in well under a
second (short campaign, sparse cadence, coarse period grid) -- enough to exercise
the whole pipeline in the test suite without waiting on a full-resolution run.
"""
from exomoonsim.sim import SimParams


def fast_params(**overrides):
    kw = dict(
        duration_days=365.25,   # 1-year campaign
        texp=6.0,               # one sample every 6 hours -> ~1460 points
        astrometric_precision=1e-4,
        pstart=1.0,
        pend=40.0,              # narrow synodic search
        ptestwidth=0.01,        # coarse period grid
    )
    kw.update(overrides)
    return SimParams(**kw)
