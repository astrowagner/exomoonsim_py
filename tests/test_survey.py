"""Tests for the parallel survey driver + save/load."""
import numpy as np
from exomoonsim.survey import SurveyConfig, SurveyResult, run_survey
from conftest import fast_params


def test_small_survey_shapes_and_ranges(tmp_path):
    cfg = SurveyConfig(
        a_grid=np.array([10.0, 20.0]),
        mass_grid=np.array([1.0, 5.0]),
        ntrials=2,
        base=fast_params(),
        nworkers=1,        # keep the test single-process and deterministic
    )
    res = run_survey(cfg, progress=False)

    assert res.sigcube.shape == (2, 2)
    assert res.detfrac.shape == (2, 2)
    assert res.trials.shape == (2, 2, 2, 3)
    assert np.all((res.detfrac >= 0) & (res.detfrac <= 1))

    # save / load round-trip
    path = tmp_path / "s.npz"
    res.save(path)
    back = SurveyResult.load(path)
    assert np.allclose(back.sigcube, res.sigcube, equal_nan=True)
    assert back.ntrials == res.ntrials
