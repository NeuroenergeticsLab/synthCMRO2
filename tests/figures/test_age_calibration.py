"""Figure 4b: age calibration of synthetic CMRO2 from out-of-fold training predictions."""

import numpy as np
import pandas as pd
import pytest

from synthcmro2.figures.stats import age_bias_calibration, age_slopes, apply_age_calibration


def test_age_calibration_from_training_removes_the_age_dependent_error():
    def cohort(seed, n):
        rng = np.random.default_rng(seed)
        age = rng.uniform(20, 85, n)
        cmro2 = 160 - 0.5 * age + 10 * rng.normal(size=n)
        synth = cmro2 + 20 - 0.2 * age + 3 * rng.normal(size=n)  # error = 20 - 0.2 x age
        return pd.DataFrame({"age": age, "CMRO2": cmro2, "synthCMRO2": synth})

    train, test = cohort(1, 400), cohort(2, 400)
    fit = age_bias_calibration(train)
    assert fit["a"] == pytest.approx(20, abs=1.5) and fit["b"] == pytest.approx(-0.2, abs=0.03)
    test["calibrated"] = apply_age_calibration(test["synthCMRO2"], test["age"], fit)  # no analytical values used
    table = age_slopes(test, {"analytical": "CMRO2", "synthetic": "synthCMRO2", "calibrated": "calibrated"},
                       "CMRO2", n_bootstrap=500).set_index("variable")  # fmt: skip
    assert table.loc["analytical", "slope"] == pytest.approx(-5, abs=1)  # per decade
    assert table.loc["synthetic - reference", "slope"] == pytest.approx(-2, abs=0.5)
    assert table.loc["synthetic - reference", "p"] < 0.001
    assert abs(table.loc["calibrated - reference", "slope"]) < 0.5
    lo, hi = table.loc["synthetic - reference", ["ci_low", "ci_high"]]
    assert lo < table.loc["synthetic - reference", "slope"] < hi


def test_age_slopes_adjusted_matches_multiple_regression():
    import numpy as np
    import pandas as pd

    from synthcmro2.figures.stats import age_slopes_adjusted

    rng = np.random.default_rng(0)
    n = 60
    age = rng.uniform(20, 80, n)
    slope = 1.0 - 0.004 * age + rng.normal(0, 0.05, n)
    ana = 150 - 0.5 * age + rng.normal(0, 5, n)
    syn = ana - 0.2 * age + 30 * slope + rng.normal(0, 3, n)
    df = pd.DataFrame({"age": age, "slope": slope, "CMRO2": ana, "synthCMRO2": syn})
    out = age_slopes_adjusted(df, {"analytical": "CMRO2", "synthetic": "synthCMRO2"}, "CMRO2", "slope", n_bootstrap=200)
    design = np.column_stack([np.ones(n), age, slope])
    expected = 10 * np.linalg.lstsq(design, syn - ana, rcond=None)[0][1]
    excess = out[out["kind"] == "excess"].iloc[0]
    assert np.isclose(excess["slope"], expected)
    assert excess["ci_low"] < excess["slope"] < excess["ci_high"]
    assert 0 <= excess["p"] <= 1
