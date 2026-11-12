"""Checks that the explainers keep the promises their Feynman cards make."""
import numpy as np
import pandas as pd
import pytest

from xai_attrition.data import ACTIONABLE_FEATURES, FEATURES, load_data, prepare
from xai_attrition.explainers import (CounterfactualExplainer, GlobalSurrogate,
                                      IntegratedGradientsExplainer, LimeExplainer, ShapleyExplainer)
from xai_attrition.model import train_ann
from xai_attrition.trust import deletion_curve


@pytest.fixture(scope="module")
def setup():
    df, src = load_data(n_simulated=3000, seed=1)
    data = prepare(df, src, seed=1)
    model = train_ann(data, hidden=(16, 8), seed=1)
    bg = data.X_train.sample(60, random_state=1).reset_index(drop=True)
    return data, model, bg


def test_hand_written_gradient_matches_finite_differences(setup):
    data, model, _ = setup
    Z = data.preprocessor.transform(data.X_test.head(5))
    logit, grad = model.logit_and_grad(Z)
    eps = 1e-5
    for j in range(0, Z.shape[1], 5):
        Zp = Z.copy()
        Zp[:, j] += eps
        num = (model.logit_and_grad(Zp)[0] - logit) / eps
        np.testing.assert_allclose(grad[:, j], num, atol=1e-3)


def test_logit_matches_sklearn_probability(setup):
    data, model, _ = setup
    Z = data.preprocessor.transform(data.X_test.head(20))
    logit, _ = model.logit_and_grad(Z)
    np.testing.assert_allclose(1 / (1 + np.exp(-logit)), model.mlp.predict_proba(Z)[:, 1], atol=1e-8)


def test_shapley_values_add_up_to_prediction(setup):
    data, model, bg = setup
    e = ShapleyExplainer(model.predict_proba, bg, n_permutations=32).explain(data.X_test.iloc[0])
    assert abs(e.base_value + e.contributions.sum() - e.prediction) < 1e-9
    assert e.prediction == pytest.approx(model.predict_proba(data.X_test.iloc[[0]])[0])


def test_integrated_gradients_completeness(setup):
    data, model, bg = setup
    e = IntegratedGradientsExplainer(model, bg, steps=256).explain(data.X_test.iloc[3])
    assert abs(e.diagnostics["completeness_gap"]) < 0.05
    assert list(e.contributions.index) == FEATURES


def test_lime_returns_one_weight_per_feature(setup):
    data, model, bg = setup
    e = LimeExplainer(model.predict_proba, bg, n_samples=500).explain(data.X_test.iloc[2])
    assert list(e.contributions.index) == list(bg.columns)
    assert 0 <= e.diagnostics["local_r2"] <= 1


def test_counterfactual_only_touches_actionable_features_and_lowers_risk(setup):
    data, model, _ = setup
    p = model.predict_proba(data.X_test)
    x = data.X_test.iloc[int(np.argmax(p))]
    cf = CounterfactualExplainer(model, data.X_train).explain(x)
    assert cf.p_after < cf.p_before
    derived = {"Avg_Utilization_Ratio"}          # recomputed from balance, by design
    for f in set(FEATURES) - set(ACTIONABLE_FEATURES) - derived:
        assert cf.counterfactual[f] == x[f]


def test_surrogate_is_faithful_and_produces_rules(setup):
    data, model, _ = setup
    s = GlobalSurrogate(model, max_depth=3).fit(data.X_train, data.X_test)
    assert s.fidelity > 0.75
    assert s.rules and all(isinstance(c, str) for c in s.rules[0]["conditions"])


def test_deletion_curve_endpoints(setup):
    data, model, bg = setup
    e = ShapleyExplainer(model.predict_proba, bg, n_permutations=16).explain(data.X_test.iloc[0])
    curve = deletion_curve(model.predict_proba, e, bg)
    assert len(curve) == len(FEATURES) + 1
    assert curve[0] == pytest.approx(e.prediction)
    assert isinstance(curve, np.ndarray) and not np.isnan(curve).any()
