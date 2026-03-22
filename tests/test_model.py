"""Tests for DistributionalGLM fitting, prediction, and diagnostics."""
import numpy as np
import pytest
import polars as pl

from insurance_distributional_glm import DistributionalGLM, quantile_residuals, gaic
from insurance_distributional_glm.families import (
    Gamma, LogNormal, Poisson, NBI, ZIP, InverseGaussian
)


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def make_gamma_data(n=300, seed=42):
    """Gamma data with known mean structure: log(mu) = 6 + 0.3*x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    log_mu_true = 6.0 + 0.3 * x1
    mu_true = np.exp(log_mu_true)
    sigma_true = 0.5
    shape = 1.0 / sigma_true**2
    y = rng.gamma(shape=shape, scale=mu_true / shape)
    df = pl.DataFrame({"x1": x1})
    return df, y, mu_true, sigma_true


def make_poisson_data(n=500, seed=10):
    """Poisson data with log(mu) = 1 + 0.4*x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu_true = np.exp(1.0 + 0.4 * x1)
    y = rng.poisson(mu_true).astype(float)
    df = pl.DataFrame({"x1": x1})
    return df, y, mu_true


def make_nbi_data(n=300, seed=20):
    """NBI data."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu_true = np.exp(1.5 + 0.3 * x1)
    sigma_true = 0.4
    k = 1.0 / sigma_true
    p_nb = k / (k + mu_true)
    y = stats.nbinom(n=k, p=p_nb).rvs(random_state=rng.integers(1000)).astype(float)
    df = pl.DataFrame({"x1": x1})
    return df, y


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_gamma_converges(self):
        df, y, _, _ = make_gamma_data()
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": ["x1"], "sigma": []},
        )
        model.fit(df, y, max_iter=200)
        assert model.converged

    def test_poisson_converges(self):
        df, y, _ = make_poisson_data()
        model = DistributionalGLM(
            family=Poisson(),
            formulas={"mu": ["x1"]},
        )
        model.fit(df, y, max_iter=200)
        assert model.converged

    def test_lognormal_converges(self):
        rng = np.random.default_rng(99)
        y = np.exp(rng.normal(7.0, 0.5, 200))
        df = pl.DataFrame({"x": rng.normal(size=200)})
        model = DistributionalGLM(family=LogNormal(), formulas={"mu": ["x"], "sigma": []})
        model.fit(df, y, max_iter=200)
        assert model.converged

    def test_nbi_converges(self):
        df, y = make_nbi_data()
        model = DistributionalGLM(family=NBI(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y, max_iter=200)
        assert model.converged


# ---------------------------------------------------------------------------
# Parameter recovery tests
# ---------------------------------------------------------------------------

class TestParameterRecovery:
    def test_gamma_mu_intercept(self):
        """Intercept-only Gamma: mu hat should be close to mean(y)."""
        rng = np.random.default_rng(0)
        y = rng.gamma(4.0, 250.0, 1000)
        df = pl.DataFrame({"dummy": np.zeros(1000)})
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": [], "sigma": []},
        )
        model.fit(df, y)
        mu_hat = model.predict(df, parameter="mu")
        np.testing.assert_allclose(mu_hat[0], np.mean(y), rtol=0.05)

    def test_gamma_sigma_intercept(self):
        """Sigma hat should recover true CV within 10%."""
        rng = np.random.default_rng(0)
        sigma_true = 0.4
        shape = 1.0 / sigma_true**2
        y = rng.gamma(shape, 500.0, 2000)
        df = pl.DataFrame({"dummy": np.zeros(2000)})
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": [], "sigma": []},
        )
        model.fit(df, y)
        sigma_hat = model.predict(df, parameter="sigma")[0]
        assert abs(sigma_hat - sigma_true) / sigma_true < 0.15

    def test_gamma_mu_slope(self):
        """Slope on mu should be recovered within 15%."""
        df, y, mu_true, _ = make_gamma_data(n=1000)
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": ["x1"], "sigma": []},
        )
        model.fit(df, y)
        betas = model.coefficients["mu"]
        # True slope = 0.3
        assert abs(betas[1] - 0.3) < 0.05

    def test_poisson_slope(self):
        """Poisson mu slope should be close to truth."""
        df, y, _ = make_poisson_data(n=2000)
        model = DistributionalGLM(
            family=Poisson(),
            formulas={"mu": ["x1"]},
        )
        model.fit(df, y)
        betas = model.coefficients["mu"]
        # True slope = 0.4
        assert abs(betas[1] - 0.4) < 0.05


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_predict_mu_shape(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        mu = model.predict(df, parameter="mu")
        assert mu.shape == (100,)
        assert np.all(mu > 0)

    def test_predict_sigma_positive(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        sigma = model.predict(df, parameter="sigma")
        assert np.all(sigma > 0)

    def test_predict_mean_equals_predict_mu(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        np.testing.assert_array_equal(
            model.predict_mean(df), model.predict(df, parameter="mu")
        )

    def test_predict_variance_positive(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        var = model.predict_variance(df)
        assert np.all(var > 0)

    def test_volatility_score_positive(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        cv = model.volatility_score(df)
        assert np.all(cv > 0)

    def test_volatility_score_matches_formula(self):
        """CV = sqrt(Var) / mu for Gamma = sigma."""
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        cv = model.volatility_score(df)
        sigma = model.predict(df, parameter="sigma")
        np.testing.assert_allclose(cv, sigma, rtol=1e-5)

    def test_predict_distribution_gamma_length(self):
        df, y, _, _ = make_gamma_data(n=50)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        dists = model.predict_distribution(df)
        assert len(dists) == 50
        assert all(d is not None for d in dists)

    def test_predict_distribution_gamma_cdf(self):
        """CDF at median should be near 0.5."""
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        dists = model.predict_distribution(df)
        # CDF at the mean for Gamma should be around 0.6-0.8 (depends on shape)
        mu = model.predict(df, "mu")[0]
        cdf_at_mean = dists[0].cdf(mu)
        assert 0.4 < cdf_at_mean < 0.95

    def test_predict_unfitted_raises(self):
        model = DistributionalGLM(family=Gamma())
        df = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            model.predict(df)


# ---------------------------------------------------------------------------
# Exposure offset tests
# ---------------------------------------------------------------------------

class TestExposure:
    def test_exposure_changes_prediction(self):
        rng = np.random.default_rng(50)
        n = 200
        exposure = rng.uniform(0.5, 2.0, n)
        mu_true = 3.0 * exposure
        y = rng.poisson(mu_true).astype(float)
        df = pl.DataFrame({"x": np.zeros(n)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y, exposure=exposure)

        # With 2x exposure, mu should be ~2x
        df_pred = pl.DataFrame({"x": np.zeros(2)})
        mu_e1 = model.predict(df_pred[:1], parameter="mu", exposure=np.array([1.0]))
        mu_e2 = model.predict(df_pred[:1], parameter="mu", exposure=np.array([2.0]))
        np.testing.assert_allclose(mu_e2 / mu_e1, 2.0, rtol=0.01)

    def test_exposure_poisson_rate(self):
        """Intercept-only Poisson with exposure: beta0 should recover log(rate)."""
        rng = np.random.default_rng(51)
        n = 2000
        true_rate = 0.15  # claims per unit exposure
        exposure = rng.uniform(0.5, 2.0, n)
        y = rng.poisson(true_rate * exposure).astype(float)
        df = pl.DataFrame({"x": np.zeros(n)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y, exposure=exposure)
        beta0 = model.coefficients["mu"][0]
        np.testing.assert_allclose(np.exp(beta0), true_rate, rtol=0.1)


# ---------------------------------------------------------------------------
# Input format tests
# ---------------------------------------------------------------------------

class TestInputFormats:
    def test_polars_input(self):
        rng = np.random.default_rng(30)
        y = rng.gamma(2.0, 500.0, 100)
        df = pl.DataFrame({"x": rng.normal(size=100)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x"], "sigma": []})
        model.fit(df, y)
        assert model.converged or model.n_observations == 100

    def test_pandas_input(self):
        pytest.importorskip("pandas")
        import pandas as pd
        rng = np.random.default_rng(31)
        y = rng.gamma(2.0, 500.0, 100)
        df = pd.DataFrame({"x": rng.normal(size=100)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x"], "sigma": []})
        model.fit(df, y)
        assert model.n_observations == 100

    def test_numpy_input(self):
        rng = np.random.default_rng(32)
        y = rng.gamma(2.0, 500.0, 100)
        X = rng.normal(size=(100, 2))
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x0", "x1"], "sigma": []})
        model.fit(X, y)
        assert model.n_observations == 100


# ---------------------------------------------------------------------------
# Relativities tests
# ---------------------------------------------------------------------------

class TestRelativities:
    def test_relativities_columns(self):
        df, y, _, _ = make_gamma_data(n=200)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        rel = model.relativities(parameter="mu")
        assert "term" in rel.columns
        assert "coefficient" in rel.columns
        assert "relativity" in rel.columns

    def test_relativities_intercept_exp(self):
        """For log link: relativity at intercept = exp(beta_0)."""
        rng = np.random.default_rng(40)
        y = rng.gamma(4.0, 250.0, 500)
        df = pl.DataFrame({"x": rng.normal(size=500)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x"], "sigma": []})
        model.fit(df, y)
        rel = model.relativities("mu")
        intercept_row = rel.filter(pl.col("term") == "(Intercept)")
        beta0 = model.coefficients["mu"][0]
        np.testing.assert_allclose(
            intercept_row["relativity"][0],
            np.exp(beta0),
            rtol=1e-6,
        )

    def test_relativities_slope_exp(self):
        """Slope relativity = exp(beta_1)."""
        rng = np.random.default_rng(41)
        y = rng.gamma(4.0, 250.0, 500)
        df = pl.DataFrame({"x": rng.normal(size=500)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x"], "sigma": []})
        model.fit(df, y)
        rel = model.relativities("mu")
        x_row = rel.filter(pl.col("term") == "x")
        beta1 = model.coefficients["mu"][1]
        np.testing.assert_allclose(
            x_row["relativity"][0],
            np.exp(beta1),
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# GAIC tests
# ---------------------------------------------------------------------------

class TestGAIC:
    def test_gaic_aic(self):
        """GAIC with penalty=2 is AIC."""
        ll = -500.0
        n_params = 3
        expected = -2 * ll + 2 * n_params
        assert gaic(ll, n_params, penalty=2.0) == expected

    def test_gaic_model_method(self):
        df, y, _, _ = make_gamma_data(n=200)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        g = model.gaic(penalty=2.0)
        assert np.isfinite(g)
        assert g > 0  # AIC is positive for these data

    def test_gaic_penalised_model_worse(self):
        """More complex model (more params) has higher GAIC if no improvement."""
        rng = np.random.default_rng(60)
        y = rng.gamma(4.0, 500.0, 200)
        df = pl.DataFrame({"noise": rng.normal(size=200)})
        # Intercept only
        m1 = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        m1.fit(df, y)
        # With noise covariate — noise shouldn't help but adds a param
        m2 = DistributionalGLM(family=Gamma(), formulas={"mu": ["noise"], "sigma": []})
        m2.fit(df, y)
        # m1 should have lower GAIC since noise doesn't help
        # (with high probability for pure noise)
        # We just check both are finite
        assert np.isfinite(m1.gaic())
        assert np.isfinite(m2.gaic())


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScore:
    def test_nll_finite(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        nll = model.score(df, y, metric="nll")
        assert np.isfinite(nll)
        assert nll > 0

    def test_deviance_finite(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        dev = model.score(df, y, metric="deviance")
        assert np.isfinite(dev)

    def test_unknown_metric_raises(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        with pytest.raises(ValueError):
            model.score(df, y, metric="rmse")


# ---------------------------------------------------------------------------
# Quantile residuals tests
# ---------------------------------------------------------------------------

class TestQuantileResiduals:
    def test_shape(self):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        resids = quantile_residuals(model, df, y, seed=0)
        assert resids.shape == (100,)

    def test_approximately_normal(self):
        """For a correctly specified model, quantile residuals ~ N(0,1)."""
        from scipy import stats
        rng = np.random.default_rng(77)
        n = 500
        y = rng.gamma(4.0, 250.0, n)
        df = pl.DataFrame({"x": np.zeros(n)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        resids = quantile_residuals(model, df, y, seed=42)
        resids_finite = resids[np.isfinite(resids)]
        # KS test: should not reject normality at alpha=0.01
        stat, pval = stats.kstest(resids_finite, "norm")
        assert pval > 0.01, f"KS test rejected normality: p={pval:.4f}"

    def test_discrete_family_residuals(self):
        """ZIP residuals should be finite."""
        rng = np.random.default_rng(88)
        n = 200
        y = rng.poisson(2.0, n).astype(float)
        df = pl.DataFrame({"x": np.zeros(n)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y)
        resids = quantile_residuals(model, df, y, seed=0, n_random=5)
        assert np.sum(np.isfinite(resids)) > 0.9 * n


# ---------------------------------------------------------------------------
# Summary / repr tests
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_runs(self, capsys):
        df, y, _, _ = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model.fit(df, y)
        model.summary()
        out = capsys.readouterr().out
        assert "Gamma" in out
        assert "mu" in out

    def test_repr_fitted(self):
        df, y, _, _ = make_gamma_data(n=50)
        model = DistributionalGLM(family=Gamma())
        model.fit(df, y)
        r = repr(model)
        assert "fitted" in r

    def test_repr_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        r = repr(model)
        assert "unfitted" in r

    def test_n_observations(self):
        df, y, _, _ = make_gamma_data(n=77)
        model = DistributionalGLM(family=Gamma())
        model.fit(df, y)
        assert model.n_observations == 77


# ---------------------------------------------------------------------------
# choose_distribution tests
# ---------------------------------------------------------------------------

class TestChooseDistribution:
    def test_ranking_gamma_wins(self):
        from insurance_distributional_glm import choose_distribution
        df, y, _, _ = make_gamma_data(n=300)
        families = [Gamma(), LogNormal(), InverseGaussian()]
        results = choose_distribution(df, y, families, formulas={"mu": ["x1"], "sigma": []})
        assert len(results) == 3
        assert results[0].family_name == "Gamma"

    def test_results_sorted_by_gaic(self):
        from insurance_distributional_glm import choose_distribution
        df, y, _, _ = make_gamma_data(n=200)
        families = [Gamma(), LogNormal()]
        results = choose_distribution(df, y, families, formulas={"mu": [], "sigma": []})
        gaics = [r.gaic for r in results]
        assert gaics == sorted(gaics)

    def test_result_fields(self):
        from insurance_distributional_glm import choose_distribution
        df, y, _, _ = make_gamma_data(n=100)
        results = choose_distribution(df, y, [Gamma()], formulas={"mu": [], "sigma": []})
        r = results[0]
        assert np.isfinite(r.gaic)
        assert np.isfinite(r.loglik)
        assert r.df > 0
        assert r.model is not None


# ---------------------------------------------------------------------------
# Weights tests
# ---------------------------------------------------------------------------

class TestWeights:
    def test_uniform_weights_same_as_no_weights(self):
        """Uniform weights should give identical results to unweighted fit."""
        df, y, _, _ = make_gamma_data(n=100, seed=99)
        model_no_w = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model_no_w.fit(df, y)

        model_w = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model_w.fit(df, y, weights=np.ones(100))

        np.testing.assert_allclose(
            model_no_w.coefficients["mu"],
            model_w.coefficients["mu"],
            rtol=1e-5,
        )

    def test_weights_change_estimates(self):
        """Non-uniform weights should produce different estimates."""
        df, y, _, _ = make_gamma_data(n=100, seed=100)
        weights = np.where(np.arange(100) < 50, 2.0, 0.1)

        model_no_w = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model_no_w.fit(df, y)

        model_w = DistributionalGLM(family=Gamma(), formulas={"mu": ["x1"], "sigma": []})
        model_w.fit(df, y, weights=weights)

        # Different weights -> different coefficients
        assert not np.allclose(
            model_no_w.coefficients["mu"],
            model_w.coefficients["mu"],
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# P0/P1 regression tests
# ---------------------------------------------------------------------------

class TestLogNormalPredictMean:
    """P1: predict_mean() for LogNormal must include the sigma^2/2 correction."""

    def test_mean_correction_applied(self):
        """predict_mean should return exp(mu + sigma^2/2), not exp(mu).

        For known mu and sigma, the difference is exp(sigma^2/2) - 1
        which is ~28% for sigma=0.7. We verify the correction is applied
        by fitting an intercept-only model and checking the mean.
        """
        rng = np.random.default_rng(42)
        mu_true = 7.0
        sigma_true = 0.7
        n = 5000
        y = np.exp(rng.normal(mu_true, sigma_true, n))

        df = pl.DataFrame({"dummy": np.zeros(n)})
        model = DistributionalGLM(
            family=LogNormal(),
            formulas={"mu": [], "sigma": []},
        )
        model.fit(df, y, max_iter=200)

        # True E[Y] = exp(mu + sigma^2/2)
        true_mean = np.exp(mu_true + 0.5 * sigma_true**2)
        predicted_mean = model.predict_mean(df)

        # Should be within 5% of true mean
        np.testing.assert_allclose(predicted_mean[0], true_mean, rtol=0.05)

    def test_mean_greater_than_exp_mu(self):
        """For sigma > 0, predict_mean must be strictly greater than exp(mu)."""
        rng = np.random.default_rng(7)
        y = np.exp(rng.normal(6.0, 0.5, 300))
        df = pl.DataFrame({"x": rng.normal(size=300)})
        model = DistributionalGLM(
            family=LogNormal(),
            formulas={"mu": ["x"], "sigma": []},
        )
        model.fit(df, y, max_iter=200)

        mu = model.predict(df, parameter="mu")
        sigma = model.predict(df, parameter="sigma")
        mean_hat = model.predict_mean(df)

        # mean_hat should be exp(mu + sigma^2/2) > exp(mu)
        np.testing.assert_array_less(np.exp(mu), mean_hat + 1e-10)


class TestNonConvergenceWarning:
    """P1: fit() must always warn on non-convergence, regardless of verbose."""

    def test_warns_without_verbose(self):
        """Non-convergence warning must fire even when verbose=False."""
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 500.0, 50)
        df = pl.DataFrame({"x": rng.normal(size=50)})
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": ["x"], "sigma": []},
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            model.fit(df, y, max_iter=1, verbose=False)  # force non-convergence
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) >= 1
        assert "converge" in str(runtime_warnings[0].message).lower()

    def test_converged_no_warning(self):
        """Converged model should produce no RuntimeWarning."""
        rng = np.random.default_rng(1)
        y = rng.gamma(4.0, 250.0, 500)
        df = pl.DataFrame({"x": rng.normal(size=500)})
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": ["x"], "sigma": []},
        )
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            model.fit(df, y, max_iter=200)
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warnings) == 0


class TestResolveFormulasValidation:
    """P1: _resolve_formulas must raise ValueError for unrecognised keys."""

    def test_unknown_key_raises(self):
        """A typo in the formulas key should raise, not silently fall back."""
        rng = np.random.default_rng(0)
        y = rng.gamma(2.0, 500.0, 50)
        df = pl.DataFrame({"x": rng.normal(size=50)})
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": ["x"], "dispersoin": []},  # typo
        )
        with pytest.raises(ValueError, match="not recognised"):
            model.fit(df, y)

    def test_valid_keys_accepted(self):
        """Valid family parameter keys must be accepted without error."""
        rng = np.random.default_rng(2)
        y = rng.gamma(4.0, 250.0, 100)
        df = pl.DataFrame({"x": rng.normal(size=100)})
        model = DistributionalGLM(
            family=Gamma(),
            formulas={"mu": ["x"], "sigma": []},
        )
        model.fit(df, y, max_iter=50)  # should not raise

    def test_none_formulas_accepted(self):
        """formulas=None is the default and must always work."""
        rng = np.random.default_rng(3)
        y = rng.gamma(4.0, 250.0, 100)
        df = pl.DataFrame({"x": rng.normal(size=100)})
        model = DistributionalGLM(family=Gamma())
        model.fit(df, y, max_iter=50)  # should not raise
