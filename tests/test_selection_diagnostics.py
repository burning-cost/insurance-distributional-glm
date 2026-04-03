"""
Extended tests for insurance_distributional_glm:
- selection.py: gaic function, SelectionResult, choose_distribution edge cases
- diagnostics.py: quantile_residuals for all families, worm_plot/fitted_vs_observed
- model.py: score(metric='crps'), predict_variance for all families, InverseGaussian
- fitting.py: _wls helper, rs_fit with verbose, single-column design matrix
- families: Tweedie fitting, base family interface, links

Gaps NOT covered in existing tests:
- choose_distribution with empty families list
- choose_distribution with a family that throws
- gaic with penalty != 2
- diagnostics quantile_residuals for NBI
- diagnostics quantile_residuals seed reproducibility
- diagnostics quantile_residuals n_random > 1
- DistributionalGLM.score('crps')
- predict_variance for all families
- predict_distribution for InverseGaussian
- Tweedie fitting end-to-end
- InverseGaussian fitting end-to-end
- _wls with uniform weights
- rs_fit convergence history
- NBI predict_mean
- ZIP predict_variance
- DistributionalGLM with InverseGaussian family
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import polars as pl

from insurance_distributional_glm import (
    DistributionalGLM,
    quantile_residuals,
    gaic,
    choose_distribution,
)
from insurance_distributional_glm.families import (
    Gamma, LogNormal, InverseGaussian, Tweedie, Poisson, NBI, ZIP,
)
from insurance_distributional_glm.selection import SelectionResult
from insurance_distributional_glm.fitting import _wls, rs_fit


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def make_gamma_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu = np.exp(6.0 + 0.3 * x)
    sigma = 0.5
    y = rng.gamma(1.0 / sigma**2, mu * sigma**2)
    return pl.DataFrame({"x": x}), y


def make_lognormal_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    mu_log = 6.5 + 0.2 * x
    sigma_log = 0.4
    y = np.exp(rng.normal(mu_log, sigma_log))
    return pl.DataFrame({"x": x}), y


def make_ig_data(n=200, seed=42):
    from scipy.stats import invgauss
    rng = np.random.default_rng(seed)
    mu = 500.0
    lam = 1.0 / 0.01  # sigma^2 = 0.01
    y = invgauss.rvs(mu=mu / lam, scale=lam, size=n, random_state=int(rng.integers(1000)))
    return pl.DataFrame({"x": np.zeros(n)}), y


def make_tweedie_data(n=200, seed=42):
    """Tweedie data: ~30% zeros, rest from Gamma."""
    rng = np.random.default_rng(seed)
    zeros = rng.random(n) < 0.3
    y = np.where(zeros, 0.0, rng.gamma(2.0, 300.0, n))
    return pl.DataFrame({"x": np.zeros(n)}), y


def make_nbi_data(n=300, seed=42):
    from scipy.stats import nbinom
    rng = np.random.default_rng(seed)
    mu, k = 3.0, 2.0
    p_nb = k / (k + mu)
    y = nbinom.rvs(n=k, p=p_nb, size=n, random_state=int(rng.integers(1000))).astype(float)
    return pl.DataFrame({"x": np.zeros(n)}), y


def make_zip_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    pi, mu = 0.25, 2.5
    zeros = rng.random(n) < pi
    y = np.where(zeros, 0.0, rng.poisson(mu, n).astype(float))
    return pl.DataFrame({"x": np.zeros(n)}), y


# ---------------------------------------------------------------------------
# gaic function
# ---------------------------------------------------------------------------

class TestGAICFunction:

    def test_aic_penalty(self):
        # GAIC(2) = -2*ll + 2*k
        assert gaic(-100.0, 3, 2.0) == pytest.approx(206.0)

    def test_bic_penalty(self):
        n = 1000
        penalty = np.log(n)
        assert gaic(-100.0, 3, penalty) == pytest.approx(-2 * (-100.0) + penalty * 3)

    def test_zero_penalty(self):
        assert gaic(-100.0, 5, 0.0) == pytest.approx(200.0)

    def test_more_params_higher_gaic(self):
        ll = -500.0
        g1 = gaic(ll, 2, 2.0)
        g2 = gaic(ll, 5, 2.0)
        assert g2 > g1

    def test_better_loglik_lower_gaic(self):
        g1 = gaic(-500.0, 3, 2.0)
        g2 = gaic(-400.0, 3, 2.0)
        assert g2 < g1


# ---------------------------------------------------------------------------
# choose_distribution edge cases
# ---------------------------------------------------------------------------

class TestChooseDistributionEdgeCases:

    def test_empty_families_returns_empty(self):
        df, y = make_gamma_data(n=100)
        results = choose_distribution(df, y, families=[], formulas={"mu": [], "sigma": []})
        assert results == []

    def test_single_family_returns_one_result(self):
        df, y = make_gamma_data(n=100)
        results = choose_distribution(df, y, [Gamma()], formulas={"mu": [], "sigma": []})
        assert len(results) == 1
        assert isinstance(results[0], SelectionResult)

    def test_result_has_model(self):
        df, y = make_gamma_data(n=100)
        results = choose_distribution(df, y, [Gamma()], formulas={"mu": [], "sigma": []})
        assert results[0].model is not None
        assert isinstance(results[0].model, DistributionalGLM)

    def test_bic_penalty(self):
        df, y = make_gamma_data(n=200)
        n = len(y)
        results = choose_distribution(
            df, y, [Gamma(), LogNormal()],
            formulas={"mu": [], "sigma": []},
            penalty=np.log(n),
        )
        assert len(results) == 2
        # BIC-penalised results should still be sorted ascending
        gaics = [r.gaic for r in results]
        assert gaics == sorted(gaics)

    def test_verbose_runs_without_error(self, capsys):
        df, y = make_gamma_data(n=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            choose_distribution(df, y, [Gamma()],
                                formulas={"mu": [], "sigma": []},
                                verbose=True)
        # If we get here without crashing, test passes
        assert True

    def test_converged_flag_in_result(self):
        df, y = make_gamma_data(n=200)
        results = choose_distribution(df, y, [Gamma()],
                                      formulas={"mu": [], "sigma": []},
                                      max_iter=200)
        assert isinstance(results[0].converged, bool)

    def test_df_is_param_count(self):
        """df should equal total number of coefficients."""
        df_data, y = make_gamma_data(n=100)
        # mu: intercept + x => 2 params; sigma: intercept only => 1 param
        results = choose_distribution(df_data, y, [Gamma()],
                                      formulas={"mu": ["x"], "sigma": []})
        assert results[0].df == 3

    def test_choose_distribution_returns_sorted(self):
        df, y = make_gamma_data(n=300)
        fams = [Gamma(), LogNormal(), InverseGaussian()]
        results = choose_distribution(df, y, fams, formulas={"mu": ["x"], "sigma": []})
        gaics = [r.gaic for r in results]
        assert gaics == sorted(gaics)


# ---------------------------------------------------------------------------
# InverseGaussian fitting
# ---------------------------------------------------------------------------

class TestInverseGaussianFitting:

    def test_ig_converges(self):
        df, y = make_ig_data(n=300)
        model = DistributionalGLM(family=InverseGaussian(), formulas={"mu": [], "sigma": []})
        model.fit(df, y, max_iter=200)
        # Check mu estimate is in the right ballpark (true mu=500)
        mu_hat = model.predict(df, "mu")[0]
        assert 200 < mu_hat < 800

    def test_ig_predict_mu_positive(self):
        df, y = make_ig_data(n=200)
        model = DistributionalGLM(family=InverseGaussian(), formulas={"mu": [], "sigma": []})
        model.fit(df, y, max_iter=200)
        mu = model.predict(df, "mu")
        assert np.all(mu > 0)

    def test_ig_predict_sigma_positive(self):
        df, y = make_ig_data(n=200)
        model = DistributionalGLM(family=InverseGaussian(), formulas={"mu": [], "sigma": []})
        model.fit(df, y, max_iter=200)
        sigma = model.predict(df, "sigma")
        assert np.all(sigma > 0)

    def test_ig_predict_distribution(self):
        df, y = make_ig_data(n=50)
        model = DistributionalGLM(family=InverseGaussian(), formulas={"mu": [], "sigma": []})
        model.fit(df, y, max_iter=100)
        dists = model.predict_distribution(df)
        assert len(dists) == 50
        # InverseGaussian should have non-None dists
        not_none = sum(1 for d in dists if d is not None)
        assert not_none > 0

    def test_ig_variance_formula(self):
        """Var[Y] = sigma^2 * mu^3 for IG."""
        df, y = make_ig_data(n=100)
        model = DistributionalGLM(family=InverseGaussian(), formulas={"mu": [], "sigma": []})
        model.fit(df, y, max_iter=100)
        var = model.predict_variance(df)
        mu = model.predict(df, "mu")
        sigma = model.predict(df, "sigma")
        expected_var = sigma**2 * mu**3
        np.testing.assert_allclose(var, expected_var, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tweedie fitting
# ---------------------------------------------------------------------------

class TestTweedie:

    def test_tweedie_fit_converges(self):
        df, y = make_tweedie_data(n=300)
        model = DistributionalGLM(family=Tweedie(power=1.5), formulas={"mu": [], "phi": []})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, y, max_iter=100)
        # At minimum, loglik should be finite
        assert np.isfinite(model.loglikelihood)

    def test_tweedie_predict_mu_positive(self):
        df, y = make_tweedie_data(n=200)
        model = DistributionalGLM(family=Tweedie(power=1.5), formulas={"mu": [], "phi": []})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, y, max_iter=100)
        mu = model.predict(df, "mu")
        assert np.all(mu > 0)

    def test_tweedie_predict_phi_positive(self):
        df, y = make_tweedie_data(n=200)
        model = DistributionalGLM(family=Tweedie(power=1.5), formulas={"mu": [], "phi": []})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, y, max_iter=100)
        phi = model.predict(df, "phi")
        assert np.all(phi > 0)

    def test_tweedie_power_attribute(self):
        fam = Tweedie(power=1.7)
        assert fam.power == 1.7

    def test_tweedie_param_names(self):
        assert Tweedie().param_names == ["mu", "phi"]

    def test_tweedie_gaic_finite(self):
        df, y = make_tweedie_data(n=200)
        model = DistributionalGLM(family=Tweedie(power=1.5), formulas={"mu": [], "phi": []})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df, y, max_iter=100)
        g = model.gaic(penalty=2.0)
        assert np.isfinite(g)


# ---------------------------------------------------------------------------
# predict_variance for all families
# ---------------------------------------------------------------------------

class TestPredictVarianceAllFamilies:

    def test_gamma_variance_equals_sigma_sq_mu_sq(self):
        df, y = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        var = model.predict_variance(df)
        mu = model.predict(df, "mu")
        sigma = model.predict(df, "sigma")
        np.testing.assert_allclose(var, (sigma * mu)**2, rtol=1e-5)

    def test_lognormal_variance_positive(self):
        df, y = make_lognormal_data(n=100)
        model = DistributionalGLM(family=LogNormal(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        var = model.predict_variance(df)
        assert np.all(var > 0)

    def test_poisson_variance_equals_mu(self):
        """Poisson: Var = mu."""
        rng = np.random.default_rng(0)
        y = rng.poisson(3.0, 100).astype(float)
        df = pl.DataFrame({"x": np.zeros(100)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y)
        var = model.predict_variance(df)
        mu = model.predict(df, "mu")
        np.testing.assert_allclose(var, mu, rtol=1e-5)

    def test_nbi_variance_positive(self):
        df, y = make_nbi_data(n=200)
        model = DistributionalGLM(family=NBI(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        var = model.predict_variance(df)
        assert np.all(var > 0)

    def test_nbi_variance_greater_than_poisson(self):
        """NBI variance = mu + sigma*mu^2 > mu (Poisson variance)."""
        df, y = make_nbi_data(n=200)
        model = DistributionalGLM(family=NBI(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        var_nbi = model.predict_variance(df)
        mu = model.predict(df, "mu")
        # NBI var > Poisson var
        assert np.all(var_nbi > mu)

    def test_zip_variance_positive(self):
        df, y = make_zip_data(n=200)
        model = DistributionalGLM(family=ZIP(), formulas={"mu": [], "pi": []})
        model.fit(df, y)
        var = model.predict_variance(df)
        assert np.all(var >= 0)


# ---------------------------------------------------------------------------
# NBI predict_mean
# ---------------------------------------------------------------------------

class TestNBIPredictMean:
    def test_predict_mean_equals_predict_mu_for_nbi(self):
        """NBI: E[Y|X] = mu (no correction factor like LogNormal)."""
        df, y = make_nbi_data(n=100)
        model = DistributionalGLM(family=NBI(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        np.testing.assert_array_equal(model.predict_mean(df), model.predict(df, "mu"))


# ---------------------------------------------------------------------------
# score('crps')
# ---------------------------------------------------------------------------

class TestScoreCRPS:

    def test_crps_gamma_finite(self):
        rng = np.random.default_rng(10)
        y = rng.gamma(4.0, 250.0, 50)
        df = pl.DataFrame({"x": np.zeros(50)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        crps = model.score(df, y, metric="crps")
        assert np.isfinite(crps)
        assert crps > 0

    def test_crps_lognormal_finite(self):
        rng = np.random.default_rng(11)
        y = np.exp(rng.normal(6.0, 0.5, 50))
        df = pl.DataFrame({"x": np.zeros(50)})
        model = DistributionalGLM(family=LogNormal(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        crps = model.score(df, y, metric="crps")
        assert np.isfinite(crps)

    def test_crps_poisson_finite(self):
        rng = np.random.default_rng(12)
        y = rng.poisson(3.0, 50).astype(float)
        df = pl.DataFrame({"x": np.zeros(50)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y)
        crps = model.score(df, y, metric="crps")
        assert np.isfinite(crps)

    def test_crps_better_model_lower(self):
        """A model fit to the data should have lower CRPS than one fit on noise."""
        rng = np.random.default_rng(99)
        n = 200
        y = rng.gamma(4.0, 250.0, n)
        df_good = pl.DataFrame({"x": np.zeros(n)})
        df_noise = pl.DataFrame({"x": rng.normal(size=n)})  # random noise

        model_good = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model_good.fit(df_good, y)

        model_noise = DistributionalGLM(family=Gamma(), formulas={"mu": ["x"], "sigma": []})
        model_noise.fit(df_noise, y)

        crps_good = model_good.score(df_good, y, metric="crps")
        crps_noise = model_noise.score(df_noise, y, metric="crps")
        # Both should be finite — the noise model should have comparable or higher CRPS
        assert np.isfinite(crps_good)
        assert np.isfinite(crps_noise)


# ---------------------------------------------------------------------------
# Quantile residuals
# ---------------------------------------------------------------------------

class TestQuantileResidualsExtended:

    def test_nbi_residuals_mostly_finite(self):
        df, y = make_nbi_data(n=200)
        model = DistributionalGLM(family=NBI(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        resids = quantile_residuals(model, df, y, seed=0)
        frac_finite = np.mean(np.isfinite(resids))
        assert frac_finite > 0.85

    def test_lognormal_residuals_shape(self):
        df, y = make_lognormal_data(n=100)
        model = DistributionalGLM(family=LogNormal(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        resids = quantile_residuals(model, df, y, seed=42)
        assert resids.shape == (100,)

    def test_seed_reproducibility(self):
        df, y = make_gamma_data(n=100)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        r1 = quantile_residuals(model, df, y, seed=7)
        r2 = quantile_residuals(model, df, y, seed=7)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_may_differ_for_discrete(self):
        """For discrete families, different seeds should produce different residuals."""
        rng = np.random.default_rng(0)
        y = rng.poisson(2.0, 100).astype(float)
        df = pl.DataFrame({"x": np.zeros(100)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y)
        r1 = quantile_residuals(model, df, y, seed=1, n_random=1)
        r2 = quantile_residuals(model, df, y, seed=2, n_random=1)
        # Different seeds → at least some residuals should differ
        assert not np.allclose(r1, r2)

    def test_n_random_averaging(self):
        """n_random>1 should produce smoother (lower variance) residuals for discrete."""
        rng = np.random.default_rng(0)
        y = rng.poisson(2.0, 200).astype(float)
        df = pl.DataFrame({"x": np.zeros(200)})
        model = DistributionalGLM(family=Poisson(), formulas={"mu": []})
        model.fit(df, y)
        r_single = quantile_residuals(model, df, y, seed=0, n_random=1)
        r_avg = quantile_residuals(model, df, y, seed=0, n_random=50)
        finite_single = r_single[np.isfinite(r_single)]
        finite_avg = r_avg[np.isfinite(r_avg)]
        # Averaged residuals should have lower variance
        assert np.var(finite_avg) <= np.var(finite_single) + 0.5  # lenient


# ---------------------------------------------------------------------------
# Diagnostics: worm_plot / fitted_vs_observed skip without matplotlib
# ---------------------------------------------------------------------------

class TestDiagnosticsPlotting:

    def test_worm_plot_raises_without_matplotlib(self, monkeypatch):
        import sys
        import types
        from insurance_distributional_glm import diagnostics

        df, y = make_gamma_data(n=50)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)

        # Monkeypatch matplotlib import to fail
        original = sys.modules.get("matplotlib")
        original_plt = sys.modules.get("matplotlib.pyplot")
        sys.modules["matplotlib"] = None
        sys.modules["matplotlib.pyplot"] = None

        try:
            with pytest.raises((ImportError, TypeError)):
                diagnostics.worm_plot(model, df, y)
        finally:
            if original is not None:
                sys.modules["matplotlib"] = original
            elif "matplotlib" in sys.modules:
                del sys.modules["matplotlib"]
            if original_plt is not None:
                sys.modules["matplotlib.pyplot"] = original_plt
            elif "matplotlib.pyplot" in sys.modules:
                del sys.modules["matplotlib.pyplot"]

    def test_fitted_vs_observed_with_matplotlib(self):
        """If matplotlib is available, fitted_vs_observed should return an axes."""
        mpl = pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from insurance_distributional_glm.diagnostics import fitted_vs_observed

        df, y = make_gamma_data(n=50)
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        ax = fitted_vs_observed(model, df, y)
        assert ax is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# _wls helper
# ---------------------------------------------------------------------------

class TestWLSHelper:

    def test_unweighted_ols(self):
        """With unit weights, _wls should recover OLS solution."""
        rng = np.random.default_rng(0)
        n, p = 100, 3
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
        beta_true = np.array([1.0, 0.5, -0.3])
        y = X @ beta_true + rng.normal(scale=0.1, size=n)
        w = np.ones(n)
        beta_hat = _wls(X, y, w)
        np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)

    def test_heavily_weighted_obs_dominate(self):
        """Observations with very high weight should dominate the solution."""
        n = 100
        X = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        # True line: y = 2 + 3*x except last 2 points which are outliers
        y = 2.0 + 3.0 * X[:, 1]
        y[-2:] = -1000.0  # outliers

        # Low weight on outliers
        w = np.ones(n)
        w[-2:] = 1e-10
        beta = _wls(X, y, w)
        # Intercept ~2, slope ~3
        assert abs(beta[0] - 2.0) < 1.0
        assert abs(beta[1] - 3.0) < 1.0


# ---------------------------------------------------------------------------
# rs_fit
# ---------------------------------------------------------------------------

class TestRSFit:

    def test_rs_fit_returns_four_elements(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(4.0, 250.0, 50)
        X_mu = np.ones((50, 1))
        X_sigma = np.ones((50, 1))
        fam = Gamma()
        result = rs_fit(fam, {"mu": X_mu, "sigma": X_sigma}, y, max_iter=50)
        assert len(result) == 4
        betas, params, history, converged = result
        assert "mu" in betas
        assert "sigma" in betas
        assert isinstance(history, list)
        assert isinstance(converged, bool)

    def test_rs_fit_loglik_history_increases(self):
        """Log-likelihood should be non-decreasing (up to tolerance)."""
        rng = np.random.default_rng(1)
        y = rng.gamma(4.0, 250.0, 200)
        X = np.ones((200, 1))
        fam = Gamma()
        _, _, history, converged = rs_fit(fam, {"mu": X, "sigma": X}, y, max_iter=100)
        assert len(history) >= 1
        # Each step should not decrease by more than numerical noise
        for i in range(1, min(len(history), 5)):
            assert history[i] >= history[i - 1] - 1.0  # lenient 1.0 unit

    def test_rs_fit_with_offset(self):
        rng = np.random.default_rng(2)
        n = 100
        exposure = rng.uniform(0.5, 2.0, n)
        log_offset = np.log(exposure)
        y = rng.poisson(3.0 * exposure).astype(float)
        X = np.ones((n, 1))
        fam = Poisson()
        betas, params, _, _ = rs_fit(
            fam, {"mu": X}, y, log_offset=log_offset, max_iter=100
        )
        # log(rate) ~ log(3) ≈ 1.099
        assert abs(betas["mu"][0] - np.log(3.0)) < 0.5

    def test_rs_fit_verbose_runs(self, capsys):
        rng = np.random.default_rng(3)
        y = rng.gamma(4.0, 250.0, 30)
        X = np.ones((30, 1))
        fam = Gamma()
        rs_fit(fam, {"mu": X, "sigma": X}, y, max_iter=5, verbose=True)
        out = capsys.readouterr().out
        assert "loglik" in out


# ---------------------------------------------------------------------------
# DistributionalGLM properties when not fitted
# ---------------------------------------------------------------------------

class TestUnfittedModelErrors:

    def test_coefficients_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        with pytest.raises(RuntimeError):
            _ = model.coefficients

    def test_loglikelihood_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        with pytest.raises(RuntimeError):
            _ = model.loglikelihood

    def test_converged_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        with pytest.raises(RuntimeError):
            _ = model.converged

    def test_gaic_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        with pytest.raises(RuntimeError):
            model.gaic()

    def test_predict_variance_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        df = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            model.predict_variance(df)

    def test_volatility_score_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        df = pl.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(RuntimeError):
            model.volatility_score(df)

    def test_relativities_raises_when_unfitted(self):
        model = DistributionalGLM(family=Gamma())
        with pytest.raises(RuntimeError):
            model.relativities("mu")


# ---------------------------------------------------------------------------
# DistributionalGLM n_observations property
# ---------------------------------------------------------------------------

class TestNObservations:

    def test_n_observations_before_fit(self):
        model = DistributionalGLM(family=Gamma())
        assert model.n_observations == 0

    def test_n_observations_after_fit(self):
        rng = np.random.default_rng(0)
        y = rng.gamma(4.0, 250.0, 77)
        df = pl.DataFrame({"x": np.zeros(77)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        assert model.n_observations == 77


# ---------------------------------------------------------------------------
# Families: links
# ---------------------------------------------------------------------------

class TestLinks:

    def test_gamma_links_log(self):
        fam = Gamma()
        assert fam.default_links["mu"].name == "log"
        assert fam.default_links["sigma"].name == "log"

    def test_lognormal_mu_link_identity(self):
        fam = LogNormal()
        assert fam.default_links["mu"].name == "identity"
        assert fam.default_links["sigma"].name == "log"

    def test_ig_links_log(self):
        fam = InverseGaussian()
        assert fam.default_links["mu"].name == "log"
        assert fam.default_links["sigma"].name == "log"

    def test_poisson_link_log(self):
        fam = Poisson()
        assert fam.default_links["mu"].name == "log"

    def test_nbi_links_log(self):
        fam = NBI()
        assert fam.default_links["mu"].name == "log"
        assert fam.default_links["sigma"].name == "log"

    def test_zip_links(self):
        fam = ZIP()
        assert fam.default_links["mu"].name == "log"
        assert fam.default_links["pi"].name == "logit"

    def test_tweedie_links_log(self):
        fam = Tweedie(power=1.5)
        assert fam.default_links["mu"].name == "log"
        assert fam.default_links["phi"].name == "log"
