"""Tests for GAMLSS family implementations."""
import numpy as np
import pytest
from insurance_distributional_glm.families import (
    Gamma, LogNormal, InverseGaussian, Tweedie,
    Poisson, NBI, ZIP,
)


def numerical_gradient(fn, params, param_key, h=1e-5):
    """Numerical gradient of fn w.r.t. params[param_key] using central differences."""
    p_plus = {k: v.copy() for k, v in params.items()}
    p_minus = {k: v.copy() for k, v in params.items()}
    p_plus[param_key] = params[param_key] + h
    p_minus[param_key] = params[param_key] - h
    return (fn(p_plus) - fn(p_minus)) / (2 * h)


def gradient_in_eta(family, y, params, param_name, h=1e-5):
    """Numerical d(loglik)/d(eta_k) via finite differences on eta."""
    link = family.default_links[param_name]
    eta = link.link(params[param_name])

    def ll_at_eta(e):
        p = {k: v.copy() for k, v in params.items()}
        p[param_name] = link.inverse(e)
        return family.log_likelihood(y, p)

    return (ll_at_eta(eta + h) - ll_at_eta(eta - h)) / (2 * h)


# ---------------------------------------------------------------------------
# Gamma family tests
# ---------------------------------------------------------------------------

class TestGamma:
    fam = Gamma()
    rng = np.random.default_rng(42)
    y = rng.gamma(shape=4.0, scale=250.0, size=50)
    params = {
        "mu": np.full(50, 1000.0),
        "sigma": np.full(50, 0.5),
    }

    def test_param_names(self):
        assert self.fam.param_names == ["mu", "sigma"]

    def test_default_links(self):
        assert self.fam.default_links["mu"].name == "log"
        assert self.fam.default_links["sigma"].name == "log"

    def test_log_likelihood_finite(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert ll.shape == (50,)
        assert np.all(np.isfinite(ll))

    def test_log_likelihood_negative(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert np.all(np.isfinite(ll))

    def test_dl_deta_mu_shape(self):
        score = self.fam.dl_deta(self.y, self.params, "mu")
        assert score.shape == (50,)

    def test_dl_deta_sigma_shape(self):
        score = self.fam.dl_deta(self.y, self.params, "sigma")
        assert score.shape == (50,)

    def test_dl_deta_mu_matches_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "mu")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "mu")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)

    def test_dl_deta_sigma_matches_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "sigma")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "sigma")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-4)

    def test_d2l_deta2_positive(self):
        w_mu = self.fam.d2l_deta2(self.y, self.params, "mu")
        w_sigma = self.fam.d2l_deta2(self.y, self.params, "sigma")
        assert np.all(w_mu > 0)
        assert np.all(w_sigma > 0)

    def test_starting_values(self):
        sv = self.fam.starting_values(self.y)
        assert "mu" in sv and "sigma" in sv
        assert np.all(sv["mu"] > 0)
        assert np.all(sv["sigma"] > 0)

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "gamma")

    def test_repr(self):
        r = repr(self.fam)
        assert "Gamma" in r
        assert "log" in r


class TestLogNormal:
    fam = LogNormal()
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(loc=7.0, scale=0.5, size=50))
    params = {
        "mu": np.full(50, 7.0),
        "sigma": np.full(50, 0.5),
    }

    def test_param_names(self):
        assert self.fam.param_names == ["mu", "sigma"]

    def test_links(self):
        assert self.fam.default_links["mu"].name == "identity"
        assert self.fam.default_links["sigma"].name == "log"

    def test_log_likelihood_finite(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert np.all(np.isfinite(ll))

    def test_dl_deta_mu_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "mu")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "mu")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-8)

    def test_dl_deta_sigma_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "sigma")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "sigma")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-8)

    def test_d2l_deta2_positive(self):
        for pname in ["mu", "sigma"]:
            w = self.fam.d2l_deta2(self.y, self.params, pname)
            assert np.all(w > 0), f"Non-positive weights for {pname}"

    def test_starting_values(self):
        sv = self.fam.starting_values(self.y)
        assert np.all(sv["sigma"] > 0)

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "kappa")


class TestInverseGaussian:
    fam = InverseGaussian()
    rng = np.random.default_rng(3)
    # scipy invgauss: mu_param = mu/lambda, scale = lambda
    # With lambda=100, mu=100: y ~ IG(mu=100, sigma=0.1) approx
    from scipy import stats as _stats
    y = _stats.invgauss(mu=1.0, scale=100.0).rvs(50, random_state=3)
    params = {
        "mu": np.full(50, 100.0),
        "sigma": np.full(50, 0.1),
    }

    def test_param_names(self):
        assert self.fam.param_names == ["mu", "sigma"]

    def test_log_likelihood_finite(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert np.all(np.isfinite(ll))

    def test_dl_deta_mu_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "mu")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "mu")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_dl_deta_sigma_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "sigma")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "sigma")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-6)

    def test_d2l_positive(self):
        for pname in ["mu", "sigma"]:
            w = self.fam.d2l_deta2(self.y, self.params, pname)
            assert np.all(w > 0)

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "kappa")


class TestPoisson:
    fam = Poisson()
    rng = np.random.default_rng(5)
    y = rng.poisson(lam=3.0, size=100).astype(float)
    params = {"mu": np.full(100, 3.0)}

    def test_log_likelihood_finite(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert np.all(np.isfinite(ll))

    def test_dl_deta_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "mu")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "mu")
        # Use absolute tolerance: when y=mu, score=0 exactly, numerical≈0 with fp error
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-7)

    def test_d2l_positive(self):
        w = self.fam.d2l_deta2(self.y, self.params, "mu")
        assert np.all(w > 0)

    def test_starting_values(self):
        sv = self.fam.starting_values(self.y)
        assert np.all(sv["mu"] > 0)

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "lambda")


class TestNBI:
    fam = NBI()
    rng = np.random.default_rng(7)
    from scipy import stats as _stats
    y = _stats.nbinom(n=2, p=0.4).rvs(100, random_state=7).astype(float)
    params = {
        "mu": np.full(100, 3.0),
        "sigma": np.full(100, 0.5),
    }

    def test_log_likelihood_finite(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert np.all(np.isfinite(ll))

    def test_dl_deta_mu_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "mu")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "mu")
        # Integer y data: use atol for near-zero cases
        np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-7)

    def test_dl_deta_sigma_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "sigma")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "sigma")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-4)

    def test_d2l_positive(self):
        for pname in ["mu", "sigma"]:
            w = self.fam.d2l_deta2(self.y, self.params, pname)
            assert np.all(w > 0), f"Non-positive for {pname}"

    def test_starting_values_overdispersion(self):
        sv = self.fam.starting_values(self.y)
        assert sv["sigma"][0] > 0

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "nu")


class TestZIP:
    fam = ZIP()
    rng = np.random.default_rng(11)
    n = 200
    is_zero = rng.uniform(size=n) < 0.2
    y_poisson = rng.poisson(lam=2.0, size=n).astype(float)
    y = np.where(is_zero, 0.0, y_poisson)
    params = {
        "mu": np.full(n, 2.0),
        "pi": np.full(n, 0.2),
    }

    def test_log_likelihood_finite(self):
        ll = self.fam.log_likelihood(self.y, self.params)
        assert np.all(np.isfinite(ll))

    def test_log_likelihood_zero_obs(self):
        y0 = np.array([0.0])
        p0 = {"mu": np.array([2.0]), "pi": np.array([0.3])}
        ll = self.fam.log_likelihood(y0, p0)
        assert ll[0] < 0

    def test_dl_deta_mu_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "mu")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "mu")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-6)

    def test_dl_deta_pi_numerical(self):
        analytical = self.fam.dl_deta(self.y, self.params, "pi")
        numerical = gradient_in_eta(self.fam, self.y, self.params, "pi")
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-6)

    def test_d2l_positive(self):
        for pname in ["mu", "pi"]:
            w = self.fam.d2l_deta2(self.y, self.params, pname)
            assert np.all(w > 0), f"Non-positive for {pname}"

    def test_starting_values(self):
        sv = self.fam.starting_values(self.y)
        assert 0 < sv["pi"][0] < 1
        assert sv["mu"][0] > 0

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "nu")


class TestTweedie:
    fam = Tweedie(power=1.5)
    rng = np.random.default_rng(13)
    n = 100
    y = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.gamma(2.0, 500.0, n))
    params = {
        "mu": np.full(n, 500.0),
        "phi": np.full(n, 1.0),
    }

    def test_power_validation(self):
        with pytest.raises(ValueError):
            Tweedie(power=0.5)
        with pytest.raises(ValueError):
            Tweedie(power=2.5)

    def test_log_likelihood_zeros(self):
        y0 = np.array([0.0, 0.0])
        p0 = {"mu": np.array([500.0, 500.0]), "phi": np.array([1.0, 1.0])}
        ll = self.fam.log_likelihood(y0, p0)
        assert np.all(np.isfinite(ll))
        assert np.all(ll < 0)

    def test_log_likelihood_positives(self):
        y_pos = np.array([100.0, 500.0, 1000.0])
        p_pos = {"mu": np.array([500.0, 500.0, 500.0]), "phi": np.array([1.0, 1.0, 1.0])}
        ll = self.fam.log_likelihood(y_pos, p_pos)
        assert np.all(np.isfinite(ll))

    def test_starting_values(self):
        sv = self.fam.starting_values(self.y)
        assert np.all(sv["mu"] > 0)
        assert np.all(sv["phi"] > 0)

    def test_d2l_positive(self):
        for pname in ["mu", "phi"]:
            w = self.fam.d2l_deta2(self.y, self.params, pname)
            assert np.all(w > 0)

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError):
            self.fam.dl_deta(self.y, self.params, "sigma")
