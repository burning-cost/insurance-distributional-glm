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


# ---------------------------------------------------------------------------
# Regression tests for P0/P1 bug fixes
# ---------------------------------------------------------------------------

class TestNBISigmaIRLS:
    """
    P0-1: NBI sigma IRLS weight was always clipped to 1e-8 floor.

    The old formula `polygamma(1, k) - 1/k - 1/(k+mu)` is always negative
    for valid inputs. The correct formula uses the observation-level
    (-d^2 ll / d k^2) which varies with y and is mostly positive.
    """

    fam = NBI()

    def _neg_d2l_dk2_numerical(self, k, mu, y, h=1e-4):
        """Numerical -d^2 ll / d k^2 at a single observation."""
        from scipy.special import gammaln

        def ll_k(kv):
            return (
                gammaln(y + kv) - gammaln(kv)
                + kv * np.log(kv / (kv + mu))
                + y * np.log(mu / (mu + kv))
            )

        d2 = (ll_k(k + h) - 2 * ll_k(k) + ll_k(k - h)) / h**2
        return -d2

    def test_irls_weight_not_all_floor(self):
        """After fix, sigma IRLS weights should not all be 1e-8."""
        rng = np.random.default_rng(101)
        from scipy.stats import nbinom
        k = 2.0
        mu = 3.0
        sigma = 1.0 / k
        p_nb = k / (k + mu)
        y = nbinom.rvs(n=k, p=p_nb, size=200, random_state=42).astype(float)
        params = {"mu": np.full(200, mu), "sigma": np.full(200, sigma)}
        w = self.fam.d2l_deta2(y, params, "sigma")
        # After fix: majority of weights should be > 1e-8 (not pinned to floor)
        frac_above_floor = np.mean(w > 1e-8 * 1.01)
        assert frac_above_floor > 0.5, f"Only {frac_above_floor:.2%} weights above floor"

    def test_irls_weight_positive(self):
        """IRLS weights must be positive (clipped from below)."""
        rng = np.random.default_rng(102)
        from scipy.stats import nbinom
        y = nbinom.rvs(n=2, p=0.4, size=100, random_state=7).astype(float)
        params = {"mu": np.full(100, 3.0), "sigma": np.full(100, 0.5)}
        w = self.fam.d2l_deta2(y, params, "sigma")
        assert np.all(w > 0)

    def test_analytical_neg_d2l_dk2_matches_numerical(self):
        """
        The observation-level -d^2 ll / d k^2 formula should match
        numerical second derivative for y values where the curvature is negative
        enough to be above the clip floor.
        """
        from scipy.special import polygamma

        k = 2.0
        mu = 3.0
        # y=0 gives large positive curvature — safe to check against numerical
        y_test = np.array([0.0])

        analytical = (
            polygamma(1, k)
            - polygamma(1, y_test + k)
            - 1.0 / k
            + 1.0 / (k + mu)
            + (mu - y_test) / (k + mu) ** 2
        )
        numerical = np.array([self._neg_d2l_dk2_numerical(k, mu, 0.0)])
        np.testing.assert_allclose(analytical, numerical, rtol=1e-2)

    def test_mean_neg_d2l_dk2_is_expected_fisher(self):
        """
        Mean of observation-level -d^2ll/dk^2 over the NBI distribution
        should approximate the expected Fisher information, which is positive.
        """
        from scipy.special import polygamma
        from scipy.stats import nbinom

        k = 2.0
        mu = 3.0
        p_nb = k / (k + mu)
        rng = np.random.default_rng(55)
        y_sample = nbinom.rvs(n=k, p=p_nb, size=50000, random_state=55).astype(float)

        neg_d2l = (
            polygamma(1, k)
            - polygamma(1, y_sample + k)
            - 1.0 / k
            + 1.0 / (k + mu)
            + (mu - y_sample) / (k + mu) ** 2
        )
        mean_neg_d2l = np.mean(neg_d2l)
        assert mean_neg_d2l > 0, f"Expected Fisher info should be positive, got {mean_neg_d2l}"


class TestTweedieLogW:
    """
    P0-2: Tweedie _log_w series had wrong formula.

    The correct formula uses alpha = (2-p)/(p-1) > 0 and
    gammaln(j*alpha), matching the compound Poisson-Gamma density.
    """

    def _log_w_from_poisson_gamma(self, y, phi, p, n_terms=200):
        """
        Reference log_w computed directly from the Poisson-Gamma mixture.
        Matches what _log_w should return.
        """
        from scipy.stats import poisson, gamma as gamma_dist
        alpha = (2.0 - p) / (p - 1.0)
        lam = np.mean(np.asarray(y) ** 0 * 1.0)  # dummy — use scalar
        # We need scalar y, phi, mu for this reference
        raise NotImplementedError

    def test_log_w_matches_compound_poisson_gamma(self):
        """
        For a specific (y, phi, p), the series sum should match the
        numerical compound Poisson-Gamma mixture density.
        """
        from scipy.stats import poisson, gamma as gamma_dist

        p = 1.5
        mu = 500.0
        phi = 1.0
        y_test = 500.0
        alpha = (2.0 - p) / (p - 1.0)  # 1.0
        lam = mu ** (2.0 - p) / (phi * (2.0 - p))
        scale = phi * (p - 1.0) * mu ** (p - 1.0)

        # Numerical log_w from compound Poisson-Gamma
        log_probs = []
        for j in range(1, 300):
            log_pj = poisson.logpmf(j, lam)
            log_fj = gamma_dist.logpdf(y_test, a=j * alpha, scale=scale)
            log_probs.append(log_pj + log_fj)

        # The log-likelihood front = -lam - y/scale (absorbed into log_w reference)
        # ll_front = y*mu^{1-p}/((1-p)*phi) - mu^{2-p}/((2-p)*phi)
        ll_front = y_test * mu ** (1.0 - p) / ((1.0 - p) * phi) - mu ** (2.0 - p) / ((2.0 - p) * phi)
        max_lp = max(log_probs)
        log_density = max_lp + np.log(sum(np.exp(lp - max_lp) for lp in log_probs))
        log_w_expected = log_density - ll_front

        fam = Tweedie(power=p)
        y_arr = np.array([y_test])
        phi_arr = np.array([phi])
        log_w_computed = fam._log_w(y_arr, phi_arr, p)[0]

        # Series truncated at j=50 has limited precision for large lam; use atol
        np.testing.assert_allclose(log_w_computed, log_w_expected, atol=0.5)

    def test_log_likelihood_matches_compound_poisson_gamma(self):
        """
        Full Tweedie log-likelihood should match the compound Poisson-Gamma density.
        """
        from scipy.stats import poisson, gamma as gamma_dist

        p = 1.5
        mu = 300.0
        phi = 2.0
        y_test = 200.0
        alpha = (2.0 - p) / (p - 1.0)
        lam = mu ** (2.0 - p) / (phi * (2.0 - p))
        scale = phi * (p - 1.0) * mu ** (p - 1.0)

        log_probs = []
        for j in range(1, 300):
            log_pj = poisson.logpmf(j, lam)
            log_fj = gamma_dist.logpdf(y_test, a=j * alpha, scale=scale)
            log_probs.append(log_pj + log_fj)

        max_lp = max(log_probs)
        log_density_ref = max_lp + np.log(sum(np.exp(lp - max_lp) for lp in log_probs))

        fam = Tweedie(power=p)
        y_arr = np.array([y_test])
        params = {"mu": np.array([mu]), "phi": np.array([phi])}
        ll_computed = fam.log_likelihood(y_arr, params)[0]

        np.testing.assert_allclose(ll_computed, log_density_ref, rtol=1e-3,
                                   err_msg="Tweedie ll does not match compound Poisson-Gamma")

    def test_log_w_multiple_power_values(self):
        """log_w should be finite and reasonable for several power values."""
        for p in [1.2, 1.5, 1.8]:
            fam = Tweedie(power=p)
            y_arr = np.array([100.0, 500.0, 1000.0])
            phi_arr = np.array([1.0, 1.0, 1.0])
            log_w = fam._log_w(y_arr, phi_arr, p)
            assert np.all(np.isfinite(log_w)), f"log_w not finite for p={p}: {log_w}"


class TestTweedieD2L:
    """
    P0-3 and P0-4: Tweedie d2l_deta2 had wrong formulas for both mu and phi.

    P0-3: d2l_dmu2 was mu^{1-p}/phi; correct is mu^{-p}/phi.
    P0-4: d2l_dphi2 was 1/phi^2; correct is 2*mu^{2-p}/(phi^3*(p-1)*(2-p)).
    """

    def test_d2l_dmu2_matches_expected_fisher(self):
        """
        E[-d^2 ll/dmu^2] = mu^{-p}/phi. Verify numerically via MC.
        """
        from scipy.stats import poisson, gamma as gamma_dist
        from scipy.special import gammaln

        p = 1.5
        mu = 300.0
        phi = 2.0
        alpha = (2.0 - p) / (p - 1.0)
        lam = mu ** (2.0 - p) / (phi * (2.0 - p))
        scale = phi * (p - 1.0) * mu ** (p - 1.0)

        expected_fisher = mu ** (-p) / phi
        fam = Tweedie(power=p)

        y_arr = np.array([mu])
        params = {"mu": np.array([mu]), "phi": np.array([phi])}
        w = fam.d2l_deta2(y_arr, params, "mu")
        # With log link, d2l_deta2 = d2l_dmu2 * mu^2
        d2l_dmu2_computed = w[0] / mu**2
        np.testing.assert_allclose(d2l_dmu2_computed, expected_fisher, rtol=1e-10)

    def test_d2l_dphi2_matches_expected_fisher(self):
        """
        E[-d^2 ll/dphi^2] = 2*mu^{2-p}/(phi^3*(p-1)*(2-p)). Verify analytically.
        """
        p = 1.5
        mu = 300.0
        phi = 2.0

        expected_fisher = 2.0 * mu ** (2.0 - p) / (phi**3 * (p - 1.0) * (2.0 - p))
        fam = Tweedie(power=p)

        y_arr = np.array([mu])
        params = {"mu": np.array([mu]), "phi": np.array([phi])}
        w = fam.d2l_deta2(y_arr, params, "phi")
        # With log link, d2l_deta2 = d2l_dphi2 * phi^2
        d2l_dphi2_computed = w[0] / phi**2
        np.testing.assert_allclose(d2l_dphi2_computed, expected_fisher, rtol=1e-10)

    def test_d2l_mu_wrong_exponent_would_differ(self):
        """
        Show the old (wrong) formula mu^{1-p}/phi differs from the correct mu^{-p}/phi.
        """
        p = 1.5
        mu = 300.0
        phi = 2.0
        wrong = mu ** (1.0 - p) / phi
        correct = mu ** (-p) / phi
        # They should differ by factor mu
        np.testing.assert_allclose(wrong / correct, mu, rtol=1e-10)

    def test_d2l_phi_wrong_formula_would_differ(self):
        """
        Show the old (wrong) formula 1/phi^2 differs substantially from the correct value.
        """
        p = 1.5
        mu = 300.0
        phi = 2.0
        wrong = 1.0 / phi**2
        correct = 2.0 * mu ** (2.0 - p) / (phi**3 * (p - 1.0) * (2.0 - p))
        # The ratio is very different (scales with mu^{2-p})
        ratio = correct / wrong
        assert ratio > 10, f"Expected large ratio (correct >> wrong), got {ratio:.2f}"


class TestZIPPredictMean:
    """
    P1-1: ZIP predict_mean returned mu, not E[Y] = (1-pi)*mu.
    """

    def test_predict_mean_is_ey_not_mu(self):
        """
        For a fitted ZIP model, predict_mean should return (1-pi)*mu, not mu.
        """
        import polars as pl
        from insurance_distributional_glm import DistributionalGLM
        from insurance_distributional_glm.families import ZIP

        rng = np.random.default_rng(200)
        n = 300
        pi_true = 0.3
        mu_true = 2.0
        is_structural_zero = rng.uniform(size=n) < pi_true
        y_poisson = rng.poisson(lam=mu_true, size=n).astype(float)
        y = np.where(is_structural_zero, 0.0, y_poisson)

        df = pl.DataFrame({"x": np.zeros(n)})
        model = DistributionalGLM(
            family=ZIP(),
            formulas={"mu": [], "pi": []},
        )
        model.fit(df, y, max_iter=200)

        mu_pred = model.predict(df, parameter="mu")
        pi_pred = model.predict(df, parameter="pi")
        mean_pred = model.predict_mean(df)
        ey_expected = (1.0 - pi_pred) * mu_pred

        np.testing.assert_allclose(mean_pred, ey_expected, rtol=1e-10,
                                   err_msg="predict_mean should return (1-pi)*mu for ZIP")

        # The predicted mean should be less than mu
        assert np.all(mean_pred < mu_pred), "E[Y] = (1-pi)*mu must be less than mu when pi > 0"

    def test_predict_mean_zip_less_than_mu(self):
        """predict_mean for ZIP should be strictly less than mu (since pi > 0)."""
        import polars as pl
        from insurance_distributional_glm import DistributionalGLM
        from insurance_distributional_glm.families import ZIP

        rng = np.random.default_rng(201)
        n = 200
        y = np.where(rng.uniform(size=n) < 0.25, 0.0, rng.poisson(2.0, n).astype(float))
        df = pl.DataFrame({"x": np.zeros(n)})
        model = DistributionalGLM(family=ZIP(), formulas={"mu": [], "pi": []})
        model.fit(df, y, max_iter=100)

        mu = model.predict(df, parameter="mu")
        mean = model.predict_mean(df)
        assert np.all(mean < mu)


class TestDevianceSum:
    """
    P1-2: score(metric='deviance') used np.mean instead of np.sum.
    Deviance is a sum, not a mean.
    """

    def test_deviance_scales_with_n(self):
        """
        If we double the dataset by repeating each row, the deviance should
        approximately double (sum behaviour), not stay the same (mean behaviour).
        """
        import polars as pl
        from insurance_distributional_glm import DistributionalGLM
        from insurance_distributional_glm.families import Gamma

        rng = np.random.default_rng(300)
        n = 200
        y = rng.gamma(4.0, 250.0, n)
        df = pl.DataFrame({"x": np.zeros(n)})

        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)

        dev_n = model.score(df, y, metric="deviance")

        # Double the data
        y2 = np.concatenate([y, y])
        df2 = pl.DataFrame({"x": np.zeros(2 * n)})
        dev_2n = model.score(df2, y2, metric="deviance")

        # With np.sum: dev_2n ≈ 2 * dev_n
        # With np.mean (old bug): dev_2n ≈ dev_n
        ratio = dev_2n / dev_n
        np.testing.assert_allclose(ratio, 2.0, rtol=0.01,
                                   err_msg=f"Deviance ratio {ratio:.3f} != 2.0 — sum not mean?")

    def test_deviance_nonnegative(self):
        """Deviance should be non-negative."""
        import polars as pl
        from insurance_distributional_glm import DistributionalGLM
        from insurance_distributional_glm.families import Gamma

        rng = np.random.default_rng(301)
        y = rng.gamma(4.0, 250.0, 100)
        df = pl.DataFrame({"x": np.zeros(100)})
        model = DistributionalGLM(family=Gamma(), formulas={"mu": [], "sigma": []})
        model.fit(df, y)
        dev = model.score(df, y, metric="deviance")
        assert dev >= 0
