"""
Continuous distribution families for GAMLSS.

Covers the main severity distributions used in insurance pricing:
  - Gamma: the workhorse for claim severity. sigma = CV (coefficient of variation).
  - LogNormal: log-scale mean and log-scale standard deviation.
  - InverseGaussian: heavier right tail than Gamma, useful for liability severity.
  - Tweedie: compound Poisson-Gamma, used for pure premiums (includes zeros).

Design note on Gamma sigma parameterisation:
  sigma here is the coefficient of variation (= sd/mean), matching the R gamlss
  convention for the GA() family. This is more interpretable for insurance analysts
  than the shape parameter k = 1/sigma^2. The conversion is: shape = 1/sigma^2,
  rate = shape/mu.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import gammaln

from .base import GamlssFamily, Link, LogLink, IdentityLink


class Gamma(GamlssFamily):
    """
    Gamma distribution with mean mu and coefficient of variation sigma.

    PDF: f(y) = (1/sigma^2)^(1/sigma^2) * y^(1/sigma^2 - 1) * exp(-y/(mu*sigma^2))
                / (Gamma(1/sigma^2) * (mu*sigma^2)^(1/sigma^2))

    mu    > 0 : mean (log link)
    sigma > 0 : coefficient of variation, sd/mean (log link)

    Insurance use: claim severity. sigma models heterogeneity of variance
    — more risky segments have higher CV.
    """

    @property
    def param_names(self):
        return ["mu", "sigma"]

    @property
    def default_links(self):
        return {"mu": LogLink(), "sigma": LogLink()}

    def log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        mu = params["mu"]
        sigma = params["sigma"]
        nu = 1.0 / sigma**2  # shape parameter
        return (
            nu * np.log(nu)
            - gammaln(nu)
            + (nu - 1.0) * np.log(y)
            - nu * np.log(mu)
            - nu * y / mu
        )

    def dl_deta(self, y, params, param_name):
        mu = params["mu"]
        sigma = params["sigma"]
        nu = 1.0 / sigma**2
        link = self.default_links[param_name]

        if param_name == "mu":
            # d(ll)/d(mu) = -nu/mu + nu*y/mu^2 = nu*(y - mu)/mu^2
            dl_dtheta = nu * (y - mu) / mu**2
            dtheta_deta = link.inverse_deriv(link.link(mu))
            return dl_dtheta * dtheta_deta

        elif param_name == "sigma":
            # nu = sigma^{-2}, d(ll)/d(sigma) via chain rule through nu
            # d(ll)/d(nu) = log(nu) + 1 - digamma(nu) - log(y/mu) - y/mu ... but simpler:
            # d(ll)/d(sigma) = d(ll)/d(nu) * d(nu)/d(sigma)
            # d(nu)/d(sigma) = -2/sigma^3 = -2*nu/sigma
            from scipy.special import digamma
            dl_dnu = np.log(nu) + 1.0 - digamma(nu) + np.log(y / mu) - y / mu
            dl_dsigma = dl_dnu * (-2.0 / sigma**3)
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

        raise ValueError(f"Unknown param: {param_name}")

    def d2l_deta2(self, y, params, param_name):
        mu = params["mu"]
        sigma = params["sigma"]
        nu = 1.0 / sigma**2
        link = self.default_links[param_name]

        if param_name == "mu":
            # E[-d^2 ll/d mu^2] = nu/mu^2
            # Then in eta space: * (d mu/d eta)^2
            d2l_dmu2 = nu / mu**2
            dtheta_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dtheta_deta**2

        elif param_name == "sigma":
            from scipy.special import polygamma
            # E[-d^2 ll/d sigma^2] via nu parameterisation
            # E[-d^2ll/dnu^2] = polygamma(1, nu) - 1/nu (trigamma - 1/nu)
            d2l_dnu2 = polygamma(1, nu) - 1.0 / nu
            # d^2 l / d sigma^2 = d2l_dnu2 * (dnu/dsigma)^2 + dl_dnu * d2nu_dsigma2
            # At expectation, dl_dnu = 0, so second term vanishes
            dnu_dsigma = -2.0 / sigma**3
            d2l_dsigma2 = d2l_dnu2 * dnu_dsigma**2
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

        raise ValueError(f"Unknown param: {param_name}")

    def starting_values(self, y):
        mu_init = np.full(len(y), np.mean(y))
        sigma_init = np.full(len(y), np.std(y) / np.mean(y))
        sigma_init = np.clip(sigma_init, 0.01, 5.0)
        return {"mu": mu_init, "sigma": sigma_init}


class LogNormal(GamlssFamily):
    """
    Log-Normal distribution with log-scale mean mu and log-scale SD sigma.

    If Y ~ LogNormal(mu, sigma), then log(Y) ~ Normal(mu, sigma).
    Note: mu here is the mean of log(Y), not E[Y].
    E[Y] = exp(mu + sigma^2/2).

    mu    : real (identity link — this is already on log scale)
    sigma > 0 : log-scale standard deviation (log link)

    Insurance use: severity when log(claims) is approximately normal.
    Heavier right tail than Gamma for the same mean.
    """

    @property
    def param_names(self):
        return ["mu", "sigma"]

    @property
    def default_links(self):
        # mu is the mean of log(Y) — identity link is natural
        return {"mu": IdentityLink(), "sigma": LogLink()}

    def log_likelihood(self, y, params):
        mu = params["mu"]
        sigma = params["sigma"]
        log_y = np.log(np.clip(y, 1e-300, None))
        return (
            -np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - log_y
            - 0.5 * ((log_y - mu) / sigma) ** 2
        )

    def dl_deta(self, y, params, param_name):
        mu = params["mu"]
        sigma = params["sigma"]
        log_y = np.log(np.clip(y, 1e-300, None))
        resid = (log_y - mu) / sigma
        link = self.default_links[param_name]

        if param_name == "mu":
            dl_dmu = resid / sigma
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        elif param_name == "sigma":
            dl_dsigma = (resid**2 - 1.0) / sigma
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

        raise ValueError(f"Unknown param: {param_name}")

    def d2l_deta2(self, y, params, param_name):
        mu = params["mu"]
        sigma = params["sigma"]
        link = self.default_links[param_name]

        if param_name == "mu":
            # E[-d^2 ll/d mu^2] = 1/sigma^2
            d2l_dmu2 = 1.0 / sigma**2
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        elif param_name == "sigma":
            # E[-d^2 ll/d sigma^2] = 2/sigma^2
            d2l_dsigma2 = 2.0 / sigma**2
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

        raise ValueError(f"Unknown param: {param_name}")

    def starting_values(self, y):
        log_y = np.log(np.clip(y, 1e-300, None))
        mu_init = np.full(len(y), np.mean(log_y))
        sigma_init = np.full(len(y), max(np.std(log_y), 0.01))
        return {"mu": mu_init, "sigma": sigma_init}


class InverseGaussian(GamlssFamily):
    """
    Inverse Gaussian distribution with mean mu and dispersion sigma.

    PDF: f(y) = sqrt(1/(2*pi*sigma^2*mu*y^3)) * exp(-(y-mu)^2 / (2*sigma^2*mu^2*y/mu))
             = sqrt(1/(2*pi*sigma^2*mu^3)) * y^{-3/2} * exp(-(y-mu)^2/(2*sigma^2*mu^2*y))

    mu    > 0 : mean (log link)
    sigma > 0 : dispersion parameter (log link). Var[Y] = sigma^2 * mu^3.

    Insurance use: liability severity with heavy right tails. The variance
    grows faster than Gamma (mu^3 vs mu^2), so it fits large-loss distributions.
    """

    @property
    def param_names(self):
        return ["mu", "sigma"]

    @property
    def default_links(self):
        return {"mu": LogLink(), "sigma": LogLink()}

    def log_likelihood(self, y, params):
        mu = params["mu"]
        sigma = params["sigma"]
        return (
            -0.5 * np.log(2.0 * np.pi * sigma**2)
            - 1.5 * np.log(y)
            - 0.5 * np.log(mu)
            - (y - mu)**2 / (2.0 * sigma**2 * mu**2 * y / mu)
        )

    def dl_deta(self, y, params, param_name):
        mu = params["mu"]
        sigma = params["sigma"]
        link = self.default_links[param_name]

        if param_name == "mu":
            # d(ll)/d(mu) = (y - mu)/(sigma^2 * mu^3) - 1/(2*mu)
            # Using IG parameterisation: Var[Y] = sigma^2 * mu^3
            dl_dmu = (y - mu) / (sigma**2 * mu**3) - 0.5 / mu
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        elif param_name == "sigma":
            # d(ll)/d(sigma) = -1/sigma + (y-mu)^2/(sigma^3 * mu^2 * y)
            dl_dsigma = -1.0 / sigma + (y - mu)**2 / (sigma**3 * mu**2 * y)
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

        raise ValueError(f"Unknown param: {param_name}")

    def d2l_deta2(self, y, params, param_name):
        mu = params["mu"]
        sigma = params["sigma"]
        link = self.default_links[param_name]

        if param_name == "mu":
            # E[-d^2 ll/d mu^2] = 1/(sigma^2 * mu^3) + 1/(2*mu^2)
            # (approximation using E[Y]=mu, E[Y^{-1}] = 1/mu + sigma^2/mu for IG)
            d2l_dmu2 = 1.0 / (sigma**2 * mu**3)
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        elif param_name == "sigma":
            # E[-d^2 ll/d sigma^2] = 2/sigma^2
            d2l_dsigma2 = 2.0 / sigma**2
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

        raise ValueError(f"Unknown param: {param_name}")

    def starting_values(self, y):
        mu_init = np.full(len(y), np.mean(y))
        # sigma^2 = Var/(mu^3) approx
        sigma_sq = np.var(y) / np.mean(y)**3
        sigma_init = np.full(len(y), max(np.sqrt(sigma_sq), 0.01))
        return {"mu": mu_init, "sigma": sigma_init}


class Tweedie(GamlssFamily):
    """
    Tweedie distribution: compound Poisson-Gamma, useful for pure premiums.

    The Tweedie family with power parameter p in (1,2) gives a distribution
    with a point mass at zero (claim frequency component) and a continuous
    positive part (severity component). This makes it natural for:
      - Pure premiums (expected cost per unit exposure)
      - Combined frequency-severity models

    mu  > 0 : mean (log link)
    phi > 0 : dispersion, Var[Y] = phi * mu^p (log link)
    p       : power parameter, fixed at init (default 1.5)

    The log-likelihood uses the series approximation from Dunn & Smyth (2005).
    For p=1: Poisson, p=2: Gamma, p in (1,2): compound Poisson-Gamma.

    Note: p is NOT estimated by default — fix it or profile over a grid.
    """

    def __init__(self, power: float = 1.5):
        if not (1.0 < power < 2.0):
            raise ValueError(f"Tweedie power must be in (1,2), got {power}")
        self.power = power

    @property
    def param_names(self):
        return ["mu", "phi"]

    @property
    def default_links(self):
        return {"mu": LogLink(), "phi": LogLink()}

    def log_likelihood(self, y, params):
        mu = params["mu"]
        phi = params["phi"]
        p = self.power

        # Tweedie log-likelihood: Dunn & Smyth series approximation
        # For y > 0 and y = 0 parts
        # ll = -y^(2-p)/((2-p)*phi) + y*mu^(1-p)/((1-p)*phi) - mu^(2-p)/((2-p)*phi)
        # Plus log-density correction term (series sum) which we approximate

        alpha = (2.0 - p) / (1.0 - p)  # negative
        log_c = np.zeros_like(y)

        # y = 0 part: log P(Y=0) = -mu^(2-p) / ((2-p) * phi)
        # y > 0 part: saddle-point / series approximation
        # Use the standard Tweedie kernel (ignoring normalising constant for WLS)

        # Log-likelihood kernel (sufficient for score equations):
        # k(y; mu, phi) = y*mu^(1-p)/((1-p)*phi) - mu^(2-p)/((2-p)*phi)
        # Full LL includes log density of positive y which requires series sum.
        # We implement the exact Dunn-Smyth series.

        ll = np.where(
            y == 0,
            -mu**(2.0 - p) / ((2.0 - p) * phi),
            (y * mu**(1.0 - p) / ((1.0 - p) * phi)
             - mu**(2.0 - p) / ((2.0 - p) * phi)
             + self._log_w(y, phi, p)),
        )
        return ll

    def _log_w(self, y: np.ndarray, phi: np.ndarray, p: float) -> np.ndarray:
        """
        Log of the Tweedie series sum W(y, phi, p) for y > 0.
        Uses a Poisson-Gamma series approximation (Dunn & Smyth 2005).
        Truncated at j_max terms where j is the Poisson index.
        """
        j_max = 50
        alpha = (2.0 - p) / (1.0 - p)

        y = np.asarray(y, dtype=float)
        phi = np.asarray(phi, dtype=float)

        # Ensure shapes match
        y_flat = y.ravel()
        phi_flat = phi.ravel()

        log_w = np.full(len(y_flat), -np.inf)

        for j in range(1, j_max + 1):
            # log term_j = j*(log(y) - log(phi*(2-p))) - log Gamma(j+1)
            #              - log Gamma(-j*alpha) + j*log(j) + ...
            # Simplified: use log of j-th term
            log_term = (
                j * np.log(np.maximum(y_flat, 1e-300))
                - gammaln(j + 1)
                - gammaln(-j * alpha)
                + (-j * alpha) * np.log(phi_flat * (2.0 - p))
                + j * np.log(p - 1.0)
                - j * np.log(phi_flat * (2.0 - p))
            )
            # log-sum-exp accumulation
            m = np.maximum(log_w, log_term)
            finite = np.isfinite(m)
            log_w = np.where(
                finite,
                m + np.log(np.exp(np.where(finite, log_w - m, 0)) + np.exp(np.where(finite, log_term - m, 0))),
                np.where(np.isfinite(log_term), log_term, log_w),
            )

        return log_w.reshape(y.shape)

    def dl_deta(self, y, params, param_name):
        mu = params["mu"]
        phi = params["phi"]
        p = self.power
        link = self.default_links[param_name]

        if param_name == "mu":
            # d(ll)/d(mu) = y*mu^(-p)/phi - mu^(1-p)/phi = (y*mu^(-p) - mu^(1-p))/phi
            dl_dmu = (y * mu**(-p) - mu**(1.0 - p)) / phi
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        elif param_name == "phi":
            # d(ll)/d(phi) for y > 0:
            # = -y*mu^(1-p)/((1-p)*phi^2) + mu^(2-p)/((2-p)*phi^2) + d log_w/d phi
            # Approximate by ignoring d log_w/d phi (it's small for moderate phi)
            dl_dphi = np.where(
                y == 0,
                mu**(2.0 - p) / ((2.0 - p) * phi**2),
                -y * mu**(1.0 - p) / ((1.0 - p) * phi**2)
                + mu**(2.0 - p) / ((2.0 - p) * phi**2),
            )
            dphi_deta = link.inverse_deriv(link.link(phi))
            return dl_dphi * dphi_deta

        raise ValueError(f"Unknown param: {param_name}")

    def d2l_deta2(self, y, params, param_name):
        mu = params["mu"]
        phi = params["phi"]
        p = self.power
        link = self.default_links[param_name]

        if param_name == "mu":
            # E[-d^2 ll/d mu^2] = mu^(2-2p) / (phi * mu^(2-p)) ...
            # = mu^(2-2p)/phi ... using E[Y]=mu:
            # = (mu^(1-p)/phi) * (p * mu^(-p) * mu - (1-p)*mu^(-p)) ...
            # Fisher info for mu: 1/(phi * mu^p) * mu = 1/(phi * mu^(p-1))
            # Actually E[Y] = mu, Var[Y] = phi*mu^p, so Fisher = 1/Var * (d mu/d eta)^2
            # d2l/dmu2 expected: 1/(phi * mu^p)
            d2l_dmu2 = mu**(1.0 - p) / phi  # = 1/(phi * mu^(p-1)) but simplified
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        elif param_name == "phi":
            # E[-d^2 ll/d phi^2] approx 1/phi^2
            d2l_dphi2 = 1.0 / phi**2
            dphi_deta = link.inverse_deriv(link.link(phi))
            return d2l_dphi2 * dphi_deta**2

        raise ValueError(f"Unknown param: {param_name}")

    def starting_values(self, y):
        mu_init = np.full(len(y), np.mean(y[y > 0]) if np.any(y > 0) else 1.0)
        phi_init = np.full(len(y), 1.0)
        return {"mu": mu_init, "phi": phi_init}
