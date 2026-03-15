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
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]
        nu = 1.0 / sigma**2

        if param_name == "mu":
            link = self.default_links["mu"]
            dl_dtheta = nu * (y - mu) / mu**2
            dtheta_deta = link.inverse_deriv(link.link(mu))
            return dl_dtheta * dtheta_deta

        else:  # sigma
            from scipy.special import digamma
            link = self.default_links["sigma"]
            dl_dnu = np.log(nu) + 1.0 - digamma(nu) + np.log(y / mu) - y / mu
            dl_dsigma = dl_dnu * (-2.0 / sigma**3)
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]
        nu = 1.0 / sigma**2

        if param_name == "mu":
            link = self.default_links["mu"]
            d2l_dmu2 = nu / mu**2
            dtheta_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dtheta_deta**2

        else:  # sigma
            from scipy.special import polygamma
            link = self.default_links["sigma"]
            d2l_dnu2 = polygamma(1, nu) - 1.0 / nu
            dnu_dsigma = -2.0 / sigma**3
            d2l_dsigma2 = d2l_dnu2 * dnu_dsigma**2
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

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
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]
        log_y = np.log(np.clip(y, 1e-300, None))
        resid = (log_y - mu) / sigma

        if param_name == "mu":
            link = self.default_links["mu"]
            dl_dmu = resid / sigma
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        else:  # sigma
            link = self.default_links["sigma"]
            dl_dsigma = (resid**2 - 1.0) / sigma
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]

        if param_name == "mu":
            link = self.default_links["mu"]
            d2l_dmu2 = 1.0 / sigma**2
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        else:  # sigma
            link = self.default_links["sigma"]
            d2l_dsigma2 = 2.0 / sigma**2
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

    def starting_values(self, y):
        log_y = np.log(np.clip(y, 1e-300, None))
        mu_init = np.full(len(y), np.mean(log_y))
        sigma_init = np.full(len(y), max(np.std(log_y), 0.01))
        return {"mu": mu_init, "sigma": sigma_init}


class InverseGaussian(GamlssFamily):
    """
    Inverse Gaussian distribution with mean mu and dispersion sigma.

    Standard parameterisation: if lambda = 1/sigma^2 is the shape (precision)
    parameter, then:
      PDF: f(y) = sqrt(lambda/(2*pi*y^3)) * exp(-lambda*(y-mu)^2/(2*mu^2*y))
      Var[Y] = mu^3 / lambda = sigma^2 * mu^3

    mu    > 0 : mean (log link)
    sigma > 0 : dispersion, Var = sigma^2 * mu^3 (log link)

    Log-likelihood:
      ll = -log(sigma) - 0.5*log(2*pi) - 1.5*log(y) - (y-mu)^2/(2*sigma^2*mu^2*y)

    Insurance use: liability severity with heavy right tails. The variance
    grows faster with mu than Gamma (mu^3 vs mu^2), suitable for large losses.
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
        y_safe = np.clip(y, 1e-300, None)
        return (
            -np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 1.5 * np.log(y_safe)
            - (y - mu) ** 2 / (2.0 * sigma**2 * mu**2 * y_safe)
        )

    def dl_deta(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]
        y_safe = np.clip(y, 1e-300, None)

        if param_name == "mu":
            link = self.default_links["mu"]
            # d(ll)/d(mu):
            # ll contains -(y-mu)^2/(2*sigma^2*mu^2*y)
            # Expand: -(y^2/mu^2 - 2y/mu + 1)/(2*sigma^2*y)
            # d/dmu = (2y^2/mu^3 - 2y/mu^2)/(2*sigma^2*y) = (y-mu)/(sigma^2*mu^3)
            dl_dmu = (y - mu) / (sigma**2 * mu**3)
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        else:  # sigma
            link = self.default_links["sigma"]
            # d(ll)/d(sigma) = -1/sigma + (y-mu)^2/(sigma^3*mu^2*y)
            dl_dsigma = -1.0 / sigma + (y - mu)**2 / (sigma**3 * mu**2 * y_safe)
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]

        if param_name == "mu":
            link = self.default_links["mu"]
            # E[-d^2(ll)/dmu^2] = 1/(sigma^2*mu^3)
            d2l_dmu2 = 1.0 / (sigma**2 * mu**3)
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        else:  # sigma
            link = self.default_links["sigma"]
            # E[-d^2 ll/d sigma^2] = 2/sigma^2
            d2l_dsigma2 = 2.0 / sigma**2
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

    def starting_values(self, y):
        mu_init = np.full(len(y), np.mean(y))
        sigma_sq = max(np.var(y) / np.mean(y)**3, 1e-6)
        sigma_init = np.full(len(y), np.sqrt(sigma_sq))
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

        Uses the compound Poisson-Gamma series from Dunn & Smyth (2005).
        Each term j corresponds to a Poisson count of j severity events.

        For fixed y > 0 the density is:
          f(y) * exp(ll_front_inverse) = (1/y) * sum_j V_j
        where:
          V_j = lam^j * y^{j*alpha - 1} / (j! * scale^{j*alpha} * Gamma(j*alpha))
          lam   = mu^{2-p} / (phi * (2-p))   [Poisson mean]
          scale = phi * (p-1) * mu^{p-1}      [Gamma scale per event]
          alpha = (2-p) / (p-1)               [> 0 for p in (1,2)]

        The log-offset terms for lam and scale simplify to:
          j*log(lam) - j*alpha*log(scale) = -j*log(phi*(2-p)) - j*alpha*log(phi*(p-1))
        (the mu terms cancel because alpha*(p-1) = 2-p exactly).

        Per-term log formula:
          log(W_j) = (j*alpha - 1)*log(y) - gammaln(j*alpha) - gammaln(j+1)
                     - j*log(phi*(2-p)) - j*alpha*log(phi*(p-1))
        """
        j_max = 50
        alpha = (2.0 - p) / (p - 1.0)  # positive for p in (1, 2)

        y_flat = np.asarray(y, dtype=float).ravel()
        phi_flat = np.asarray(phi, dtype=float).ravel()

        log_phi_2mp = np.log(phi_flat * (2.0 - p))
        log_phi_pm1 = np.log(phi_flat * (p - 1.0))
        log_y = np.log(np.maximum(y_flat, 1e-300))

        log_w = np.full(len(y_flat), -np.inf)

        for j in range(1, j_max + 1):
            log_term = (
                (j * alpha - 1.0) * log_y
                - gammaln(j * alpha)
                - gammaln(j + 1)
                - j * log_phi_2mp
                - j * alpha * log_phi_pm1
            )
            m = np.maximum(log_w, log_term)
            finite = np.isfinite(m)
            log_w = np.where(
                finite,
                m + np.log(
                    np.exp(np.where(finite, log_w - m, 0))
                    + np.exp(np.where(finite, log_term - m, 0))
                ),
                np.where(np.isfinite(log_term), log_term, log_w),
            )

        return log_w.reshape(np.asarray(y).shape)

    def dl_deta(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        phi = params["phi"]
        p = self.power

        if param_name == "mu":
            link = self.default_links["mu"]
            dl_dmu = (y * mu**(-p) - mu**(1.0 - p)) / phi
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        else:  # phi
            link = self.default_links["phi"]
            dl_dphi = np.where(
                y == 0,
                mu**(2.0 - p) / ((2.0 - p) * phi**2),
                -y * mu**(1.0 - p) / ((1.0 - p) * phi**2)
                + mu**(2.0 - p) / ((2.0 - p) * phi**2),
            )
            dphi_deta = link.inverse_deriv(link.link(phi))
            return dl_dphi * dphi_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        phi = params["phi"]
        p = self.power

        if param_name == "mu":
            link = self.default_links["mu"]
            # E[-d^2 ll/d mu^2] = mu^{-p} / phi
            # Derived: d^2ll/dmu^2 = -p*y*mu^{-p-1}/phi + (p-1)*mu^{-p}/phi
            # Taking expectation with E[y]=mu: E[-d^2ll/dmu^2] = mu^{-p}/phi
            d2l_dmu2 = mu ** (-p) / phi
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        else:  # phi
            link = self.default_links["phi"]
            # E[-d^2 ll/d phi^2] = 2 * mu^{2-p} / (phi^3 * (p-1) * (2-p))
            # Derived: d^2ll/dphi^2 = 2*y*mu^{1-p}/((1-p)*phi^3) - 2*mu^{2-p}/((2-p)*phi^3)
            # Taking expectation with E[y]=mu:
            #   E[-d^2ll/dphi^2] = -2*mu^{2-p}/((1-p)*phi^3) + 2*mu^{2-p}/((2-p)*phi^3)
            #                    = 2*mu^{2-p}/phi^3 * [1/(p-1) + 1/(2-p)]
            #                    = 2*mu^{2-p}/(phi^3 * (p-1) * (2-p))
            d2l_dphi2 = 2.0 * mu ** (2.0 - p) / (phi**3 * (p - 1.0) * (2.0 - p))
            dphi_deta = link.inverse_deriv(link.link(phi))
            return d2l_dphi2 * dphi_deta**2

    def starting_values(self, y):
        mu_init = np.full(len(y), np.mean(y[y > 0]) if np.any(y > 0) else 1.0)
        phi_init = np.full(len(y), 1.0)
        return {"mu": mu_init, "phi": phi_init}
