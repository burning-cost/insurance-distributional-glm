"""
Count distribution families for GAMLSS.

Insurance frequency distributions:
  - Poisson: baseline frequency model, equidispersed.
  - NBI (Negative Binomial Type I): overdispersed frequency. sigma is
    the overdispersion parameter — standard in insurance for claims count.
  - ZIP (Zero-Inflated Poisson): handles excess zeros from non-claimants
    or policy-year effects.

NBI parameterisation note:
  The NBI family uses Var[Y] = mu + sigma*mu^2, where sigma > 0 is the
  overdispersion. At sigma=0 this collapses to Poisson. This is the
  "Type I" negative binomial (constant overdispersion), distinct from the
  NB2 (variance = mu + mu^2/k) convention in some texts. Our sigma = 1/k.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import gammaln, xlogy

from .base import GamlssFamily, Link, LogLink, LogitLink


class Poisson(GamlssFamily):
    """
    Poisson distribution for claim frequency.

    mu > 0 : rate parameter (log link)

    One-parameter family — the RS algorithm only updates mu.
    Use NBI if your data shows overdispersion (most insurance data does).
    """

    @property
    def param_names(self):
        return ["mu"]

    @property
    def default_links(self):
        return {"mu": LogLink()}

    def log_likelihood(self, y, params):
        mu = params["mu"]
        return xlogy(y, mu) - mu - gammaln(y + 1.0)

    def dl_deta(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        link = self.default_links["mu"]
        # d(ll)/d(mu) = (y - mu) / mu; with log link, d(mu)/d(eta) = mu
        # so dl/d(eta) = (y - mu) / mu * mu = y - mu
        dl_dmu = (y - mu) / mu
        dmu_deta = link.inverse_deriv(link.link(mu))
        return dl_dmu * dmu_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        link = self.default_links["mu"]
        # E[-d^2 ll/d mu^2] = 1/mu; with log link dmu/deta=mu: E[-d^2/deta^2] = mu^2/mu = mu
        d2l_dmu2 = 1.0 / mu
        dmu_deta = link.inverse_deriv(link.link(mu))
        return d2l_dmu2 * dmu_deta**2

    def starting_values(self, y):
        mu_init = np.full(len(y), max(np.mean(y), 0.01))
        return {"mu": mu_init}


class NBI(GamlssFamily):
    """
    Negative Binomial Type I (NBI) for overdispersed claim counts.

    Parameterisation: Var[Y] = mu + sigma * mu^2
      mu    > 0 : mean (log link)
      sigma > 0 : overdispersion parameter (log link)

    PMF: P(Y=y) = Gamma(y + 1/sigma) / (y! * Gamma(1/sigma))
                  * (mu*sigma/(1+mu*sigma))^y * (1/(1+mu*sigma))^(1/sigma)

    Insurance use: the standard choice when claim counts are overdispersed,
    which is almost always true in motor and property portfolios. sigma
    captures unobserved heterogeneity not captured by the covariates.
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
        k = 1.0 / sigma  # shape / reciprocal overdispersion

        return (
            gammaln(y + k)
            - gammaln(k)
            - gammaln(y + 1.0)
            + k * np.log(k / (k + mu))
            + xlogy(y, mu / (mu + k))
        )

    def dl_deta(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]
        k = 1.0 / sigma

        if param_name == "mu":
            link = self.default_links["mu"]
            # d(ll)/d(mu) = y/mu - (y+k)/(mu+k)
            dl_dmu = y / mu - (y + k) / (mu + k)
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        else:  # sigma
            from scipy.special import digamma
            link = self.default_links["sigma"]
            dl_dk = (
                digamma(y + k)
                - digamma(k)
                + np.log(k / (k + mu))
                + 1.0
                - (y + k) / (k + mu)
            )
            dk_dsigma = -1.0 / sigma**2
            dl_dsigma = dl_dk * dk_dsigma
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return dl_dsigma * dsigma_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        sigma = params["sigma"]
        k = 1.0 / sigma

        if param_name == "mu":
            link = self.default_links["mu"]
            # E[-d^2 ll/d mu^2] = k/(mu*(mu+k))
            d2l_dmu2 = k / (mu * (mu + k))
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        else:  # sigma
            from scipy.special import polygamma
            link = self.default_links["sigma"]
            # E[-d^2 ll/d k^2] = polygamma(1, k) - 1/k - 1/(k+mu)
            d2l_dk2 = polygamma(1, k) - 1.0 / k - 1.0 / (k + mu)
            dk_dsigma = -1.0 / sigma**2
            d2l_dsigma2 = np.maximum(d2l_dk2 * dk_dsigma**2, 1e-8)
            dsigma_deta = link.inverse_deriv(link.link(sigma))
            return d2l_dsigma2 * dsigma_deta**2

    def starting_values(self, y):
        mu_val = max(np.mean(y), 0.01)
        var_val = max(np.var(y), mu_val + 1e-6)
        sigma_val = max((var_val - mu_val) / mu_val**2, 0.01)
        return {
            "mu": np.full(len(y), mu_val),
            "sigma": np.full(len(y), sigma_val),
        }


class ZIP(GamlssFamily):
    """
    Zero-Inflated Poisson (ZIP) for count data with excess zeros.

    Parameterisation:
      mu  > 0       : Poisson rate for the non-zero-inflated component (log link)
      pi  in (0,1)  : probability of structural zero (logit link)

    P(Y=0) = pi + (1-pi)*exp(-mu)
    P(Y=y) = (1-pi) * Poisson(y; mu)  for y > 0

    Insurance use: portfolios where a proportion of risks have zero claims
    due to non-exposure (parked vehicles, seasonal risks). pi can then be
    modelled as a function of coverage period or risk activity covariates.

    Implementation note: the zero component introduces separate score
    equations for mu and pi, with different forms for y=0 and y>0.
    """

    @property
    def param_names(self):
        return ["mu", "pi"]

    @property
    def default_links(self):
        return {"mu": LogLink(), "pi": LogitLink()}

    def log_likelihood(self, y, params):
        mu = params["mu"]
        pi = params["pi"]
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        ll_zero = np.log(pi + (1.0 - pi) * np.exp(-mu))
        ll_pos = np.log(1.0 - pi) + xlogy(y, mu) - mu - gammaln(y + 1.0)

        return np.where(y == 0, ll_zero, ll_pos)

    def dl_deta(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        pi = params["pi"]
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        p0 = pi + (1.0 - pi) * np.exp(-mu)
        p0 = np.clip(p0, 1e-300, None)

        if param_name == "mu":
            link = self.default_links["mu"]
            dl_dmu = np.where(
                y == 0,
                -(1.0 - pi) * np.exp(-mu) / p0,
                y / mu - 1.0,
            )
            dmu_deta = link.inverse_deriv(link.link(mu))
            return dl_dmu * dmu_deta

        else:  # pi
            link = self.default_links["pi"]
            dl_dpi = np.where(
                y == 0,
                (1.0 - np.exp(-mu)) / p0,
                -1.0 / (1.0 - pi),
            )
            dpi_deta = link.inverse_deriv(link.link(pi))
            return dl_dpi * dpi_deta

    def d2l_deta2(self, y, params, param_name):
        self._check_param(param_name)
        mu = params["mu"]
        pi = params["pi"]
        pi = np.clip(pi, 1e-10, 1 - 1e-10)

        p0 = pi + (1.0 - pi) * np.exp(-mu)
        p0 = np.clip(p0, 1e-300, None)
        exp_neg_mu = np.exp(-mu)

        if param_name == "mu":
            link = self.default_links["mu"]
            d2_zero = -((1.0 - pi) * exp_neg_mu * p0 - ((1.0 - pi) * exp_neg_mu)**2) / p0**2
            d2_pos = -y / mu**2
            d2l_dmu2 = np.where(y == 0, -d2_zero, -d2_pos)
            d2l_dmu2 = np.maximum(np.abs(d2l_dmu2), 1e-8)
            dmu_deta = link.inverse_deriv(link.link(mu))
            return d2l_dmu2 * dmu_deta**2

        else:  # pi
            link = self.default_links["pi"]
            d2_zero = -((1.0 - exp_neg_mu) * p0 - (1.0 - exp_neg_mu)**2) / p0**2
            d2_pos = -1.0 / (1.0 - pi)**2
            d2l_dpi2 = np.where(y == 0, -d2_zero, -d2_pos)
            d2l_dpi2 = np.maximum(np.abs(d2l_dpi2), 1e-8)
            dpi_deta = link.inverse_deriv(link.link(pi))
            return d2l_dpi2 * dpi_deta**2

    def starting_values(self, y):
        prop_zero = np.mean(y == 0)
        mu_init = max(np.mean(y[y > 0]) if np.any(y > 0) else 1.0, 0.01)
        pi_init = max(prop_zero - np.exp(-mu_init) * (1 - prop_zero), 0.05)
        pi_init = min(pi_init, 0.95)
        return {
            "mu": np.full(len(y), mu_init),
            "pi": np.full(len(y), pi_init),
        }
