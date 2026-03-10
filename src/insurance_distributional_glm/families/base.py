"""
Base classes for GAMLSS families and link functions.

Link functions implement the relationship between the linear predictor eta
and the distribution parameter theta: eta = g(theta), theta = g^{-1}(eta).

Every GamlssFamily subclass must provide log_likelihood, derivatives for
the RS algorithm's IRLS steps, and starting values. The design keeps
family code self-contained — adding a new distribution means subclassing
GamlssFamily without touching the fitting engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Link functions
# ---------------------------------------------------------------------------

class Link(ABC):
    """Abstract link function g: parameter space -> real line."""

    @abstractmethod
    def link(self, theta: np.ndarray) -> np.ndarray:
        """Forward: eta = g(theta)."""

    @abstractmethod
    def inverse(self, eta: np.ndarray) -> np.ndarray:
        """Inverse: theta = g^{-1}(eta)."""

    @abstractmethod
    def deriv(self, theta: np.ndarray) -> np.ndarray:
        """Derivative: d(eta)/d(theta) = g'(theta)."""

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        """d(theta)/d(eta) = 1 / g'(g^{-1}(eta))."""
        theta = self.inverse(eta)
        return 1.0 / self.deriv(theta)

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LogLink(Link):
    """Log link: eta = log(theta), theta = exp(eta). Used for positive parameters."""

    def link(self, theta: np.ndarray) -> np.ndarray:
        return np.log(theta)

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(eta, -30, 30))

    def deriv(self, theta: np.ndarray) -> np.ndarray:
        return 1.0 / theta

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(eta, -30, 30))

    @property
    def name(self) -> str:
        return "log"


class LogitLink(Link):
    """Logit link: eta = log(theta/(1-theta)), theta = expit(eta). Used for probabilities."""

    def link(self, theta: np.ndarray) -> np.ndarray:
        theta_clipped = np.clip(theta, 1e-10, 1 - 1e-10)
        return np.log(theta_clipped / (1.0 - theta_clipped))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

    def deriv(self, theta: np.ndarray) -> np.ndarray:
        theta_clipped = np.clip(theta, 1e-10, 1 - 1e-10)
        return 1.0 / (theta_clipped * (1.0 - theta_clipped))

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        p = self.inverse(eta)
        return p * (1.0 - p)

    @property
    def name(self) -> str:
        return "logit"


class IdentityLink(Link):
    """Identity link: eta = theta. Used for location parameters on the original scale."""

    def link(self, theta: np.ndarray) -> np.ndarray:
        return theta.copy()

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return eta.copy()

    def deriv(self, theta: np.ndarray) -> np.ndarray:
        return np.ones_like(theta)

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)

    @property
    def name(self) -> str:
        return "identity"


class SqrtLink(Link):
    """Square root link: eta = sqrt(theta), theta = eta^2."""

    def link(self, theta: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(theta, 0, None))

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        return np.clip(eta, 0, None) ** 2

    def deriv(self, theta: np.ndarray) -> np.ndarray:
        return 0.5 / np.sqrt(np.clip(theta, 1e-10, None))

    def inverse_deriv(self, eta: np.ndarray) -> np.ndarray:
        return 2.0 * np.clip(eta, 0, None)

    @property
    def name(self) -> str:
        return "sqrt"


# Singleton instances for convenience
log = LogLink()
logit = LogitLink()
identity = IdentityLink()
sqrt = SqrtLink()


# ---------------------------------------------------------------------------
# GAMLSS family base class
# ---------------------------------------------------------------------------

class GamlssFamily(ABC):
    """
    Abstract base for GAMLSS distribution families.

    Each subclass implements a specific distribution and its relationship
    to the RS fitting algorithm. The key quantities are:
      - log_likelihood: total log-likelihood (sum over obs)
      - dl_deta: d(log L)/d(eta_k) — the score for the IRLS working response
      - d2l_deta2: E[d^2(log L)/d(eta_k)^2] — negative for IRLS weights

    These drive the weighted least squares step in the RS algorithm.
    """

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Ordered list of distribution parameter names, e.g. ['mu', 'sigma']."""

    @property
    @abstractmethod
    def default_links(self) -> Dict[str, Link]:
        """Default link function for each parameter."""

    @abstractmethod
    def log_likelihood(self, y: np.ndarray, params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Per-observation log-likelihood.

        Returns array of shape (n,), one value per observation.
        The fitting engine sums these to get total log-likelihood.
        """

    @abstractmethod
    def dl_deta(
        self,
        y: np.ndarray,
        params: Dict[str, np.ndarray],
        param_name: str,
    ) -> np.ndarray:
        """
        d(log L_i)/d(eta_k) for parameter param_name.

        This is the derivative of log-likelihood with respect to the linear
        predictor eta_k (not theta_k). Used to construct the working response
        in the IRLS step.

        By chain rule: d(log L)/d(eta_k) = d(log L)/d(theta_k) * d(theta_k)/d(eta_k)
        """

    @abstractmethod
    def d2l_deta2(
        self,
        y: np.ndarray,
        params: Dict[str, np.ndarray],
        param_name: str,
    ) -> np.ndarray:
        """
        Expected -d^2(log L_i)/d(eta_k)^2 for parameter param_name.

        Returns positive values (this is the negative expected second derivative).
        Used as IRLS weights. If analytic expectation is hard, use the observed
        Fisher information (negative of actual second derivative).
        """

    @abstractmethod
    def starting_values(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Initial parameter estimates from the marginal distribution of y.

        Called once before the RS loop begins. Should be robust — moment
        estimates are fine here; precision doesn't matter much.
        """

    def validate_params(self, params: Dict[str, np.ndarray]) -> None:
        """Optional validation hook. Raise ValueError if params are out of range."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(links={{{', '.join(f'{k}: {v.name}' for k, v in self.default_links.items())}}})"
