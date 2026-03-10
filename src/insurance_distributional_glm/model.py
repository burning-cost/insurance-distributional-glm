"""
DistributionalGLM: the main user-facing class for GAMLSS models.

This wraps the RS fitting engine and provides a scikit-learn-ish API
designed for insurance pricing teams. Key design decisions:

1. The `formulas` dict controls which covariates affect which parameters.
   Most users model mu fully and sigma with an intercept or 1-2 risk factors.
   This is intentional — modelling sigma with too many covariates causes
   identifiability problems in practice.

2. Exposure is applied as a log-offset on mu only. This is the standard
   insurance convention (rate = claims / exposure, model rate with offset).

3. Relativities are computed on the response scale for log-linked parameters,
   matching the GLM output format actuaries expect.

4. The intercept is always added automatically to every component's design
   matrix. Pass formulas={'mu': []} to get intercept-only models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np

from .families.base import GamlssFamily
from .fitting import rs_fit
from .selection import gaic as _gaic


# Type alias for what we accept as DataFrames
try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _to_numpy(X, columns: List[str]) -> np.ndarray:
    """Extract columns from polars/pandas/numpy into a numpy matrix."""
    if _HAS_POLARS and isinstance(X, pl.DataFrame):
        return X.select(columns).to_numpy().astype(float)
    elif _HAS_PANDAS and isinstance(X, pd.DataFrame):
        return X[columns].values.astype(float)
    elif isinstance(X, np.ndarray):
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X.astype(float)
    else:
        raise TypeError(f"X must be a polars/pandas DataFrame or numpy array, got {type(X)}")


def _get_columns(X) -> List[str]:
    """Get column names from a DataFrame, or generate defaults for arrays."""
    if _HAS_POLARS and isinstance(X, pl.DataFrame):
        return X.columns
    elif _HAS_PANDAS and isinstance(X, pd.DataFrame):
        return list(X.columns)
    elif isinstance(X, np.ndarray):
        n_cols = X.shape[1] if X.ndim > 1 else 1
        return [f"x{i}" for i in range(n_cols)]
    raise TypeError(f"Unsupported X type: {type(X)}")


def _build_design_matrix(X, columns: List[str]) -> np.ndarray:
    """Build design matrix with intercept prepended."""
    n = X.shape[0] if hasattr(X, "shape") else len(X)
    if columns:
        mat = _to_numpy(X, columns)
        return np.column_stack([np.ones(n), mat])
    else:
        return np.ones((n, 1))


class DistributionalGLM:
    """
    GAMLSS model: model ALL distribution parameters as functions of covariates.

    Standard GLM models only E[Y|X]. This models the full conditional
    distribution p(Y|X) by allowing each parameter (mean, variance, shape)
    to depend on covariates.

    Example
    -------
    >>> import polars as pl
    >>> from insurance_distributional_glm import DistributionalGLM
    >>> from insurance_distributional_glm.families import Gamma
    >>>
    >>> df = pl.DataFrame({"age": [25, 35, 45], "vehicle_value": [5000, 15000, 30000]})
    >>> y = np.array([500.0, 1200.0, 3500.0])
    >>>
    >>> model = DistributionalGLM(
    ...     family=Gamma(),
    ...     formulas={"mu": ["age", "vehicle_value"], "sigma": ["age"]},
    ... )
    >>> model.fit(df, y)
    >>> model.predict(df, parameter="mu")
    """

    def __init__(
        self,
        family: GamlssFamily,
        formulas: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Parameters
        ----------
        family : GamlssFamily
            Distribution family. Determines which parameters exist and their
            default link functions.
        formulas : dict, optional
            Maps parameter name -> list of column names to include.
            Intercept is always added automatically.
            If None: mu gets all columns, other parameters get intercept-only.
            If {'mu': [], 'sigma': []}: intercept-only for all parameters.
        """
        self.family = family
        self.formulas = formulas
        self._betas: Dict[str, np.ndarray] = {}
        self._params: Dict[str, np.ndarray] = {}
        self._columns: List[str] = []
        self._loglik: float = float("nan")
        self._loglik_history: List[float] = []
        self._converged: bool = False
        self._n: int = 0
        self._fitted = False

    def fit(
        self,
        X,
        y,
        exposure: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> "DistributionalGLM":
        """
        Fit the model via the RS algorithm.

        Parameters
        ----------
        X : polars/pandas DataFrame or numpy array
            Covariate matrix. Column names must match the formulas dict.
        y : array-like
            Response observations. Must be positive for Gamma/LogNormal/IG.
        exposure : array-like, optional
            Exposure values (not log). Applied as log-offset on mu.
        weights : array-like, optional
            Observation weights.
        max_iter : int
            Maximum RS outer iterations (default 100).
        tol : float
            Convergence tolerance on log-likelihood change (default 1e-6).
        verbose : bool
            Print log-likelihood at each RS iteration.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=float)
        self._n = len(y)
        self._columns = _get_columns(X)

        # Resolve formulas
        formulas = self._resolve_formulas()

        # Build design matrices
        design_matrices = {}
        for pname in self.family.param_names:
            cols = formulas.get(pname, [])
            design_matrices[pname] = _build_design_matrix(X, cols)

        # Exposure offset
        log_offset = None
        if exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            log_offset = np.log(np.clip(exposure, 1e-10, None))

        if weights is not None:
            weights = np.asarray(weights, dtype=float)

        # Run RS algorithm
        betas, params, ll_history, converged = rs_fit(
            family=self.family,
            design_matrices=design_matrices,
            y=y,
            log_offset=log_offset,
            weights=weights,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

        self._betas = betas
        self._params = params
        self._loglik_history = ll_history
        self._loglik = ll_history[-1] if ll_history else float("nan")
        self._converged = converged
        self._design_matrices = design_matrices
        self._formulas = formulas
        self._fitted = True

        if not converged and verbose:
            print(f"Warning: RS algorithm did not converge in {max_iter} iterations.")

        return self

    def _resolve_formulas(self) -> Dict[str, List[str]]:
        """Apply default formula logic if formulas is None."""
        param_names = self.family.param_names
        if self.formulas is None:
            result = {}
            for i, pname in enumerate(param_names):
                result[pname] = self._columns if i == 0 else []
            return result
        # Fill in missing params with intercept-only
        result = {}
        for pname in param_names:
            result[pname] = self.formulas.get(pname, [])
        return result

    def predict(
        self,
        X,
        parameter: str = "mu",
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict a distribution parameter on the response scale.

        Parameters
        ----------
        X : DataFrame or array
            Covariates.
        parameter : str
            Which parameter to predict (e.g. 'mu', 'sigma', 'pi').
        exposure : array-like, optional
            Exposure offset for mu predictions.

        Returns
        -------
        ndarray, shape (n,)
            Predicted parameter values on response scale.
        """
        self._check_fitted()
        cols = self._formulas.get(parameter, [])
        X_mat = _build_design_matrix(X, cols)
        link = self.family.default_links[parameter]

        eta = X_mat @ self._betas[parameter]
        if parameter == "mu" and exposure is not None:
            exposure = np.asarray(exposure, dtype=float)
            eta = eta + np.log(np.clip(exposure, 1e-10, None))

        return link.inverse(eta)

    def predict_distribution(self, X, exposure: Optional[np.ndarray] = None):
        """
        Return a list of scipy.stats frozen distribution objects, one per row.

        Parameters
        ----------
        X : DataFrame or array
        exposure : array-like, optional

        Returns
        -------
        list of scipy.stats frozen distributions (or None if unsupported)
        """
        self._check_fitted()
        from scipy import stats

        param_arrays = {}
        for pname in self.family.param_names:
            param_arrays[pname] = self.predict(X, parameter=pname, exposure=exposure)

        n = len(param_arrays[self.family.param_names[0]])
        family_name = self.family.__class__.__name__
        dists = []

        for i in range(n):
            p = {k: v[i] for k, v in param_arrays.items()}
            dist = _make_scipy_dist(family_name, p, self.family)
            dists.append(dist)

        return dists

    def predict_mean(self, X, exposure: Optional[np.ndarray] = None) -> np.ndarray:
        """E[Y|X] — convenience wrapper."""
        return self.predict(X, parameter="mu", exposure=exposure)

    def predict_variance(self, X, exposure: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Var[Y|X] from the fitted distribution parameters.

        Uses the distribution's variance formula rather than empirical variance.
        """
        self._check_fitted()
        family_name = self.family.__class__.__name__
        param_arrays = {
            pname: self.predict(X, parameter=pname, exposure=exposure)
            for pname in self.family.param_names
        }
        return _compute_variance(family_name, param_arrays, self.family)

    def volatility_score(self, X, exposure: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Coefficient of variation CV = sqrt(Var[Y|X]) / E[Y|X] per risk.

        Insurance interpretation: risks with CV > 1 have larger standard
        deviation than mean — typical for heavy-tailed severity.
        Useful for pricing loading decisions and risk selection.
        """
        mu = self.predict_mean(X, exposure=exposure)
        var = self.predict_variance(X, exposure=exposure)
        return np.sqrt(np.clip(var, 0, None)) / np.clip(mu, 1e-10, None)

    def relativities(self, parameter: str = "mu"):
        """
        Return coefficient relativities for a given parameter.

        For log-linked parameters (mu, sigma on most families):
          relativity = exp(beta_j) / exp(beta_base)
        where beta_base is the intercept.

        For logit-linked parameters (pi in ZIP):
          Returns odds ratios: exp(beta_j) / 1.

        Returns
        -------
        polars DataFrame with columns: param, term, coefficient, relativity, link
        """
        self._check_fitted()
        if not _HAS_POLARS:
            raise ImportError("polars required for relativities(). Install polars.")

        betas = self._betas[parameter]
        cols = self._formulas.get(parameter, [])
        link = self.family.default_links[parameter]

        terms = ["(Intercept)"] + cols
        intercept = betas[0]

        rows = []
        for term, beta in zip(terms, betas):
            if link.name == "log":
                rel = float(np.exp(beta))
                if term == "(Intercept)":
                    base = float(np.exp(intercept))
                    rows.append({"param": parameter, "term": term, "coefficient": float(beta),
                                 "relativity": base, "link": "log"})
                else:
                    rows.append({"param": parameter, "term": term, "coefficient": float(beta),
                                 "relativity": rel, "link": "log"})
            elif link.name == "logit":
                rows.append({"param": parameter, "term": term, "coefficient": float(beta),
                             "relativity": float(np.exp(beta)), "link": "logit (odds ratio)"})
            else:
                rows.append({"param": parameter, "term": term, "coefficient": float(beta),
                             "relativity": float(beta), "link": link.name})

        return pl.DataFrame(rows)

    def summary(self) -> None:
        """Print coefficient summary for all parameters."""
        self._check_fitted()
        print(f"\nDistributionalGLM — {self.family.__class__.__name__}")
        print(f"  n = {self._n}, loglik = {self._loglik:.4f}")
        print(f"  Converged: {self._converged}")
        print(f"  GAIC(2): {self.gaic(penalty=2):.4f}")
        print()

        for pname in self.family.param_names:
            betas = self._betas[pname]
            cols = self._formulas.get(pname, [])
            terms = ["(Intercept)"] + cols
            link = self.family.default_links[pname]
            print(f"  Parameter: {pname}  (link: {link.name})")
            print(f"  {'Term':<30} {'Coef':>12}")
            print(f"  {'-'*44}")
            for term, beta in zip(terms, betas):
                print(f"  {term:<30} {beta:>12.5f}")
            print()

    def gaic(self, penalty: float = 2.0) -> float:
        """
        Generalised AIC with given penalty.

        penalty=2   -> standard AIC
        penalty=log(n)  -> BIC
        """
        self._check_fitted()
        total_params = sum(len(b) for b in self._betas.values())
        return _gaic(self._loglik, total_params, penalty=penalty)

    def score(
        self,
        X,
        y,
        metric: str = "nll",
        exposure: Optional[np.ndarray] = None,
    ) -> float:
        """
        Evaluate model on held-out data.

        Parameters
        ----------
        X : DataFrame or array
        y : array-like
        metric : str
            'nll' (negative log-likelihood), 'deviance', or 'crps'.
        exposure : array-like, optional

        Returns
        -------
        float
            Metric value. Lower is better for nll and deviance.
        """
        self._check_fitted()
        y = np.asarray(y, dtype=float)

        param_arrays = {
            pname: self.predict(X, parameter=pname, exposure=exposure)
            for pname in self.family.param_names
        }
        ll = self.family.log_likelihood(y, param_arrays)

        if metric == "nll":
            return float(-np.mean(ll[np.isfinite(ll)]))
        elif metric == "deviance":
            # Saturated model: each obs gets its own parameter
            ll_sat = self._saturated_loglik(y, param_arrays)
            return float(2.0 * np.mean(ll_sat - ll))
        elif metric == "crps":
            dists = self.predict_distribution(X, exposure=exposure)
            crps_vals = []
            for dist, yi in zip(dists, y):
                if dist is not None:
                    # Monte Carlo CRPS approximation
                    samples = dist.rvs(1000)
                    crps_vals.append(np.mean(np.abs(samples - yi)) - 0.5 * np.mean(np.abs(
                        samples[:500, None] - samples[None, 500:]
                    )))
            return float(np.mean(crps_vals))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'nll', 'deviance', or 'crps'.")

    def _saturated_loglik(self, y, params):
        """Saturated log-likelihood (each obs perfectly fitted to itself)."""
        sat_params = {k: v.copy() for k, v in params.items()}
        sat_params["mu"] = y.copy()
        try:
            ll = self.family.log_likelihood(y, sat_params)
            return np.where(np.isfinite(ll), ll, 0.0)
        except Exception:
            return np.zeros(len(y))

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

    @property
    def coefficients(self) -> Dict[str, np.ndarray]:
        """Fitted coefficients for all parameters."""
        self._check_fitted()
        return {k: v.copy() for k, v in self._betas.items()}

    @property
    def n_observations(self) -> int:
        return self._n

    @property
    def loglikelihood(self) -> float:
        self._check_fitted()
        return self._loglik

    @property
    def converged(self) -> bool:
        self._check_fitted()
        return self._converged

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"DistributionalGLM({self.family.__class__.__name__}, "
            f"n={self._n}, {status})"
        )


# ---------------------------------------------------------------------------
# Helpers for predict_distribution and predict_variance
# ---------------------------------------------------------------------------

def _make_scipy_dist(family_name: str, p: dict, family):
    """Create a frozen scipy.stats distribution from family parameters."""
    from scipy import stats

    try:
        if family_name == "Gamma":
            mu, sigma = p["mu"], p["sigma"]
            k = 1.0 / sigma**2
            scale = mu / k
            return stats.gamma(a=k, scale=scale)

        elif family_name == "LogNormal":
            mu, sigma = p["mu"], p["sigma"]
            return stats.lognorm(s=sigma, scale=np.exp(mu))

        elif family_name == "InverseGaussian":
            mu, sigma = p["mu"], p["sigma"]
            # scipy InvGauss: mu param = mu/lambda, scale = lambda
            # Var = mu^3 / lambda = sigma^2 * mu^3 => lambda = 1/sigma^2
            lam = 1.0 / sigma**2
            return stats.invgauss(mu=mu / lam, scale=lam)

        elif family_name == "Poisson":
            return stats.poisson(mu=p["mu"])

        elif family_name == "NBI":
            mu, sigma = p["mu"], p["sigma"]
            k = 1.0 / sigma
            p_nb = k / (k + mu)
            return stats.nbinom(n=k, p=p_nb)

        elif family_name == "Tweedie":
            # No standard scipy Tweedie — return None
            return None

        elif family_name == "ZIP":
            # Custom wrapper — return None for now (CRPS/worm plot skips None)
            return None

    except Exception:
        return None


def _compute_variance(family_name: str, param_arrays: dict, family) -> np.ndarray:
    """Compute Var[Y|X] from distribution parameters."""
    if family_name == "Gamma":
        mu = param_arrays["mu"]
        sigma = param_arrays["sigma"]
        return (sigma * mu) ** 2  # Var = (CV * mu)^2

    elif family_name == "LogNormal":
        mu = param_arrays["mu"]
        sigma = param_arrays["sigma"]
        return (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)

    elif family_name == "InverseGaussian":
        mu = param_arrays["mu"]
        sigma = param_arrays["sigma"]
        return sigma**2 * mu**3

    elif family_name == "Poisson":
        return param_arrays["mu"].copy()

    elif family_name == "NBI":
        mu = param_arrays["mu"]
        sigma = param_arrays["sigma"]
        return mu + sigma * mu**2

    elif family_name == "ZIP":
        mu = param_arrays["mu"]
        pi = param_arrays["pi"]
        # E[Y] = (1-pi)*mu, Var[Y] = (1-pi)*mu*(1 + pi*mu)
        return (1.0 - pi) * mu * (1.0 + pi * mu)

    elif family_name == "Tweedie":
        mu = param_arrays["mu"]
        phi = param_arrays["phi"]
        p = getattr(family, "power", 1.5)
        return phi * mu**p

    else:
        return np.full(len(param_arrays[list(param_arrays.keys())[0]]), np.nan)
