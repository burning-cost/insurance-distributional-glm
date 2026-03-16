"""
Model selection utilities for GAMLSS.

GAIC (Generalised AIC) is the primary selection criterion:
  GAIC(k) = -2 * loglik + k * df

where df is the total number of parameters across all distribution
components and k is the penalty (k=2 gives AIC, k=log(n) gives BIC).

choose_distribution fits all candidate families and ranks by GAIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .families.base import GamlssFamily


@dataclass
class SelectionResult:
    """Result from choose_distribution for one family."""
    family: GamlssFamily
    family_name: str
    gaic: float
    loglik: float
    df: int
    converged: bool
    model: object  # DistributionalGLM, typed as object to avoid circular import


def gaic(loglik: float, n_params: int, penalty: float = 2.0) -> float:
    """
    Generalised AIC.

    Parameters
    ----------
    loglik : float
        Total log-likelihood at convergence.
    n_params : int
        Total number of estimated parameters (sum of df across all
        distribution components).
    penalty : float
        Penalty per parameter. Use 2 for AIC, log(n) for BIC.

    Returns
    -------
    float
        GAIC value. Lower is better.
    """
    return -2.0 * loglik + penalty * n_params


def choose_distribution(
    X,
    y: np.ndarray,
    families: List[GamlssFamily],
    formulas: Optional[Dict] = None,
    exposure: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    penalty: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> List[SelectionResult]:
    """
    Fit multiple families and rank by GAIC.

    Parameters
    ----------
    X : DataFrame (polars or pandas)
        Covariate matrix.
    y : ndarray
        Response observations.
    families : list of GamlssFamily
        Candidate distribution families to compare.
    formulas : dict, optional
        Passed to DistributionalGLM. If None, all columns for mu,
        intercept-only for other parameters.
    exposure : ndarray, optional
        Exposure offset (values, not log). Applied to mu.
    weights : ndarray, optional
        Observation weights.
    penalty : float
        GAIC penalty. Default 2 (AIC).
    max_iter : int
        Maximum RS iterations per family.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.

    Returns
    -------
    list of SelectionResult
        Sorted ascending by GAIC (best first).
    """
    # Import here to avoid circular dependency
    from .model import DistributionalGLM

    results = []
    for fam in families:
        fname = fam.__class__.__name__
        if verbose:
            print(f"Fitting {fname}...")
        try:
            mdl = DistributionalGLM(family=fam, formulas=formulas)
            mdl.fit(
                X, y,
                exposure=exposure,
                weights=weights,
                max_iter=max_iter,
                tol=tol,
            )
            ll = mdl._loglik
            df = sum(len(b) for b in mdl._betas.values())
            g = gaic(ll, df, penalty=penalty)
            results.append(SelectionResult(
                family=fam,
                family_name=fname,
                gaic=g,
                loglik=ll,
                df=df,
                converged=mdl._converged,
                model=mdl,
            ))
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as e:
            if verbose:
                print(f"  {fname} failed: {e}")

    results.sort(key=lambda r: r.gaic)
    return results
