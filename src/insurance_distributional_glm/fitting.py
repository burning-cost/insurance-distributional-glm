"""
RS (Rigby-Stasinopoulos) fitting algorithm.

The RS algorithm is the standard fitting engine for GAMLSS. It cycles through
each distribution parameter in turn, updating it via an IRLS (Iteratively
Reweighted Least Squares) step while holding all others fixed.

Each IRLS step constructs:
  z_k = eta_k + (dl/d eta_k) / (E[-d^2 l / d eta_k^2])   (working response)
  w_k = E[-d^2 l / d eta_k^2]                              (IRLS weights)

Then solves the weighted least squares problem:
  beta_k = argmin sum_i w_ki (z_ki - X_ki @ beta_k)^2

which has the closed form: (X^T W X)^{-1} X^T W z.

We use numpy.linalg.lstsq rather than direct solve for numerical stability
with near-collinear design matrices.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .families.base import GamlssFamily


def _wls(
    X: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Weighted least squares: solve (X^T W X) beta = X^T W z.

    Uses lstsq for stability. w is 1-D array of non-negative weights.
    Returns coefficient vector beta of shape (p,).
    """
    sqrt_w = np.sqrt(np.clip(w, 1e-10, None))
    Xw = X * sqrt_w[:, np.newaxis]
    zw = z * sqrt_w
    beta, _, _, _ = np.linalg.lstsq(Xw, zw, rcond=None)
    return beta


def rs_fit(
    family: GamlssFamily,
    design_matrices: Dict[str, np.ndarray],
    y: np.ndarray,
    log_offset: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[float], bool]:
    """
    Fit GAMLSS via the RS algorithm.

    Parameters
    ----------
    family : GamlssFamily
        Distribution family with score/weight methods.
    design_matrices : dict
        Mapping from param name to design matrix (n x p_k). Must include
        an intercept column if wanted — this function does NOT add one.
    y : ndarray, shape (n,)
        Response observations.
    log_offset : ndarray, shape (n,), optional
        log(exposure) added to eta_mu before applying link inverse.
        Only applied to the 'mu' parameter.
    weights : ndarray, shape (n,), optional
        Observation weights (e.g., policy counts). Multiplies the IRLS
        weights w_k element-wise.
    max_iter : int
        Maximum RS outer iterations.
    tol : float
        Convergence threshold on absolute change in total log-likelihood.
    verbose : bool
        Print log-likelihood at each iteration.

    Returns
    -------
    betas : dict
        Fitted coefficients for each parameter.
    params : dict
        Final distribution parameter arrays (on response scale).
    loglik_history : list of float
        Total log-likelihood at each iteration.
    converged : bool
        Whether the algorithm converged within max_iter iterations.
    """
    n = len(y)
    param_names = family.param_names

    if weights is None:
        weights = np.ones(n)

    # Initialise parameters from family's moment estimates
    params = family.starting_values(y)

    # Initialise betas from starting params
    betas: Dict[str, np.ndarray] = {}
    for pname in param_names:
        link = family.default_links[pname]
        eta = link.link(params[pname])
        if pname == "mu" and log_offset is not None:
            eta = eta - log_offset  # subtract offset (it will be added back)
        X = design_matrices[pname]
        # Least squares init
        beta, _, _, _ = np.linalg.lstsq(X, eta, rcond=None)
        betas[pname] = beta

    loglik_history = []
    prev_loglik = -np.inf
    converged = False

    for iteration in range(max_iter):
        for pname in param_names:
            X = design_matrices[pname]
            link = family.default_links[pname]

            # Current linear predictor
            eta_k = X @ betas[pname]
            if pname == "mu" and log_offset is not None:
                eta_k = eta_k + log_offset

            # Score and weight in eta space
            score = family.dl_deta(y, params, pname)
            weight = family.d2l_deta2(y, params, pname)

            # Clip weights for numerical stability
            weight = np.maximum(weight, 1e-10)

            # IRLS working response
            z = eta_k + score / weight

            # Apply observation weights
            irls_w = weight * weights

            # Design matrix for WLS (strip offset for mu)
            if pname == "mu" and log_offset is not None:
                z_adj = z - log_offset  # regress without offset in design
            else:
                z_adj = z

            # Solve WLS
            betas[pname] = _wls(X, z_adj, irls_w)

            # Update linear predictor and parameter
            eta_k = X @ betas[pname]
            if pname == "mu" and log_offset is not None:
                eta_k = eta_k + log_offset

            params[pname] = link.inverse(eta_k)

        # Total log-likelihood
        ll_arr = family.log_likelihood(y, params)
        ll_arr = ll_arr * weights
        total_ll = float(np.sum(ll_arr[np.isfinite(ll_arr)]))
        loglik_history.append(total_ll)

        if verbose:
            print(f"  RS iter {iteration + 1}: loglik = {total_ll:.4f}")

        # Convergence check
        if abs(total_ll - prev_loglik) < tol:
            converged = True
            break

        prev_loglik = total_ll

    return betas, params, loglik_history, converged
