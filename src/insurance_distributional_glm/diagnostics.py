"""
Diagnostic tools for fitted GAMLSS models.

The key diagnostic is normalised quantile residuals (Dunn & Smyth 1996).
For a correct model, these should be iid N(0,1). Worm plots (van Buuren &
Fredriks 2001) are detrended QQ plots that make deviations from normality
visible even in the tails.

Reference:
  Dunn, P.K. and Smyth, G.K. (1996). Randomized quantile residuals.
  Journal of Computational and Graphical Statistics, 5(3), 236-244.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import stats


def quantile_residuals(
    model,
    X,
    y: np.ndarray,
    exposure: Optional[np.ndarray] = None,
    n_random: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Compute randomised quantile residuals (Dunn & Smyth 1996).

    For a continuous response, the residual is:
      r_i = Phi^{-1}(F(y_i; theta_hat_i))

    For discrete responses, F(y_i) is not uniquely defined at y_i, so we
    randomise within the probability interval:
      u_i ~ Uniform(F(y_i - 1; theta_hat_i), F(y_i; theta_hat_i))
      r_i = Phi^{-1}(u_i)

    A correctly specified model produces r_i ~ iid N(0,1).

    Parameters
    ----------
    model : DistributionalGLM
        A fitted model.
    X : DataFrame
        Covariates (same format as fit).
    y : ndarray
        Observed responses.
    exposure : ndarray, optional
        Exposure for offset.
    n_random : int
        For discrete families, number of randomisations to average over.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (n,)
        Normalised quantile residuals.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    dists = model.predict_distribution(X, exposure=exposure)

    family = model.family
    is_discrete = family.__class__.__name__ in ("Poisson", "NBI", "ZIP")

    residuals = np.zeros(len(y))

    for i, (dist, yi) in enumerate(zip(dists, y)):
        if dist is None:
            residuals[i] = np.nan
            continue

        if is_discrete:
            # Randomise within [F(y-1), F(y)]
            f_lower = dist.cdf(yi - 1) if yi > 0 else 0.0
            f_upper = dist.cdf(yi)
            f_lower = np.clip(f_lower, 1e-10, 1 - 1e-10)
            f_upper = np.clip(f_upper, f_lower + 1e-10, 1 - 1e-10)
            u_vals = rng.uniform(f_lower, f_upper, size=n_random)
            residuals[i] = np.mean(stats.norm.ppf(u_vals))
        else:
            p = dist.cdf(yi)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            residuals[i] = stats.norm.ppf(p)

    return residuals


def worm_plot(
    model,
    X,
    y: np.ndarray,
    exposure: Optional[np.ndarray] = None,
    n_groups: int = 4,
    seed: int = 42,
    ax=None,
):
    """
    Worm plot: detrended QQ plot of quantile residuals.

    Shows deviation from N(0,1) per quantile group of the fitted values.
    A flat worm near zero indicates good fit. Systematic deviations indicate:
      - Upward shift: underdispersion
      - Downward shift: overdispersion
      - S-shape: wrong variance-mean relationship
      - U/inverted-U: wrong kurtosis

    Requires matplotlib (optional dependency).

    Parameters
    ----------
    model : DistributionalGLM
        Fitted model.
    X, y : as for quantile_residuals.
    exposure : ndarray, optional
    n_groups : int
        Number of groups to split fitted mu into (default 4).
    seed : int
        Random seed for quantile residuals.
    ax : matplotlib.axes.Axes or None
        Axes to draw on. Creates new figure if None.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for worm_plot. "
            "Install it with: pip install insurance-distributional-glm[plots]"
        )

    resids = quantile_residuals(model, X, y, exposure=exposure, seed=seed)
    mu_hat = model.predict(X, parameter="mu", exposure=exposure)

    if ax is None:
        fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), sharey=True)
    else:
        axes = [ax] * n_groups

    group_bounds = np.quantile(mu_hat, np.linspace(0, 1, n_groups + 1))

    for g in range(n_groups):
        mask = (mu_hat >= group_bounds[g]) & (mu_hat <= group_bounds[g + 1])
        r_group = resids[mask]
        r_group = r_group[np.isfinite(r_group)]

        if len(r_group) < 2:
            continue

        ax_g = axes[g] if n_groups > 1 else ax

        # Normal quantiles
        n = len(r_group)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical = stats.norm.ppf(probs)

        # Sort residuals
        observed = np.sort(r_group)

        # Detrend: residual minus theoretical (worm = deviation from diagonal)
        worm = observed - theoretical

        ax_g.scatter(theoretical, worm, s=10, alpha=0.5, color="steelblue")
        ax_g.axhline(0, color="black", linewidth=0.8)
        ax_g.set_title(
            f"mu: [{group_bounds[g]:.2f}, {group_bounds[g+1]:.2f}]",
            fontsize=9,
        )
        ax_g.set_xlabel("Theoretical N(0,1) quantiles")
        if g == 0:
            ax_g.set_ylabel("Residual - Theoretical")

    if ax is None:
        plt.suptitle("Worm Plot (detrended QQ by fitted mu group)", y=1.02)
        plt.tight_layout()
        return axes
    return ax


def fitted_vs_observed(
    model,
    X,
    y: np.ndarray,
    exposure: Optional[np.ndarray] = None,
    ax=None,
):
    """
    Scatter plot of fitted mu vs observed y.

    Parameters
    ----------
    model : DistributionalGLM
        Fitted model.
    X, y, exposure : as above.
    ax : matplotlib Axes, optional.

    Returns
    -------
    ax
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for plotting.")

    mu_hat = model.predict(X, parameter="mu", exposure=exposure)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(mu_hat, y, s=8, alpha=0.3, color="steelblue")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=1)
    ax.set_xlabel("Fitted mu")
    ax.set_ylabel("Observed y")
    ax.set_title("Fitted vs Observed")

    return ax
