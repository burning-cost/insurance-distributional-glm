"""
insurance-distributional-glm
=============================

GAMLSS (Generalised Additive Models for Location, Scale and Shape) for
insurance pricing in Python.

Models the full conditional distribution p(Y|X), not just E[Y|X].
Each distribution parameter (mean, variance, shape) can depend on covariates.

Quick start
-----------
>>> import numpy as np
>>> import polars as pl
>>> from insurance_distributional_glm import DistributionalGLM
>>> from insurance_distributional_glm.families import Gamma
>>>
>>> rng = np.random.default_rng(42)
>>> df = pl.DataFrame({"age_band": rng.integers(0, 3, 500).astype(float)})
>>> y = rng.gamma(shape=2.0, scale=500.0, size=500)
>>>
>>> model = DistributionalGLM(
...     family=Gamma(),
...     formulas={"mu": ["age_band"], "sigma": []},
... )
>>> model.fit(df, y)
>>> model.predict(df, parameter="mu")[:5]
"""

from .model import DistributionalGLM
from .selection import choose_distribution, gaic, SelectionResult
from .diagnostics import quantile_residuals, worm_plot
from . import families

__version__ = "0.1.2"

__all__ = [
    "DistributionalGLM",
    "choose_distribution",
    "gaic",
    "SelectionResult",
    "quantile_residuals",
    "worm_plot",
    "families",
]
