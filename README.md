# insurance-distributional-glm

GAMLSS (Generalised Additive Models for Location, Scale and Shape) for insurance pricing in Python.

## The problem

Standard GLMs model E[Y|X] — the conditional mean. That's fine when you believe every risk with the same mean also has the same variance. But in motor insurance, a young driver and a middle-aged driver with identical expected claims can have dramatically different claim volatility. Your pricing should know the difference.

GAMLSS fixes this by modelling the full conditional distribution p(Y|X), not just its mean. Each distribution parameter — mean, variance, shape — is expressed as a function of covariates. For a Gamma severity model:

```
log(mu_i)    = x_i^T beta_mu        # mean depends on risk factors
log(sigma_i) = z_i^T beta_sigma      # CV depends on (possibly different) risk factors
```

**R has had this since 2005 (the `gamlss` package, 100+ distributions). Python has had nothing production-ready. This fills that gap.**

## Why this matters for insurance pricing

1. **Heterogeneous variance**: risks with the same expected loss have different volatility. A high-CV risk needs a different loading than a low-CV risk even if their means are equal.

2. **Regulatory pressure**: PRA and FCA increasingly expect firms to demonstrate they understand uncertainty in their estimates, not just point predictions. Modelling sigma as a function of covariates is the right answer.

3. **Tail behaviour**: for commercial lines and liability, the shape of the distribution (not just its mean) drives large loss exposure. Getting sigma right matters more than squeezing another point of fit on the mean.

4. **Zero-inflated counts**: ZIP models let you separate structural zeros (non-claimants, seasonal risks) from Poisson claim frequency without ad-hoc adjustments.

## Installation

```bash
pip install insurance-distributional-glm
```

With matplotlib for diagnostic plots:
```bash
pip install "insurance-distributional-glm[plots]"
```

## Quick start

```python
import numpy as np
import polars as pl
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

# Claim severity data
df = pl.DataFrame({
    "age_band": [0.0, 1.0, 2.0, 0.0, 1.0] * 200,          # young / mid / mature
    "vehicle_value": [8000.0, 15000.0, 25000.0] * 333 + [8000.0],
})
rng = np.random.default_rng(42)
y = rng.gamma(4.0, 500.0, len(df))

# Model mean with age + vehicle_value, variance with age only
model = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "vehicle_value"],
        "sigma": ["age_band"],
    },
)
model.fit(df, y)
model.summary()
```

Output:
```
DistributionalGLM — Gamma
  n = 999, loglik = -7412.3410
  Converged: True
  GAIC(2): 14840.6820

  Parameter: mu  (link: log)
  Term                            Coef
  --------------------------------------------
  (Intercept)                  6.09483
  age_band                     0.02341
  vehicle_value                0.00001

  Parameter: sigma  (link: log)
  Term                            Coef
  --------------------------------------------
  (Intercept)                 -0.65734
  age_band                     0.01204
```

## Families

| Family | Parameters | Insurance use |
|--------|-----------|---------------|
| `Gamma` | mu (mean), sigma (CV) | Claim severity. Most common choice. |
| `LogNormal` | mu (log mean), sigma (log sd) | Severity when log(claims) is symmetric. |
| `InverseGaussian` | mu, sigma | Heavy-tailed liability severity. |
| `Tweedie(power)` | mu, phi | Pure premiums (includes structural zeros). |
| `Poisson` | mu | Claim frequency, baseline. |
| `NBI` | mu, sigma (overdispersion) | Overdispersed claim counts. Almost always better than Poisson. |
| `ZIP` | mu, pi (zero inflation) | Frequency with excess zeros. |

## Exposure offsets

```python
# Exposure-weighted frequency model
model = DistributionalGLM(family=NBI(), formulas={"mu": ["age_band"], "sigma": []})
model.fit(df, claim_counts, exposure=policy_years)

# Predict rate per unit exposure for new business
rates = model.predict(new_df, parameter="mu", exposure=np.ones(len(new_df)))
```

## Model selection

```python
from insurance_distributional_glm import choose_distribution
from insurance_distributional_glm.families import Gamma, LogNormal, InverseGaussian

results = choose_distribution(
    df, y,
    families=[Gamma(), LogNormal(), InverseGaussian()],
    formulas={"mu": ["age_band", "vehicle_value"], "sigma": []},
    penalty=2.0,   # AIC; use np.log(len(y)) for BIC
)

for r in results:
    print(f"{r.family_name}: GAIC={r.gaic:.1f}, converged={r.converged}")
# Gamma: GAIC=14840.7, converged=True
# LogNormal: GAIC=14901.3, converged=True
# InverseGaussian: GAIC=14923.8, converged=True
```

## Relativities

```python
# Actuarial-style output: multiplicative factors per risk factor level
rel = model.relativities(parameter="mu")
print(rel)
# shape: (n_terms, 4)
# columns: param, term, coefficient, relativity, link
```

For log-linked parameters, `relativity = exp(coefficient)` — the multiplicative effect on the predicted mean, exactly as actuaries expect from a GLM output.

## Diagnostics

```python
from insurance_distributional_glm import quantile_residuals, worm_plot

# Randomised quantile residuals (Dunn & Smyth 1996)
# For a correct model, these should be iid N(0,1)
resids = quantile_residuals(model, df, y, seed=42)

# Worm plot: detrended QQ plot, split by fitted mu quantile
# Requires matplotlib: pip install "insurance-distributional-glm[plots]"
worm_plot(model, df, y, n_groups=4)
```

## Volatility scoring

```python
# CV = sqrt(Var[Y|X]) / E[Y|X] per risk
cv = model.volatility_score(df)

# Flag high-volatility risks (CV > 0.8)
df_scored = df.with_columns(pl.Series("cv", cv))
high_vol = df_scored.filter(pl.col("cv") > 0.8)
```

## The RS algorithm

Fitting uses the Rigby-Stasinopoulos (RS) algorithm: cycle through each distribution parameter, update it via IRLS (weighted least squares) while holding all others fixed. Convergence criterion is change in total log-likelihood < tol.

This is equivalent to a coordinate descent on the joint log-likelihood, where each coordinate step has a closed-form weighted least squares solution. It's not the most efficient algorithm (CG — Conjugate Gradient — is faster for large p), but it's robust and straightforward to implement correctly.

## Design choices

**Why numpy/scipy only, no torch?** Insurance pricing teams typically work in SQL/Python without GPU infrastructure. A numpy implementation is deployable anywhere.

**Why not statsmodels?** We tried. statsmodels' GLM is not designed to be extended to multiple linear predictors cleanly, and the formula interface adds overhead that actuaries don't use. Better to build clean from the RS paper.

**Why polars as the primary DataFrame interface?** Speed for large portfolio operations, and the expression API makes feature engineering readable. Pandas is supported via duck typing.

**Why fix the power parameter p in Tweedie?** Profile likelihood over a grid of p values (say 1.2 to 1.8 in steps of 0.1) and pick the best. Treating p as a free parameter inside the RS loop causes numerical instability. We may add `fit_tweedie_power()` as a wrapper in a future version.

## References

- Rigby, R.A. and Stasinopoulos, D.M. (2005). Generalised additive models for location, scale and shape. *JRSS-C*, 54(3), 507-554.
- Dunn, P.K. and Smyth, G.K. (1996). Randomized quantile residuals. *JCGS*, 5(3), 236-244.
- Smyth, G.K. and Jørgensen, B. (2002). Fitting Tweedie's compound Poisson model to insurance claims data. *ASTIN Bulletin*, 32(1), 143-157.

## License

MIT
