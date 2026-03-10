# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-distributional-glm: Full GAMLSS Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates GAMLSS distributional regression for insurance pricing.
# MAGIC We model claim severity with both mean AND variance as functions of risk factors —
# MAGIC something a standard GLM cannot do.
# MAGIC
# MAGIC **Use case**: Motor insurance severity where young drivers not only have higher
# MAGIC average claims but also higher variance (more volatile loss experience).

# COMMAND ----------

# MAGIC %pip install insurance-distributional-glm

# COMMAND ----------

import numpy as np
import polars as pl
from scipy import stats

from insurance_distributional_glm import (
    DistributionalGLM,
    choose_distribution,
    quantile_residuals,
    gaic,
    SelectionResult,
)
from insurance_distributional_glm.families import (
    Gamma, LogNormal, InverseGaussian, Poisson, NBI, ZIP
)

print("insurance-distributional-glm loaded successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate synthetic motor insurance data
# MAGIC
# MAGIC We simulate a portfolio where:
# MAGIC - `age_band`: 0=young (17-25), 1=mid (26-45), 2=mature (46-65)
# MAGIC - `vehicle_value`: £5k - £40k
# MAGIC - Severity (mu) increases with vehicle_value and is higher for young drivers
# MAGIC - **Volatility (sigma) is HIGHER for young drivers** — this is what the standard GLM misses

# COMMAND ----------

rng = np.random.default_rng(2026)
n = 2000

# Risk factors
age_band = rng.integers(0, 3, n).astype(float)  # 0=young, 1=mid, 2=mature
vehicle_value = rng.uniform(5000, 40000, n)
ncb_years = rng.integers(0, 10, n).astype(float)  # No Claims Bonus years
policy_years = rng.uniform(0.1, 1.0, n)  # exposure

# True log-linear model for mu (severity)
log_mu_true = (
    7.0
    + 0.4 * (age_band == 0).astype(float)    # young: +40% severity
    - 0.1 * (age_band == 2).astype(float)    # mature: -10% severity
    + 0.00003 * vehicle_value                 # higher value = higher severity
    - 0.03 * ncb_years                        # NCB reduces severity
)
mu_true = np.exp(log_mu_true)

# True sigma model: young drivers are MORE volatile
log_sigma_true = (
    -0.7
    + 0.5 * (age_band == 0).astype(float)    # young: CV 50% higher
    - 0.2 * (age_band == 2).astype(float)    # mature: CV 20% lower
)
sigma_true = np.exp(log_sigma_true)

# Simulate Gamma severity claims
shape_true = 1.0 / sigma_true**2
y_severity = rng.gamma(shape=shape_true, scale=mu_true / shape_true)

# Polars DataFrame
df = pl.DataFrame({
    "age_band": age_band,
    "vehicle_value": vehicle_value,
    "ncb_years": ncb_years,
})

print(f"Portfolio: n={n} claims")
print(f"Mean severity: £{y_severity.mean():.0f}")
print(f"True mu range: £{mu_true.min():.0f} - £{mu_true.max():.0f}")
print(f"True sigma range: {sigma_true.min():.3f} - {sigma_true.max():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit a standard GLM (mean only)
# MAGIC
# MAGIC The standard Gamma GLM models only E[Y|X]. It cannot capture the
# MAGIC heterogeneous variance structure.

# COMMAND ----------

model_glm = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "vehicle_value", "ncb_years"],
        "sigma": [],  # intercept-only: constant CV across all risks
    },
)
model_glm.fit(df, y_severity, verbose=False)
model_glm.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit the full distributional model (GAMLSS)
# MAGIC
# MAGIC Now we let sigma also depend on age_band. This is the key GAMLSS advantage.

# COMMAND ----------

model_gamlss = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "vehicle_value", "ncb_years"],
        "sigma": ["age_band"],  # variance varies by age
    },
)
model_gamlss.fit(df, y_severity, verbose=True)
model_gamlss.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Compare models by GAIC

# COMMAND ----------

print("Model comparison (lower GAIC = better):")
print(f"  Standard GLM (sigma constant): GAIC = {model_glm.gaic():.2f}")
print(f"  GAMLSS (sigma ~ age_band):     GAIC = {model_gamlss.gaic():.2f}")
print(f"  Improvement: {model_glm.gaic() - model_gamlss.gaic():.2f} GAIC units")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Relativities

# COMMAND ----------

print("=== Mu relativities (log link) ===")
print(model_gamlss.relativities("mu"))

print("\n=== Sigma relativities (log link) ===")
print(model_gamlss.relativities("sigma"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Volatility scoring
# MAGIC
# MAGIC The key insurance application: identify high-volatility risks.
# MAGIC A risk with CV > 1 has more uncertainty than its expected value.

# COMMAND ----------

# Score each risk
cv = model_gamlss.volatility_score(df)
mu_hat = model_gamlss.predict_mean(df)
sigma_hat = model_gamlss.predict(df, parameter="sigma")

results_df = pl.DataFrame({
    "age_band": age_band,
    "vehicle_value": vehicle_value,
    "ncb_years": ncb_years,
    "mu_hat": mu_hat,
    "sigma_hat": sigma_hat,
    "cv": cv,
    "actual": y_severity,
})

# High volatility risks
high_vol = results_df.filter(pl.col("cv") > sigma_hat.mean() * 1.5)
print(f"High volatility risks: {len(high_vol)} ({100*len(high_vol)/n:.1f}% of portfolio)")
print(high_vol.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Frequency modelling with NBI
# MAGIC
# MAGIC For claim frequency with overdispersion (standard in insurance).

# COMMAND ----------

# Simulate claim counts
lambda_true = np.exp(
    -2.5
    + 0.3 * (age_band == 0).astype(float)
    - 0.1 * (age_band == 2).astype(float)
) * policy_years

sigma_nbi = 0.4
k_true = 1.0 / sigma_nbi
p_true = k_true / (k_true + lambda_true)
y_freq = stats.nbinom(n=k_true, p=p_true).rvs(random_state=rng.integers(10000)).astype(float)

print(f"Claim counts: mean={y_freq.mean():.3f}, var={y_freq.var():.3f}")
print(f"Overdispersion ratio: {y_freq.var()/y_freq.mean():.2f} (>1 = overdispersed)")

# Fit NBI with exposure offset
model_nbi = DistributionalGLM(
    family=NBI(),
    formulas={"mu": ["age_band"], "sigma": []},
)
model_nbi.fit(df, y_freq, exposure=policy_years)
model_nbi.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Family selection

# COMMAND ----------

families = [Gamma(), LogNormal(), InverseGaussian()]
results = choose_distribution(
    df, y_severity,
    families=families,
    formulas={"mu": ["age_band", "vehicle_value"], "sigma": []},
    penalty=2.0,
    verbose=True,
)

print("\nFamily ranking by GAIC (AIC):")
for r in results:
    status = "converged" if r.converged else "DID NOT CONVERGE"
    print(f"  {r.family_name:<20} GAIC={r.gaic:.2f}  loglik={r.loglik:.2f}  df={r.df}  [{status}]")

print(f"\nBest family: {results[0].family_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Quantile residuals

# COMMAND ----------

resids = quantile_residuals(model_gamlss, df, y_severity, seed=42)
print(f"Quantile residuals: mean={resids.mean():.3f}, std={resids.std():.3f}")
print(f"Expected for correct model: mean≈0, std≈1")

from scipy import stats as scipy_stats
stat, pval = scipy_stats.kstest(resids[np.isfinite(resids)], "norm")
print(f"KS test for normality: stat={stat:.4f}, p={pval:.4f}")
print(f"Conclusion: {'Good fit (p>0.05)' if pval > 0.05 else 'Potential misfit (p<0.05)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Exposure-adjusted pure premium comparison
# MAGIC
# MAGIC Demonstrating the Tweedie family for direct pure premium modelling.

# COMMAND ----------

from insurance_distributional_glm.families import Tweedie

# Pure premium = severity * frequency (simplified)
pure_premium = y_severity * y_freq / (policy_years * 5)  # normalised

# Filter positive pure premiums (Tweedie handles zeros internally)
mask = pure_premium > 0
y_pp = pure_premium[mask]
df_pp = df.filter(pl.Series(mask))

print(f"Pure premiums: n={mask.sum()}, mean=£{y_pp.mean():.2f}")

model_tweedie = DistributionalGLM(
    family=Tweedie(power=1.5),
    formulas={"mu": ["age_band", "vehicle_value"], "phi": []},
)
model_tweedie.fit(df_pp, y_pp)
model_tweedie.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This demo showed:
# MAGIC
# MAGIC 1. **GAMLSS vs standard GLM**: The GAMLSS model with sigma ~ age_band achieves
# MAGIC    meaningfully better GAIC by capturing heterogeneous variance.
# MAGIC
# MAGIC 2. **Volatility scoring**: Every risk gets a CV score. Young drivers have higher CV,
# MAGIC    justifying higher pricing loads or risk selection decisions.
# MAGIC
# MAGIC 3. **NBI for frequency**: Overdispersed claim counts are handled naturally.
# MAGIC    The sigma parameter captures unobserved heterogeneity.
# MAGIC
# MAGIC 4. **Family selection**: `choose_distribution` fits all candidate families and
# MAGIC    ranks by GAIC — same workflow as R's `gamlss` package.
# MAGIC
# MAGIC 5. **Quantile residuals**: Model validation via normalised quantile residuals
# MAGIC    that should be N(0,1) for a correct model.
# MAGIC
# MAGIC Install: `pip install insurance-distributional-glm`
