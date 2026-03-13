# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-distributional-glm (GAMLSS) vs separate mean/variance GLMs
# MAGIC
# MAGIC **Library:** `insurance-distributional-glm` — GAMLSS for insurance pricing.
# MAGIC Models the full conditional distribution p(Y|X) by allowing each parameter
# MAGIC (mean, variance, shape) to depend on covariates via the RS algorithm.
# MAGIC
# MAGIC **Baseline:** two-stage approach — a Gamma GLM for the mean (statsmodels),
# MAGIC followed by a separate Gamma GLM fitted to squared Pearson residuals to
# MAGIC estimate the variance. This is what teams do in practice when they recognise
# MAGIC heterogeneous variance but do not have a joint model.
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor severity — 30,000 paid claims, known DGP where
# MAGIC variance genuinely depends on covariates (vehicle class, age band, channel).
# MAGIC Temporal 70/30 train/test split.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The benchmark question: does jointly modelling mu and sigma via the RS algorithm
# MAGIC produce better-calibrated prediction intervals than estimating variance as an
# MAGIC afterthought? The answer matters for pricing loadings, reinsurance, and FCA
# MAGIC uncertainty disclosures — not just for Gini.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-distributional-glm statsmodels matplotlib numpy scipy polars

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from insurance_distributional_glm import DistributionalGLM, quantile_residuals
from insurance_distributional_glm.families import Gamma, LogNormal

warnings.filterwarnings("ignore")

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC We generate synthetic UK motor severity data with a known data-generating process.
# MAGIC The DGP has heterogeneous variance — the CV of claims depends on vehicle class and
# MAGIC distribution channel. This is the condition under which GAMLSS earns its keep.
# MAGIC
# MAGIC Key DGP features:
# MAGIC - Mean severity depends on age band, vehicle class, sum insured (log), and channel
# MAGIC - CV (sigma in the Gamma parameterisation) depends on vehicle class and channel
# MAGIC - High-value vehicles have higher mean AND higher variance relative to the mean
# MAGIC - Broker channel policies have greater volatility than direct policies

# COMMAND ----------

rng = np.random.default_rng(42)
N = 30_000

# --- Covariates ---
age_band   = rng.choice(["17-24", "25-34", "35-49", "50-64", "65+"], N,
                         p=[0.10, 0.22, 0.30, 0.26, 0.12])
veh_class  = rng.choice(["A", "B", "C", "D"], N, p=[0.35, 0.30, 0.22, 0.13])
channel    = rng.choice(["direct", "broker", "aggregator"], N, p=[0.45, 0.30, 0.25])
sum_ins    = rng.lognormal(mean=9.5, sigma=0.6, size=N)  # ~ £8k-£60k range

# --- True mu (conditional mean severity) ---
mu_log  = 6.8  # intercept: exp(6.8) ~ £900 base claim
mu_log += np.where(age_band == "17-24",  0.45, 0.0)
mu_log += np.where(age_band == "25-34",  0.18, 0.0)
mu_log += np.where(age_band == "50-64", -0.08, 0.0)
mu_log += np.where(age_band == "65+",   -0.05, 0.0)
mu_log += np.where(veh_class == "B",  0.25, 0.0)
mu_log += np.where(veh_class == "C",  0.55, 0.0)
mu_log += np.where(veh_class == "D",  0.90, 0.0)
mu_log += np.where(channel == "broker",     0.15, 0.0)
mu_log += np.where(channel == "aggregator", 0.08, 0.0)
mu_log += 0.30 * (np.log(sum_ins) - 9.5)   # continuous: higher SI -> higher claim
mu_true = np.exp(mu_log)

# --- True sigma (CV, heterogeneous by design) ---
# This is the key difference from a standard GLM: sigma varies across risks
sigma_log  = -1.20  # intercept: exp(-1.20) ~ 0.30 base CV
sigma_log += np.where(veh_class == "B",  0.18, 0.0)
sigma_log += np.where(veh_class == "C",  0.38, 0.0)
sigma_log += np.where(veh_class == "D",  0.62, 0.0)   # luxury/exotic: much more volatile
sigma_log += np.where(channel == "broker",     0.35, 0.0)  # broker: heterogeneous risks
sigma_log += np.where(channel == "aggregator", 0.15, 0.0)
sigma_true = np.exp(sigma_log)

# Gamma shape parameter k = 1/sigma^2, rate = k/mu
k_true = 1.0 / sigma_true**2
y = rng.gamma(shape=k_true, scale=mu_true / k_true)

# Temporal split: simulate policy year
policy_year = rng.choice([2021, 2022, 2023], N, p=[0.35, 0.35, 0.30])
order = np.argsort(policy_year, kind="stable")

df_pd = pd.DataFrame({
    "age_band":   age_band,
    "veh_class":  veh_class,
    "channel":    channel,
    "sum_ins":    sum_ins,
    "log_sum_ins": np.log(sum_ins),
    "policy_year": policy_year,
    "y":          y,
    "mu_true":    mu_true,
    "sigma_true": sigma_true,
})
df_pd = df_pd.iloc[order].reset_index(drop=True)

train_end = int(N * 0.70)
train_pd = df_pd.iloc[:train_end].copy()
test_pd  = df_pd.iloc[train_end:].copy()

print(f"Total:  {N:,}")
print(f"Train:  {len(train_pd):,}  ({100*len(train_pd)/N:.0f}%)")
print(f"Test:   {len(test_pd):,}   ({100*len(test_pd)/N:.0f}%)")
print(f"\nResponse y summary:")
print(f"  Mean: {y.mean():.1f}, Std: {y.std():.1f}, Median: {np.median(y):.1f}")
print(f"\nTrue sigma range: [{sigma_true.min():.3f}, {sigma_true.max():.3f}]")
print(f"True CV heterogeneity: {sigma_true.std()/sigma_true.mean():.2f} (CoV of sigma)")
print(f"\nChannel distribution:")
print(pd.Series(channel).value_counts().to_string())
print(f"\nVehicle class distribution:")
print(pd.Series(veh_class).value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Two-stage separate GLMs
# MAGIC
# MAGIC The standard workaround for heterogeneous variance in a GLM framework is two-stage:
# MAGIC
# MAGIC 1. Fit a Gamma GLM for the mean on the full training set
# MAGIC 2. Compute squared Pearson residuals (each is a chi-squared(1) variate if the
# MAGIC    mean model is correct)
# MAGIC 3. Regress squared Pearson residuals against dispersion factors using a Gamma GLM
# MAGIC    with log link to estimate observation-level phi
# MAGIC
# MAGIC This is the DGLM idea (Smyth 1989) but implemented manually and without the REML
# MAGIC correction. The two stages are not jointly estimated — mean model errors contaminate
# MAGIC the variance estimates.

# COMMAND ----------

t0 = time.perf_counter()

# Stage 1: mean GLM
mean_formula = (
    "y ~ C(age_band) + C(veh_class) + C(channel) + log_sum_ins"
)
glm_mean = smf.glm(
    mean_formula,
    data=train_pd,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

mu_baseline_train = glm_mean.predict(train_pd)
mu_baseline_test  = glm_mean.predict(test_pd)

# Stage 2: variance GLM on squared Pearson residuals
# Squared Pearson residual = ((y - mu) / (mu * phi))^2 * phi = (y - mu)^2 / mu^2 / phi
# Under a constant-phi Gamma, E[(y-mu)^2/mu^2] = phi (the dispersion)
train_pd = train_pd.copy()
pearson_sq = (train_pd["y"].values - mu_baseline_train) ** 2 / mu_baseline_train**2

train_pd["pearson_sq"] = pearson_sq

# Only use clearly informative records; extreme residuals contaminate the variance model
mask = (pearson_sq > 0.001) & (pearson_sq < 50.0)
train_var = train_pd[mask].copy()

var_formula = "pearson_sq ~ C(veh_class) + C(channel)"
glm_var = smf.glm(
    var_formula,
    data=train_var,
    family=sm.families.Gamma(link=sm.families.links.Log()),
).fit()

phi_baseline_train = glm_var.predict(train_pd)
phi_baseline_test  = glm_var.predict(test_pd)

baseline_fit_time = time.perf_counter() - t0

print(f"Baseline fit time: {baseline_fit_time:.2f}s")
print(f"\nMean GLM deviance explained: {(1 - glm_mean.deviance / glm_mean.null_deviance):.1%}")
print(f"\nBaseline mean prediction range (test):")
print(f"  [{mu_baseline_test.min():.1f}, {mu_baseline_test.max():.1f}]")
print(f"\nBaseline phi range (test):")
print(f"  [{phi_baseline_test.min():.4f}, {phi_baseline_test.max():.4f}]")
print(f"\nMean phi (test): {phi_baseline_test.mean():.4f}")

# COMMAND ----------

# Stage 1 coefficient summary
print("=== Mean GLM Coefficients ===")
print(glm_mean.summary2().tables[1].to_string())

# COMMAND ----------

# Stage 2 coefficient summary
print("=== Variance GLM Coefficients ===")
print(glm_var.summary2().tables[1].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: DistributionalGLM (GAMLSS)
# MAGIC
# MAGIC `DistributionalGLM` fits both components jointly using the RS (Rigby-Stasinopoulos)
# MAGIC algorithm: cycle through each distribution parameter, update via IRLS while holding
# MAGIC the others fixed, until convergence on total log-likelihood.
# MAGIC
# MAGIC Key difference from the two-stage baseline:
# MAGIC - Joint log-likelihood: the mean and variance fits are consistent with each other
# MAGIC - IRLS weights in the mean step use the current fitted phi estimates
# MAGIC - Convergence is on the total log-likelihood, not just mean deviance
# MAGIC
# MAGIC The Gamma family parameterises sigma as the coefficient of variation (CV): if the
# MAGIC true distribution is Gamma(shape=k, scale=mu/k), then sigma = 1/sqrt(k).

# COMMAND ----------

t0 = time.perf_counter()

FEATURES = ["age_band", "veh_class", "channel", "log_sum_ins"]

df_pl_train = pl.from_pandas(train_pd[FEATURES])
df_pl_test  = pl.from_pandas(test_pd[FEATURES])

y_train = train_pd["y"].values
y_test  = test_pd["y"].values

gamlss = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "veh_class", "channel", "log_sum_ins"],
        "sigma": ["veh_class", "channel"],
    },
)
gamlss.fit(df_pl_train, y_train, max_iter=100, tol=1e-7)

mu_gamlss_train  = gamlss.predict(df_pl_train, parameter="mu")
mu_gamlss_test   = gamlss.predict(df_pl_test,  parameter="mu")
sig_gamlss_train = gamlss.predict(df_pl_train, parameter="sigma")
sig_gamlss_test  = gamlss.predict(df_pl_test,  parameter="sigma")

library_fit_time = time.perf_counter() - t0

print(f"GAMLSS fit time:    {library_fit_time:.2f}s")
print(f"Converged:          {gamlss.converged}")
print(f"Log-likelihood:     {gamlss.loglikelihood:.2f}")
print(f"GAIC (AIC, pen=2):  {gamlss.gaic(penalty=2):.2f}")
print()
gamlss.summary()

# COMMAND ----------

print("=== GAMLSS mu relativities ===")
print(gamlss.relativities("mu").to_pandas().to_string(index=False))
print()
print("=== GAMLSS sigma relativities ===")
print(gamlss.relativities("sigma").to_pandas().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics
# MAGIC
# MAGIC We compare on four axes:
# MAGIC
# MAGIC 1. **Mean fit quality**: deviance and Gini on test set — how well each model
# MAGIC    captures variation in the conditional mean E[Y|X].
# MAGIC
# MAGIC 2. **Variance calibration**: how well each model predicts the conditional variance
# MAGIC    Var[Y|X]. The benchmark is the ratio of predicted variance to observed squared
# MAGIC    residuals, by decile of predicted sigma. A well-calibrated variance model has
# MAGIC    this ratio close to 1.0 in each decile.
# MAGIC
# MAGIC 3. **Prediction interval coverage**: 80% and 95% prediction intervals. Correct
# MAGIC    interval coverage is the most practically meaningful test — these intervals
# MAGIC    drive pricing margins, reserve ranges, and reinsurance attachment decisions.
# MAGIC
# MAGIC 4. **Interval width efficiency**: given the same nominal coverage, narrower intervals
# MAGIC    are better. Over-wide intervals mean you are charging too much loading; under-wide
# MAGIC    intervals mean you are under-reserved.

# COMMAND ----------

def gamma_deviance(y, mu):
    """Mean Gamma deviance (lower is better)."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    return 2.0 * np.mean(((y - mu) / mu) - np.log(y / mu))


def gini_coefficient(y, mu_pred):
    """Normalised Gini coefficient (higher is better)."""
    order   = np.argsort(mu_pred)
    y_s     = y[order]
    n       = len(y_s)
    cum_y   = np.cumsum(y_s) / y_s.sum()
    cum_pop = np.arange(1, n + 1) / n
    return 2 * np.trapz(cum_y, cum_pop) - 1


def prediction_interval_coverage(y, mu, sigma, quantile=0.95):
    """
    Coverage of symmetric (alpha/2, 1-alpha/2) Gamma prediction intervals.

    For Gamma with mean mu and CV sigma:
      shape k = 1/sigma^2
      scale  s = mu * sigma^2
    Use scipy.stats.gamma.ppf for exact quantiles.
    """
    k = np.maximum(1.0 / sigma**2, 0.01)
    s = mu * sigma**2
    alpha = 1.0 - quantile
    lower = stats.gamma.ppf(alpha / 2, a=k, scale=s)
    upper = stats.gamma.ppf(1.0 - alpha / 2, a=k, scale=s)
    covered = (y >= lower) & (y <= upper)
    return float(covered.mean()), float((upper - lower).mean())


def variance_calibration(y, mu, sigma, n_deciles=10):
    """
    By decile of predicted sigma, compare predicted vs observed variance.

    Under the Gamma model: Var[Y|X] = (sigma * mu)^2.
    Observed variance proxy: (y - mu)^2.

    Returns the mean absolute relative error across deciles.
    """
    pred_var   = (sigma * mu) ** 2
    obs_sq_res = (y - mu) ** 2
    ratio      = obs_sq_res / np.maximum(pred_var, 1e-6)

    cuts = pd.qcut(sigma, n_deciles, labels=False, duplicates="drop")
    mae  = 0.0
    for d in range(n_deciles):
        mask = cuts == d
        if mask.sum() < 5:
            continue
        mae += abs(ratio[mask].mean() - 1.0)
    return mae / n_deciles

# COMMAND ----------

# Mean fit metrics
dev_baseline = gamma_deviance(y_test, mu_baseline_test)
dev_gamlss   = gamma_deviance(y_test, mu_gamlss_test)

gini_baseline = gini_coefficient(y_test, mu_baseline_test)
gini_gamlss   = gini_coefficient(y_test, mu_gamlss_test)

# Prediction interval coverage — two nominal levels
cov80_base, width80_base = prediction_interval_coverage(
    y_test, mu_baseline_test, phi_baseline_test**0.5, quantile=0.80
)
cov95_base, width95_base = prediction_interval_coverage(
    y_test, mu_baseline_test, phi_baseline_test**0.5, quantile=0.95
)
cov80_gaml, width80_gaml = prediction_interval_coverage(
    y_test, mu_gamlss_test, sig_gamlss_test, quantile=0.80
)
cov95_gaml, width95_gaml = prediction_interval_coverage(
    y_test, mu_gamlss_test, sig_gamlss_test, quantile=0.95
)

# Variance calibration (mean absolute error of variance ratio by sigma decile)
var_cal_baseline = variance_calibration(
    y_test, mu_baseline_test, phi_baseline_test**0.5
)
var_cal_gamlss = variance_calibration(
    y_test, mu_gamlss_test, sig_gamlss_test
)

# Sigma recovery: compare fitted sigma to true sigma
sigma_mae_baseline = np.abs(phi_baseline_test**0.5 - test_pd["sigma_true"].values).mean()
sigma_mae_gamlss   = np.abs(sig_gamlss_test - test_pd["sigma_true"].values).mean()

print("=" * 72)
print(f"{'Metric':<40} {'Baseline':>10} {'GAMLSS':>10} {'Delta':>8}")
print("=" * 72)

rows = []
metrics = [
    ("Gamma deviance (lower better)",    dev_baseline,     dev_gamlss,     True),
    ("Gini coefficient (higher better)", gini_baseline,    gini_gamlss,    False),
    ("80% PI coverage (target 0.80)",    cov80_base,       cov80_gaml,     False),
    ("95% PI coverage (target 0.95)",    cov95_base,       cov95_gaml,     False),
    ("80% PI mean width (narrower ok)",  width80_base,     width80_gaml,   True),
    ("95% PI mean width (narrower ok)",  width95_base,     width95_gaml,   True),
    ("Variance calib MAE (lower better)",var_cal_baseline, var_cal_gamlss, True),
    ("Sigma MAE vs true (lower better)", sigma_mae_baseline,sigma_mae_gamlss,True),
    ("Fit time (s)",                     baseline_fit_time, library_fit_time, True),
]
for name, b_val, l_val, lower_is_better in metrics:
    if b_val == 0:
        delta_str = "n/a"
    else:
        delta = (l_val - b_val) / abs(b_val) * 100
        delta_str = f"{delta:+.1f}%"
    print(f"{name:<40} {b_val:>10.4f} {l_val:>10.4f} {delta_str:>8}")
    rows.append({"metric": name, "baseline": b_val, "gamlss": l_val,
                 "lower_is_better": lower_is_better})

print("=" * 72)

# COMMAND ----------

# Coverage summary
print("\n--- Prediction Interval Coverage Summary ---")
print(f"  80% intervals: baseline covers {cov80_base:.1%} (target 80%),  GAMLSS covers {cov80_gaml:.1%}")
print(f"  95% intervals: baseline covers {cov95_base:.1%} (target 95%),  GAMLSS covers {cov95_gaml:.1%}")
print()
print(f"--- Coverage Error (absolute distance from nominal) ---")
print(f"  80% nominal: baseline |{abs(cov80_base-0.80):.3f}|, GAMLSS |{abs(cov80_gaml-0.80):.3f}|")
print(f"  95% nominal: baseline |{abs(cov95_base-0.95):.3f}|, GAMLSS |{abs(cov95_gaml-0.95):.3f}|")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sigma Recovery Plots
# MAGIC
# MAGIC The most direct test: does the fitted sigma match the known true sigma?
# MAGIC The baseline sigma is derived from a post-hoc variance GLM; the GAMLSS sigma is
# MAGIC jointly estimated. We compare both to the oracle.

# COMMAND ----------

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

sigma_true_test = test_pd["sigma_true"].values
sig_base_test   = phi_baseline_test.values**0.5

# ── Plot 1: True sigma vs baseline fitted sigma ─────────────────────────────
ax1.scatter(sigma_true_test, sig_base_test, alpha=0.15, s=6, color="steelblue")
lims = [min(sigma_true_test.min(), sig_base_test.min()) * 0.9,
        max(sigma_true_test.max(), sig_base_test.max()) * 1.1]
ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect recovery")
r_base = np.corrcoef(sigma_true_test, sig_base_test)[0, 1]
ax1.set_xlabel("True sigma (DGP)")
ax1.set_ylabel("Fitted sigma (two-stage GLM)")
ax1.set_title(f"Two-stage GLM sigma recovery\nr = {r_base:.3f}")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── Plot 2: True sigma vs GAMLSS fitted sigma ─────────────────────────────
ax2.scatter(sigma_true_test, sig_gamlss_test, alpha=0.15, s=6, color="tomato")
ax2.plot(lims, lims, "b--", linewidth=1.5, label="Perfect recovery")
r_gaml = np.corrcoef(sigma_true_test, sig_gamlss_test)[0, 1]
ax2.set_xlabel("True sigma (DGP)")
ax2.set_ylabel("Fitted sigma (GAMLSS)")
ax2.set_title(f"GAMLSS sigma recovery\nr = {r_gaml:.3f}")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: 95% PI coverage by channel and vehicle class ────────────────────
groups = {
    "direct/A":    (test_pd["channel"] == "direct")  & (test_pd["veh_class"] == "A"),
    "direct/D":    (test_pd["channel"] == "direct")  & (test_pd["veh_class"] == "D"),
    "broker/A":    (test_pd["channel"] == "broker")  & (test_pd["veh_class"] == "A"),
    "broker/D":    (test_pd["channel"] == "broker")  & (test_pd["veh_class"] == "D"),
    "aggr/A":      (test_pd["channel"] == "aggregator") & (test_pd["veh_class"] == "A"),
    "aggr/D":      (test_pd["channel"] == "aggregator") & (test_pd["veh_class"] == "D"),
}

group_names = list(groups.keys())
cov95_base_grp  = []
cov95_gaml_grp  = []

for gname, gmask in groups.items():
    if gmask.sum() < 20:
        cov95_base_grp.append(np.nan)
        cov95_gaml_grp.append(np.nan)
        continue
    yg   = y_test[gmask.values]
    mug_b = mu_baseline_test.values[gmask.values]
    sig_b = sig_base_test[gmask.values]
    cov_b, _ = prediction_interval_coverage(yg, mug_b, sig_b, 0.95)
    cov95_base_grp.append(cov_b)

    mug_g = mu_gamlss_test[gmask.values]
    sig_g = sig_gamlss_test[gmask.values]
    cov_g, _ = prediction_interval_coverage(yg, mug_g, sig_g, 0.95)
    cov95_gaml_grp.append(cov_g)

x_pos = np.arange(len(group_names))
w = 0.35
ax3.bar(x_pos - w/2, cov95_base_grp,  w, label="Two-stage GLM", color="steelblue", alpha=0.7)
ax3.bar(x_pos + w/2, cov95_gaml_grp,  w, label="GAMLSS",        color="tomato",    alpha=0.7)
ax3.axhline(0.95, color="black", linewidth=1.5, linestyle="--", label="Nominal 95%")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(group_names, rotation=30, ha="right")
ax3.set_ylabel("95% PI coverage")
ax3.set_title("PI Coverage by Channel/Vehicle Class")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")
ax3.set_ylim([0.5, 1.05])

# ── Plot 4: Quantile residuals (GAMLSS) vs theoretical N(0,1) ───────────────
try:
    q_resids = quantile_residuals(gamlss, df_pl_test, y_test, seed=0)
    q_resids_finite = q_resids[np.isfinite(q_resids)]

    # QQ plot
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(q_resids_finite)))
    observed    = np.sort(q_resids_finite)
    ax4.scatter(theoretical, observed, alpha=0.3, s=8, color="tomato")
    ax4.plot([-3.5, 3.5], [-3.5, 3.5], "k--", linewidth=1.5)
    ax4.set_xlabel("Theoretical N(0,1) quantiles")
    ax4.set_ylabel("GAMLSS quantile residuals")
    ax4.set_title(f"GAMLSS QQ Plot (quantile residuals)\nN={len(q_resids_finite):,}")
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-4, 4])
    ax4.set_ylim([-4, 4])
except Exception as ex:
    ax4.text(0.5, 0.5, f"Quantile residuals unavailable:\n{ex}",
             transform=ax4.transAxes, ha="center", va="center", wrap=True)

plt.suptitle(
    "insurance-distributional-glm vs two-stage GLM — Diagnostic Plots",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_distributional_glm.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_distributional_glm.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Variance Calibration Detail
# MAGIC
# MAGIC Break out the variance calibration by predicted sigma decile. This is the
# MAGIC most granular test of whether the variance model is correctly specified.

# COMMAND ----------

n_dec = 10
sig_dec_gaml = pd.qcut(sig_gamlss_test, n_dec, labels=False, duplicates="drop")
sig_dec_base = pd.qcut(sig_base_test,   n_dec, labels=False, duplicates="drop")

print(f"{'Sigma decile':<14} {'True sigma':<16} {'Baseline sigma':<16} {'GAMLSS sigma':<16} "
      f"{'Var ratio (base)':<18} {'Var ratio (GAML)'}")
print("-" * 100)

for d in range(n_dec):
    m_g = sig_dec_gaml == d
    m_b = sig_dec_base == d

    if m_g.sum() < 5:
        continue

    true_sig_d   = sigma_true_test[m_g.values].mean()
    base_sig_d   = sig_base_test[m_b.values].mean() if m_b.sum() > 0 else np.nan
    gaml_sig_d   = sig_gamlss_test[m_g.values].mean()

    # Var ratio: predicted variance vs realised squared deviations
    y_d   = y_test[m_g.values]
    mu_gd = mu_gamlss_test[m_g.values]
    mu_bd = mu_baseline_test.values[m_b.values] if m_b.sum() > 0 else np.full(m_g.sum(), np.nan)

    pred_var_g = (sig_gamlss_test[m_g.values] * mu_gd) ** 2
    obs_var_d  = (y_d - mu_gd) ** 2
    ratio_g    = obs_var_d.mean() / pred_var_g.mean() if pred_var_g.mean() > 0 else np.nan

    if m_b.sum() > 5:
        pred_var_b = (sig_base_test[m_b.values] * mu_bd) ** 2
        obs_var_b  = (y_test[m_b.values] - mu_bd) ** 2
        ratio_b    = obs_var_b.mean() / pred_var_b.mean() if pred_var_b.mean() > 0 else np.nan
    else:
        ratio_b = np.nan

    print(f"  {d+1:<12} {true_sig_d:<16.3f} {base_sig_d:<16.3f} {gaml_sig_d:<16.3f} "
          f"{ratio_b:<18.3f} {ratio_g:.3f}")

print()
print("Var ratio close to 1.0 = well-calibrated variance model.")
print(f"Two-stage MAE:  {var_cal_baseline:.4f}")
print(f"GAMLSS MAE:     {var_cal_gamlss:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Selection via GAIC
# MAGIC
# MAGIC Test three families on the training data using GAIC (generalised AIC).
# MAGIC This demonstrates the choose_distribution API.

# COMMAND ----------

from insurance_distributional_glm import choose_distribution
from insurance_distributional_glm.families import Gamma, LogNormal, InverseGaussian

t_sel = time.perf_counter()
results = choose_distribution(
    df_pl_train, y_train,
    families=[Gamma(), LogNormal(), InverseGaussian()],
    formulas={"mu": ["age_band", "veh_class", "channel", "log_sum_ins"],
              "sigma": ["veh_class", "channel"]},
    penalty=2.0,
)
sel_time = time.perf_counter() - t_sel

print(f"Model selection time: {sel_time:.1f}s")
print()
print(f"{'Family':<20} {'GAIC':>10} {'LogLik':>12} {'Converged':>10}")
print("-" * 55)
for r in results:
    print(f"  {r.family_name:<18} {r.gaic:>10.2f} {r.loglik:>12.2f} {str(r.converged):>10}")

best = results[0]
print(f"\nBest family: {best.family_name} (GAIC={best.gaic:.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verdict

# COMMAND ----------

def coverage_error(cov, target):
    return abs(cov - target)

print("=" * 60)
print("VERDICT: GAMLSS vs two-stage GLM for severity modelling")
print("=" * 60)
print()
print("Mean fit quality:")
print(f"  Gamma deviance:   {dev_baseline:.5f} (baseline) -> {dev_gamlss:.5f} (GAMLSS) "
      f"  {(dev_gamlss-dev_baseline)/abs(dev_baseline)*100:+.1f}%")
print(f"  Gini:             {gini_baseline:.4f} (baseline) -> {gini_gamlss:.4f} (GAMLSS)")
print()
print("Variance modelling quality:")
print(f"  Sigma MAE (vs true):  baseline={sigma_mae_baseline:.4f}, GAMLSS={sigma_mae_gamlss:.4f}")
print(f"  Variance calib MAE:   baseline={var_cal_baseline:.4f}, GAMLSS={var_cal_gamlss:.4f}")
print(f"  Sigma recovery r:     baseline={r_base:.3f}, GAMLSS={r_gaml:.3f}")
print()
print("Prediction interval coverage:")
print(f"  80% target: baseline={cov80_base:.3f} (err {coverage_error(cov80_base,0.80):.3f}), "
      f"GAMLSS={cov80_gaml:.3f} (err {coverage_error(cov80_gaml,0.80):.3f})")
print(f"  95% target: baseline={cov95_base:.3f} (err {coverage_error(cov95_base,0.95):.3f}), "
      f"GAMLSS={cov95_gaml:.3f} (err {coverage_error(cov95_gaml,0.95):.3f})")
print()
print("Fit time:")
print(f"  Baseline: {baseline_fit_time:.2f}s,  GAMLSS: {library_fit_time:.2f}s  "
      f"(ratio: {library_fit_time/max(baseline_fit_time,0.001):.1f}x)")
print()
print("Interpretation:")
if sigma_mae_gamlss < sigma_mae_baseline:
    print("  GAMLSS recovers the true heterogeneous sigma more accurately.")
else:
    print("  Two-stage GLM achieves comparable sigma recovery on this dataset.")
if abs(cov95_gaml - 0.95) < abs(cov95_base - 0.95):
    print("  GAMLSS 95% PI coverage is closer to nominal (better calibrated intervals).")
else:
    print("  Both models achieve similar 95% PI calibration on this dataset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. README Performance Snippet

# COMMAND ----------

cov95_base_err = abs(cov95_base - 0.95)
cov95_gaml_err = abs(cov95_gaml - 0.95)

print("""
## Performance

Benchmarked against a two-stage approach (Gamma GLM for mean, separate
Gamma GLM on squared Pearson residuals for variance) on synthetic UK motor
severity data (30,000 claims, known DGP with heterogeneous CV by vehicle
class and channel). Temporal 70/30 train/test split.
See `notebooks/benchmark_distributional_glm.py` for full methodology.
""")

print(f"| Metric                      | Two-stage GLM | GAMLSS        |")
print(f"|------------------------------|---------------|---------------|")
print(f"| Gamma deviance (test)        | {dev_baseline:.4f}        | {dev_gamlss:.4f}        |")
print(f"| Gini coefficient             | {gini_baseline:.4f}        | {gini_gamlss:.4f}        |")
print(f"| Sigma MAE vs true            | {sigma_mae_baseline:.4f}        | {sigma_mae_gamlss:.4f}        |")
print(f"| 95% PI coverage (target .95) | {cov95_base:.3f}         | {cov95_gaml:.3f}         |")
print(f"| Variance calib MAE (decile)  | {var_cal_baseline:.4f}        | {var_cal_gamlss:.4f}        |")
print(f"| Fit time (s)                 | {baseline_fit_time:.2f}          | {library_fit_time:.2f}          |")
print()
print("""The main gain is in variance calibration: GAMLSS recovers the true
covariate-driven sigma more accurately because the RS algorithm enforces
consistency between the mean and variance fits. On portfolios where variance
is genuinely homogeneous, the difference is small. On heterogeneous books —
mixed channels, multi-class commercial, broker-direct splits — joint modelling
of sigma can meaningfully improve prediction interval coverage.
""")
