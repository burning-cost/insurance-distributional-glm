# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-distributional-glm (GAMLSS) vs standard Gamma GLM
# MAGIC
# MAGIC **Library:** `insurance-distributional-glm` — GAMLSS for insurance pricing.
# MAGIC Jointly models mu (mean) and sigma (CV) as functions of covariates using the
# MAGIC RS algorithm. The standard Gamma GLM treats sigma as a single constant for
# MAGIC the entire portfolio.
# MAGIC
# MAGIC **Baseline:** Gamma GLM (constant phi) — the industry default. statsmodels
# MAGIC `Gamma(link=Log())` estimates one dispersion parameter for all observations.
# MAGIC No covariate-dependent variance.
# MAGIC
# MAGIC **Dataset:** 25,000 synthetic UK motor severity claims. Variance is genuinely
# MAGIC heterogeneous by vehicle class (A–D) and distribution channel (direct, broker,
# MAGIC aggregator). Vehicle D + broker = 3x higher CV than vehicle A + direct.
# MAGIC
# MAGIC **Date:** 2026-03-15
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC When sigma genuinely depends on covariates, the standard GLM has two problems:
# MAGIC 1. Prediction intervals are wrong — too wide for low-CV risks, too narrow for
# MAGIC    high-CV risks. This directly affects pricing margins and reserve ranges.
# MAGIC 2. The mean fit degrades because IRLS weights use the wrong phi.
# MAGIC GAMLSS fixes both by jointly estimating mu and sigma.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-distributional-glm statsmodels numpy scipy matplotlib polars pandas

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

from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

warnings.filterwarnings("ignore")
print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Synthetic Data with Heterogeneous Variance

# COMMAND ----------

rng = np.random.default_rng(42)
N = 25_000

age_band  = rng.choice(["17-24","25-34","35-49","50-64","65+"], N, p=[0.10,0.22,0.30,0.26,0.12])
veh_class = rng.choice(["A","B","C","D"], N, p=[0.35,0.30,0.22,0.13])
channel   = rng.choice(["direct","broker","aggregator"], N, p=[0.45,0.30,0.25])
sum_ins   = rng.lognormal(9.5, 0.6, N)

# True mu
mu_log  = 6.8
mu_log += np.where(age_band=="17-24",  0.45, np.where(age_band=="25-34", 0.18, np.where(age_band=="50-64",-0.08,np.where(age_band=="65+",-0.05,0.0))))
mu_log += np.where(veh_class=="B",0.25,np.where(veh_class=="C",0.55,np.where(veh_class=="D",0.90,0.0)))
mu_log += np.where(channel=="broker",0.15,np.where(channel=="aggregator",0.08,0.0))
mu_log += 0.30 * (np.log(sum_ins) - 9.5)
mu_true = np.exp(mu_log)

# True sigma — heterogeneous by design
sigma_log  = -1.20
sigma_log += np.where(veh_class=="B",0.18,np.where(veh_class=="C",0.38,np.where(veh_class=="D",0.62,0.0)))
sigma_log += np.where(channel=="broker",0.35,np.where(channel=="aggregator",0.15,0.0))
sigma_true = np.exp(sigma_log)

k_true = 1.0 / sigma_true**2
y = rng.gamma(shape=k_true, scale=mu_true / k_true)

df_pd = pd.DataFrame({
    "age_band": age_band, "veh_class": veh_class, "channel": channel,
    "sum_ins": sum_ins, "log_sum_ins": np.log(sum_ins),
    "y": y, "mu_true": mu_true, "sigma_true": sigma_true,
})

train_pd = df_pd.iloc[:int(N*0.70)].copy()
test_pd  = df_pd.iloc[int(N*0.70):].copy()

print(f"Train: {len(train_pd):,}  Test: {len(test_pd):,}")
print(f"True sigma range: [{sigma_true.min():.3f}, {sigma_true.max():.3f}]")
print(f"Sigma CoV (heterogeneity): {sigma_true.std()/sigma_true.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: Standard Gamma GLM (constant phi)

# COMMAND ----------

formula_mean = "y ~ C(age_band) + C(veh_class) + C(channel) + log_sum_ins"

t0_base = time.perf_counter()
glm_mean = smf.glm(formula_mean, data=train_pd,
                   family=sm.families.Gamma(link=sm.families.links.Log())).fit()
baseline_time = time.perf_counter() - t0_base

mu_base_test = glm_mean.predict(test_pd)
# Constant phi for all observations
phi_base = float(glm_mean.scale)
sigma_base_test = np.full(len(test_pd), np.sqrt(phi_base))

print(f"Baseline fit time: {baseline_time:.2f}s")
print(f"Constant phi: {phi_base:.4f}  =>  sigma = {np.sqrt(phi_base):.4f} (same for all risks)")
print(f"Deviance explained: {(1 - glm_mean.deviance/glm_mean.null_deviance):.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: DistributionalGLM (GAMLSS)

# COMMAND ----------

FEATURES = ["age_band", "veh_class", "channel", "log_sum_ins"]

df_pl_train = pl.from_pandas(train_pd[FEATURES])
df_pl_test  = pl.from_pandas(test_pd[FEATURES])
y_train = train_pd["y"].values
y_test  = test_pd["y"].values

t0_lib = time.perf_counter()
gamlss = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "veh_class", "channel", "log_sum_ins"],
        "sigma": ["veh_class", "channel"],
    },
)
gamlss.fit(df_pl_train, y_train, max_iter=100, tol=1e-7)
lib_time = time.perf_counter() - t0_lib

mu_gamlss_test  = gamlss.predict(df_pl_test, parameter="mu")
sig_gamlss_test = gamlss.predict(df_pl_test, parameter="sigma")

print(f"GAMLSS fit time: {lib_time:.2f}s  Converged: {gamlss.converged}")
print(f"Log-likelihood:  {gamlss.loglikelihood:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics

# COMMAND ----------

def gamma_deviance(y, mu):
    y = np.asarray(y, float); mu = np.maximum(np.asarray(mu, float), 1e-10)
    return 2.0 * np.mean(((y - mu)/mu) - np.log(y/mu))

def gini(y, mu):
    order = np.argsort(mu); ys = y[order]
    n = len(ys); cum = np.cumsum(ys)/ys.sum(); pop = np.arange(1,n+1)/n
    return float(2 * np.trapz(cum, pop) - 1)

def pi_coverage(y, mu, sigma, q=0.95):
    k = np.maximum(1.0/sigma**2, 0.01); s = mu * sigma**2
    lo = stats.gamma.ppf((1-q)/2, a=k, scale=s)
    hi = stats.gamma.ppf(1-(1-q)/2, a=k, scale=s)
    return float(((y >= lo) & (y <= hi)).mean()), float((hi-lo).mean())

def var_calib_mae(y, mu, sigma, n=10):
    pred_var = (sigma*mu)**2; obs = (y-mu)**2
    ratio = obs / np.maximum(pred_var, 1e-6)
    cuts = pd.qcut(sigma, n, labels=False, duplicates="drop")
    return float(np.mean([abs(ratio[cuts==d].mean()-1.0) for d in range(n) if (cuts==d).sum()>=5]))

y_t = y_test; sig_true_t = test_pd["sigma_true"].values

dev_base  = gamma_deviance(y_t, mu_base_test)
dev_gaml  = gamma_deviance(y_t, mu_gamlss_test)
gini_base = gini(y_t, mu_base_test)
gini_gaml = gini(y_t, mu_gamlss_test)

cov80_b, w80_b = pi_coverage(y_t, mu_base_test, sigma_base_test, 0.80)
cov95_b, w95_b = pi_coverage(y_t, mu_base_test, sigma_base_test, 0.95)
cov80_g, w80_g = pi_coverage(y_t, mu_gamlss_test, sig_gamlss_test, 0.80)
cov95_g, w95_g = pi_coverage(y_t, mu_gamlss_test, sig_gamlss_test, 0.95)

vc_base = var_calib_mae(y_t, mu_base_test, sigma_base_test)
vc_gaml = var_calib_mae(y_t, mu_gamlss_test, sig_gamlss_test)

sm_base = float(np.abs(sigma_base_test - sig_true_t).mean())
sm_gaml = float(np.abs(sig_gamlss_test - sig_true_t).mean())

print("=" * 72)
print(f"{'Metric':<40} {'Gamma GLM':>10} {'GAMLSS':>10} {'Delta':>8}")
print("=" * 72)
rows = [
    ("Gamma deviance (lower better)",    dev_base,  dev_gaml,  True),
    ("Gini coefficient (higher better)", gini_base, gini_gaml, False),
    ("80% PI coverage (target 0.80)",   cov80_b,   cov80_g,   False),
    ("95% PI coverage (target 0.95)",   cov95_b,   cov95_g,   False),
    ("80% PI mean width",               w80_b,     w80_g,     True),
    ("95% PI mean width",               w95_b,     w95_g,     True),
    ("Variance calib MAE (lower better)",vc_base,  vc_gaml,   True),
    ("Sigma MAE vs true (lower better)", sm_base,  sm_gaml,   True),
    ("Fit time (s)",                     baseline_time, lib_time, True),
]
for name, b, l, low in rows:
    d = (l - b) / abs(b) * 100 if b != 0 else 0.0
    print(f"{name:<40} {b:>10.4f} {l:>10.4f} {d:>+7.1f}%")
print("=" * 72)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Plots

# COMMAND ----------

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot 1: True sigma vs fitted sigma
ax1.scatter(sig_true_t[:2000], sigma_base_test[:2000], alpha=0.15, s=6, c="steelblue", label=f"Gamma GLM (r={np.corrcoef(sig_true_t, sigma_base_test)[0,1]:.3f})")
ax1.scatter(sig_true_t[:2000], sig_gamlss_test[:2000], alpha=0.15, s=6, c="tomato", label=f"GAMLSS (r={np.corrcoef(sig_true_t, sig_gamlss_test)[0,1]:.3f})")
lim = [sig_true_t.min()*0.9, sig_true_t.max()*1.1]
ax1.plot(lim, lim, "k--", linewidth=1.5)
ax1.set_xlabel("True sigma (DGP)"); ax1.set_ylabel("Fitted sigma")
ax1.set_title("Sigma recovery: constant GLM vs GAMLSS"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Plot 2: 95% PI coverage by vehicle class / channel
groups = {"A/direct":(test_pd.veh_class=="A")&(test_pd.channel=="direct"),
          "D/direct":(test_pd.veh_class=="D")&(test_pd.channel=="direct"),
          "A/broker":(test_pd.veh_class=="A")&(test_pd.channel=="broker"),
          "D/broker":(test_pd.veh_class=="D")&(test_pd.channel=="broker")}
g_names = list(groups.keys())
cov_b_g, cov_g_g = [], []
for gm in groups.values():
    if gm.sum() < 20:
        cov_b_g.append(np.nan); cov_g_g.append(np.nan); continue
    yg = y_t[gm.values]
    cb, _ = pi_coverage(yg, mu_base_test[gm.values], sigma_base_test[gm.values])
    cg, _ = pi_coverage(yg, mu_gamlss_test[gm.values], sig_gamlss_test[gm.values])
    cov_b_g.append(cb); cov_g_g.append(cg)

xp = np.arange(len(g_names))
ax2.bar(xp-0.2, cov_b_g, 0.4, label="Gamma GLM", color="steelblue", alpha=0.8)
ax2.bar(xp+0.2, cov_g_g, 0.4, label="GAMLSS", color="tomato", alpha=0.8)
ax2.axhline(0.95, color="black", linestyle="--", linewidth=1.5, label="Nominal 95%")
ax2.set_xticks(xp); ax2.set_xticklabels(g_names, rotation=15, ha="right")
ax2.set_ylabel("95% PI coverage"); ax2.set_title("PI coverage by segment\n(Constant phi: too narrow for D/broker)")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y"); ax2.set_ylim([0.5, 1.05])

# Plot 3: Variance calibration by sigma decile
sig_dec = pd.qcut(sig_gamlss_test, 10, labels=False, duplicates="drop")
true_s, base_s, gaml_s = [], [], []
for d in range(10):
    m = sig_dec == d
    if m.sum() < 5: continue
    true_s.append(sig_true_t[m.values].mean())
    base_s.append(sigma_base_test[m.values].mean())
    gaml_s.append(sig_gamlss_test[m.values].mean())

ax3.plot(range(1,len(true_s)+1), true_s, "k-o", linewidth=2, label="True sigma", markersize=6)
ax3.plot(range(1,len(base_s)+1), base_s, "b--s", linewidth=2, label="Gamma GLM (constant)", markersize=6)
ax3.plot(range(1,len(gaml_s)+1), gaml_s, "r-^", linewidth=2, label="GAMLSS", markersize=6)
ax3.set_xlabel("GAMLSS sigma decile"); ax3.set_ylabel("Mean sigma")
ax3.set_title("Sigma by risk decile: recovery comparison")
ax3.legend(); ax3.grid(True, alpha=0.3)

# Plot 4: Coverage error by PI level
pi_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
err_base, err_gaml = [], []
for pi in pi_levels:
    cb, _ = pi_coverage(y_t, mu_base_test, sigma_base_test, pi)
    cg, _ = pi_coverage(y_t, mu_gamlss_test, sig_gamlss_test, pi)
    err_base.append(abs(cb - pi)); err_gaml.append(abs(cg - pi))

ax4.plot(pi_levels, err_base, "b-o", linewidth=2, label=f"Gamma GLM (mean={np.mean(err_base):.3f})")
ax4.plot(pi_levels, err_gaml, "r-^", linewidth=2, label=f"GAMLSS (mean={np.mean(err_gaml):.3f})")
ax4.set_xlabel("Nominal PI level"); ax4.set_ylabel("|Actual coverage - nominal|")
ax4.set_title("PI coverage error across nominal levels"); ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle("insurance-distributional-glm: GAMLSS vs Standard Gamma GLM",
             fontsize=13, fontweight="bold")
plt.savefig("/tmp/benchmark_distributional_glm.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved /tmp/benchmark_distributional_glm.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verdict

# COMMAND ----------

print("=" * 60)
print("VERDICT: GAMLSS vs Constant-phi Gamma GLM")
print("=" * 60)
print()
print("Sigma recovery:")
print(f"  Gamma GLM sigma MAE: {sm_base:.4f} (constant phi, same for all)")
print(f"  GAMLSS sigma MAE:    {sm_gaml:.4f}")
r_base = np.corrcoef(sig_true_t, sigma_base_test)[0,1]
r_gaml = np.corrcoef(sig_true_t, sig_gamlss_test)[0,1]
print(f"  Correlation with true sigma:  GLM={r_base:.3f}, GAMLSS={r_gaml:.3f}")
print()
print("Prediction interval calibration:")
print(f"  95% PI: GLM={cov95_b:.3f} (err={abs(cov95_b-0.95):.3f}), GAMLSS={cov95_g:.3f} (err={abs(cov95_g-0.95):.3f})")
print()
winner_sigma = "GAMLSS" if sm_gaml < sm_base else "GLM"
winner_pi    = "GAMLSS" if abs(cov95_g-0.95) < abs(cov95_b-0.95) else "GLM"
print(f"  Sigma MAE winner:  {winner_sigma}")
print(f"  PI coverage winner:{winner_pi}")
print()
print("Interpretation:")
print("  The Gamma GLM assigns the same phi to a vehicle-D broker policy and a")
print("  vehicle-A direct policy. Under the true DGP, their CVs differ by ~3x.")
print("  GAMLSS recovers the covariate-driven sigma, producing tighter intervals")
print("  for low-CV risks and wider intervals for high-CV risks. The overall mean")
print("  predictions are similar — but the uncertainty estimates are more honest.")

if __name__ == "__main__":
    pass
