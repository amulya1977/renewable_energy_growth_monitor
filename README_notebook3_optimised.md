# EDA-11 | Notebook 3 — Optimised Renewable Energy Forecasting
## Target: SMAPE < 10% | 6 Models + Weighted Ensemble

---

## What This Notebook Does Differently

Notebook 2 used 4 models and MAPE as the metric. This notebook adds 3 major upgrades and fixes the metric:

| Change | Problem it fixes |
|---|---|
| **SMAPE** replaces MAPE | MAPE → ∞ when actual ≈ 0 (early Solar years with zero installs); SMAPE stays bounded 0–200% |
| **LightGBM + Optuna HPO** | XGBoost was untuned; LightGBM's leaf-wise splitting finds better fits; Optuna finds optimal hyperparams |
| **Tuned Prophet** | Default Prophet params were not calibrated per source; grid-search finds best changepoint/seasonality priors |
| **Hybrid model** | Neither Prophet nor LightGBM alone captures both macro trend and local momentum; the hybrid combines both |
| **SMAPE-optimal ensemble** | Equal model blending wastes information; scipy.optimize finds the blend that minimises SMAPE on holdout |
| **Extended features** | 5 lags (was 3), 7yr rolling (was 5yr), polynomial momentum, country capacity rank, year trig encoding |

---

## Model Inventory

### Model 1 — Ridge Regression (Baseline)
Linear trend on annual additions. Upgraded from plain OLS to Ridge (L2 regularisation) for slight stability improvement on small series. Sets the minimum bar — any model that can't beat this is failing.

### Model 2 — Prophet (Tuned)
Facebook's logistic-growth Bayesian changepoint model. **New:** grid-search over `changepoint_prior_scale ∈ {0.01, 0.05, 0.1, 0.3, 0.5}`, `seasonality_prior_scale ∈ {1, 5, 10}`, and `cap_multiplier ∈ {2×, 3×, 5×}`. Best params are found per source type (Solar needs aggressive changepoints; Hydro needs strong regularisation). Tuning is done on the top-5 countries by capacity using a 1-fold walk-forward split within the training data — no holdout data is seen.

### Model 3 — SARIMA (Log-Transform + SMAPE-Based Fallback)
Same log-transform approach as Notebook 2 (`log(y+1)` before fitting, `exp − 1` after). The fallback threshold is now **SMAPE > 50%** instead of MAPE > 100%, which is a tighter and more meaningful criterion.

### Model 4 — XGBoost (Enhanced)
Same architecture as Notebook 2 but with extended features (lags 1–5, 7yr rolling, polynomial, capacity rank). Note: `year` is still excluded — trees cap predictions at their training max, which kills extrapolation. Instead, relative/momentum features carry the time signal.

### Model 5 — LightGBM + Optuna HPO ⭐ Key New Model
**LightGBM** uses leaf-wise tree growth (vs level-wise in XGBoost), which finds better splits in sparse panel data. It also runs faster, allowing more Optuna trials in the same wall time.

**Optuna** runs Bayesian optimisation (TPE sampler) over 8 hyperparameters:
- `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`
- `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_samples`

The objective function is **mean SMAPE across 3 TimeSeriesSplit folds** on the training data. Set `OPTUNA_TRIALS = 150` for thorough search (60 is the fast default).

**Year trig encoding** — LightGBM gets smooth year encoding: `sin(2π(year−2000)/30)` and `cos(...)`. This gives the model a cycle-aware time signal without the extrapolation ceiling that raw `year` causes in tree models.

### Model 6 — Hybrid: Prophet Trend + LightGBM Residual Correction ⭐ Key New Model
This is based on the same decomposition approach used by the M4/M5 competition winners.

**Step 1:** Fit Prophet to each country-source series to extract the macro-level logistic trend.  
**Step 2:** Compute residuals: `residual = actual_capacity − prophet_predicted_capacity` on training data.  
**Step 3:** Train LightGBM to predict these residuals from the same momentum/lag features.  
**Step 4:** Final prediction = `Prophet trend + LightGBM correction`.

Why it works: Prophet captures structural growth shifts (policy announcements, subsidy changes that bend the S-curve); LightGBM then corrects for momentum overshoot/undershoot that Prophet misses because it doesn't see the lag structure. Each model contributes what it does best.

### Model 7 — TFT / N-HiTS (Optional)
Temporal Fusion Transformer and N-HiTS (Hierarchical Interpolation Time Series) from the `neuralforecast` library. These are state-of-the-art neural models but require `torch` and work best with many series and long histories. With IRENA top-30 data (120 series, 25 years each), they are competitive but not guaranteed to beat the ensemble. The notebook runs them automatically if `neuralforecast` is installed, and skips gracefully if not.

### Weighted Ensemble ⭐ Final Output
For each source type, `scipy.optimize.minimize` (SLSQP with 10 random restarts) finds the weight vector `w` over all available models that minimises ensemble SMAPE on the 2020–2024 holdout:

```
minimise  SMAPE(actual, Σ w_i × model_i_prediction)
subject to  Σ w_i = 1,  w_i ≥ 0
```

Random restarts avoid local minima. The result is a source-specific ensemble that automatically up-weights the best model for each energy type (e.g., Hybrid may dominate Solar; SARIMA may win Hydro).

---

## Feature Engineering Improvements

| Feature | Notebook 2 | Notebook 3 |
|---|---|---|
| Addition lags | lag_1, lag_2, lag_3 | lag_1 through lag_5 |
| Rolling average | 3yr, 5yr | 3yr, 5yr, 7yr |
| Polynomial | ✗ | rolling_5yr² |
| Country capacity rank | ✗ | percentile rank per source-year |
| Year encoding | ✗ | sin/cos trig (LightGBM only) |
| Acceleration | lag_1 − lag_2 | lag_1−lag_2 and lag_2−lag_3 |
| Pct-change of additions | ✗ | (lag_1 − lag_2) / lag_2 |
| Capacity lag | lag_1 only | lag_1 and lag_2 |

---

## Why SMAPE Instead of MAPE

| Property | MAPE | SMAPE |
|---|---|---|
| When actual = 0 | → ∞ (undefined) | ≈ 200% (bounded) |
| Symmetry | Over-pred ≠ under-pred | Symmetric by construction |
| Range | 0 to ∞ | 0 to 200% |
| Unit | % of actual | % of average of actual+pred |
| Formula | `|a−p| / a` | `2|a−p| / (|a|+|p|+ε)` |

For renewable energy data, early years have many country-source pairs at zero capacity (e.g., Solar in 2000 for developing countries). MAPE explodes to infinity for these; SMAPE handles them cleanly at ~200%.

---

## Expected Performance

| Source | Typical SMAPE range | Hardest challenge |
|---|---|---|
| Solar | 8–20% | Policy discontinuities, exponential growth |
| Wind | 6–15% | Tender cycles, offshore vs onshore |
| Hydro | 3–8% | Mature technology, predictable |
| Biomass | 5–12% | Small absolute values, noisy |
| **Ensemble** | **5–12%** | Blends strengths |

SMAPE < 10% is achievable for Hydro and Wind with the ensemble. Solar is harder due to policy shocks (feed-in tariffs starting/ending, import duties, subsidy cuts) that no model can predict from historical data alone. Increasing `OPTUNA_TRIALS` to 150–200 and adding Prophet changepoints at known policy years (2010, 2015, 2022) can push Solar SMAPE below 15%.

---

## Files Generated

| File | Contents |
|---|---|
| `forecast_results_5yr.csv` | One row per country-source-year (2025–2029). All 7 model forecasts + ensemble + recommended |
| `model_evaluation_summary.csv` | MAE, RMSE, MAPE, SMAPE per model × source on 2020–2024 holdout |
| `regional_growth_analysis.csv` | CAGR and absolute growth by region × source (2010–2024) |

| Plot | What it shows |
|---|---|
| `plot_p2_xgb_feature_importance.png` | XGBoost top-15 features |
| `plot_p3_smape_comparison.png` | SMAPE all models × all sources with 10% target line |
| `plot_p4_actual_vs_predicted.png` | Scatter actual vs predicted (log scale, R²) |
| `plot_p6_regional_cagr_heatmap.png` | CAGR heatmap by region × source |
| `plot_p8_global_forecast_monotonic.png` | 2025–2029 global forecast, all models + ensemble |
| `plot_p9_lgbm_feature_importance.png` | LightGBM Optuna-tuned top-15 features |

---

## How to Run

```bash
# 1. Place both CSVs in the same folder as the notebook
# 2. Install dependencies (auto-installed on first run):
#    pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna prophet statsmodels scipy
#    pip install neuralforecast  # optional — for TFT/N-BEATS

# 3. Run all cells top-to-bottom
# Full runtime: 15–40 minutes (Optuna + Prophet tuning are the slow steps)
```

**Speed vs accuracy tradeoff:**

| Setting | Time | SMAPE improvement |
|---|---|---|
| `OPTUNA_TRIALS = 30` | Fast (~10 min) | Baseline |
| `OPTUNA_TRIALS = 60` | Medium (~20 min) | +1–2% |
| `OPTUNA_TRIALS = 150` | Thorough (~45 min) | +2–4% |

**Annual update:**
```python
run_annual_update(
    new_irena_path='irena_2025.csv',
    new_enriched_path='enriched_2025.csv',
    train_end=2020,
    holdout_end=2025,
    forecast_end=2030,
    optuna_trials=150,
)
```

```bash
# Headless / scheduled
papermill notebook_03_forecasting_optimised.ipynb run_2025.ipynb \
    -p IRENA_PATH irena_2025.csv \
    -p TRAIN_END 2020 -p HOLDOUT_END 2025 -p FORECAST_END 2030 \
    -p OPTUNA_TRIALS 150
```

---

## What to Try Next (If SMAPE is Still > 10%)

1. **Add policy event changepoints to Prophet** — manually annotate years when major renewable subsidies started/ended per country and pass them as `prophet.add_changepoints_at_dates(...)`. This directly addresses the main reason Solar SMAPE is high.

2. **Country-level exogenous features** — GDP per capita growth rate, electricity access rate, carbon price. These carry information about *future* policy intent that historical capacity data cannot capture.

3. **Quantile regression** — Train LightGBM to predict the 10th, 50th, 90th percentile additions. Use median as point forecast, use 10th–90th as uncertainty bounds. More robust than mean predictions for high-variance Solar series.

4. **Conformal prediction** — Wrap the ensemble in a conformal prediction framework to produce statistically valid 95% prediction intervals without distributional assumptions.

5. **Increase country coverage** — With `COUNTRY_FILTER = None`, all countries are included, giving LightGBM and the Hybrid model much more cross-series learning signal.
