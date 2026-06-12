# EDA-11 | Notebook 2 — Renewable Energy Forecasting

**Phase 2 of the Renewable Energy Infrastructure Growth Monitor**  
Trains four ML models on historical IRENA capacity data and forecasts global renewable energy capacity (Solar, Wind, Hydro, Biomass) per country from **2025 to 2029**.

---

## What Problem Does This Notebook Solve?

Given historical data on how much renewable energy capacity each country has had every year since 2000, predict how much capacity each country will have in 2025, 2026, 2027, 2028, and 2029 — **per energy source type**.

The forecast must be physically realistic: energy infrastructure is permanent. Once a solar farm is built, it doesn't disappear. So the forecast for any year must be ≥ the previous year.

---

## The Core Design Decision — Predicting Additions, Not Totals

Instead of predicting total capacity directly, this notebook predicts **annual additions (Δ MW)** — how many megawatts are *added* each year. Then total capacity is reconstructed by summing forward.

**Why? Because predicting totals directly causes three problems:**

| Problem | What Goes Wrong |
|---|---|
| Prophet sees exponential curve | Forecasts a decline — physically impossible |
| XGBoost (tree model) | Gets stuck at 2019 ceiling — can't go higher than training max |
| SARIMA on Solar data | MAPE explodes to millions of percent |

Predicting additions avoids all three. Annual additions are near-stationary, always ≥ 0, and sum back to total capacity cleanly. This is the same approach used by IEA and BloombergNEF.

---

## Train / Test / Forecast Split

```
Train data:    2000 – 2019   (models learn from this)
Holdout data:  2020 – 2024   (models are evaluated — never seen during training)
Forecast:      2025 – 2029   (the actual output)
```

---

## Section-by-Section Walkthrough

### Section 1 — Setup & Configuration
Installs 8 libraries automatically. One configuration block controls all parameters: file paths, year windows, country filter, model thresholds. Only this block needs editing.

### Section 2 — Data Loading & Preparation
Loads two files: `renewable_infra_global.csv` (raw IRENA) and `renewable_infra_full_enriched.csv` (enriched with GWA wind speeds and NREL solar flags). Filters to the top 30 countries by installed capacity (configurable). Aggregates to one row per country-source-year. Splits into train and holdout sets.

### Section 3 — Feature Engineering
Builds the feature table used by XGBoost. For each country-source-year, computes:

| Feature | What it captures |
|---|---|
| `addition_lag_1/2/3` | MW added 1, 2, 3 years ago |
| `rolling_add_3yr/5yr` | Smoothed average of recent additions |
| `rolling_add_std` | Volatility of recent additions |
| `growth_rate_lag_1` | Last year's growth rate |
| `addition_accel` | Is momentum increasing or decreasing? |
| `capacity_lag_1` | How big was this country already? |
| `gwa_wind_speed_100m` | Wind resource quality (GWA enrichment) |
| `has_nrel_data` | Solar data availability (NREL enrichment) |

The feature `year` is **intentionally excluded** from XGBoost. Decision trees cannot extrapolate beyond their training range — if year is a feature and training ends at 2019, the tree treats 2025 as capped at 2019. Relative momentum features solve this.

### Section 4 — Cross-Validation (TimeSeriesSplit, 5 Folds)
Tests whether the model generalises across different time periods, not just the 2020–2024 holdout. TimeSeriesSplit walks forward: fold 1 trains on early years and tests on the next, fold 2 trains on more and tests on the next, etc. Consistent MAE/MAPE across all 5 folds = the model is genuinely learning patterns, not overfitting to the COVID era.

**Output:** Bar chart showing MAE, RMSE, MAPE per fold with mean line.

### Section 5 — Model 1: Linear Regression (Baseline)
Fits a linear trend on annual additions per country-source pair and projects it forward. Reconstructs total capacity by cumulative sum. Sets the minimum bar — any model that can't beat Linear Regression is failing.

### Section 6 — Model 2: Prophet (Logistic Growth)
Facebook's time-series model. Uses `growth='logistic'` with a capacity cap at 3× the maximum observed value. Logistic growth enforces an S-curve (slow start → rapid growth → natural slowdown), preventing the "bend → decline" artefact that appears when Prophet sees exponential growth in linear mode.

**Output:** Plot of actual vs holdout prediction vs 2025–2029 forecast with 95% confidence interval for the top 6 country-source pairs.

### Section 7 — Model 3: SARIMA (Log-Transform + Automatic Fallback)
Classical statistics time-series model. Two upgrades:

**Log-transform:** Applied before fitting (`log(y+1)`), back-transformed after (`exp(pred) − 1`). Converts exponential growth to linear in the transformed space, preventing MAPE from exploding to millions of percent on Solar data.

**Automatic fallback:** If SARIMA's MAPE on the holdout exceeds 100% for any source, its results are discarded and replaced by Prophet for that source in the final ensemble. This is logged explicitly.

### Section 8 — Model 4: XGBoost (Recursive Forecast)
Gradient boosting tree model trained on all countries at once (cross-series learning). At forecast time, uses a recursive loop: predict additions for 2025 → add to 2024 capacity → update lag features → predict 2026 → repeat through 2029.

**Output:** Feature importance chart showing which features XGBoost relied on most (typically recent addition lags dominate).

### Section 9 — Model Comparison + Residual Analysis
Brings all four models together:

- **MAPE comparison bar chart** — all 4 models × all 4 sources. Lower = better.
- **Actual vs Predicted scatter (log scale)** — how close predictions are to reality on the holdout. Points on the red diagonal = perfect. R² shown.
- **Residual analysis (8 panels)** — scatter of errors vs actual + histogram of error distribution. A good model has errors centred at zero with no pattern.
- **Per-source difficulty text** — explains which source is hardest to forecast (Solar, due to policy discontinuities) and which is easiest (Hydro, mature technology with incremental additions).

### Section 10 — Regional Growth Analysis
Computes CAGR (2010–2024) at the region level:

```
CAGR = (capacity_2024 / capacity_2010) ^ (1/14) − 1  ×  100
```

**Two plots:**
- **CAGR heatmap** — which region × source grew fastest (darker = faster)
- **Absolute growth stacked bar** — which regions added the most GW in total

**Saves:** `regional_growth_analysis.csv`

### Section 11 — Final Forecast + Monotonicity Enforcement + Sanity Check

**Best model selection:** For each source type, the model with the lowest MAPE on the holdout is automatically selected. SARIMA is excluded from selection for any source where the fallback was triggered.

**Monotonicity enforcement:**
```python
np.maximum.accumulate(np.insert(predictions, 0, capacity_2024))[1:]
```
Ensures every forecast year ≥ the previous year ≥ the 2024 baseline. Applied to every country-source pair. Reflects physical reality: power plants don't disappear.

**Sanity check:** After enforcement, confirms exactly 0.00% of forecast rows show year-over-year decline. ✓ SANITY CHECK PASSED must appear before the notebook is considered complete.

**Final plot:** One panel per source — historical capacity (solid) + all 4 model forecasts (dashed) + the recommended forecast (bold dotted) — for 2025–2029.

### Section 12 — Automated Update Pipeline
Defines `run_annual_update()` which re-runs the full pipeline from data loading to CSV export in one call. Also supports `papermill` for scheduled headless execution.

---

## Input Files Required

| File | Description |
|---|---|
| `renewable_infra_global.csv` | Raw IRENA capacity data |
| `renewable_infra_full_enriched.csv` | Enriched with GWA wind speed and NREL solar flags |

---

## Output Files

| File | What's in it |
|---|---|
| `forecast_results_5yr.csv` | One row per country-source-year (2025–2029). Columns: country, iso3, region, source_type, year, lr/prophet/sarima/xgb forecast MW, recommended_forecast_mw, best_model |
| `model_evaluation_summary.csv` | MAE, RMSE, MAPE per model × source on the 2020–2024 holdout |
| `regional_growth_analysis.csv` | CAGR and absolute growth by region × source (2010–2024) |

### Plots (9 total)

| File | What It Shows |
|---|---|
| `plot_cv_timeseries_split.png` | Cross-validation scores across 5 time folds |
| `plot_p1_prophet_top6.png` | Prophet actual vs holdout vs forecast — top 6 series |
| `plot_p2_xgb_feature_importance.png` | XGBoost feature importance (year excluded) |
| `plot_p3_mape_comparison.png` | All 4 models × all 4 sources MAPE comparison |
| `plot_p4_actual_vs_predicted.png` | Actual vs predicted scatter (log scale, R²) |
| `plot_p5_residual_analysis.png` | Residual scatter + distribution for all 4 models |
| `plot_p6_regional_cagr_heatmap.png` | Regional CAGR heatmap by source |
| `plot_p7_regional_absolute_growth.png` | Absolute GW added by region, stacked |
| `plot_p8_global_forecast_monotonic.png` | Global 2025–2029 forecast — all models + recommended |

---

## How to Run

1. Place both CSVs in the same folder as the notebook (or update paths in the configuration block).
2. Run all cells top to bottom. Libraries install automatically.
3. Full run takes 5–15 minutes (Prophet and SARIMA are the slow steps).

**To update annually:**

```python
run_annual_update(
    new_irena_path='irena_2025.csv',
    new_enriched_path='enriched_2025.csv',
    train_end=2020,
    holdout_end=2025,
    forecast_end=2030,
)
```

```bash
# Headless / scheduled
papermill notebook_02_forecasting.ipynb run_2025.ipynb \
    -p IRENA_PATH irena_2025.csv \
    -p TRAIN_END 2020 \
    -p HOLDOUT_END 2025 \
    -p FORECAST_END 2030
```

---

## Key Finding

While China and the United States dominate in absolute renewable capacity, **emerging markets in South and Southeast Asia show the highest adoption acceleration rates for Solar and Wind**. This signals a structural shift in the global energy transition — the next decade of growth is coming from developing economies, not the countries that led the first wave.
