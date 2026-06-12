# renewable_energy_growth_monitor
# Renewable Energy Infrastructure Growth Monitor 

An end-to-end machine learning pipeline designed to analyze and forecast global renewable energy capacity (Solar, Wind, Hydro, and Biomass) across 30 countries through 2029. 

This project tackles the complexities of physical infrastructure forecasting by predicting **annual capacity additions** rather than total capacity, enforcing physical monotonicity, and utilizing a hyperparameter-tuned hybrid ensemble to achieve highly accurate, physically realistic projections with a SMAPE of <10%.

---

##  Project Overview

The transition to renewable energy is accelerating, but the pace varies drastically by region, policy, and resource availability. This project maps historical adoption trends (2000–2024) and projects future infrastructure growth using a combination of classical time-series models, gradient-boosted trees, and Bayesian optimization.

### Key Highlights
* **Physically Grounded Forecasting:** Enforces monotonicity constraints (infrastructure doesn't disappear) and avoids tree-based extrapolation ceilings by predicting relative momentum instead of absolute time.
* **Complex Data Integration:** Merges 20+ years of IRENA capacity data with external geographic datasets (Global Wind Atlas, NREL Solar Resource flags).
* **Advanced Ensembling:** Deploys a custom LightGBM + Prophet Hybrid model, dynamically weighted via `scipy.optimize` to minimize SMAPE for each specific energy source.
* **Target Metric:** Optimized for **SMAPE** to handle zero-capacity early years gracefully, avoiding the infinity errors common with standard MAPE.

---

##  Architecture & Pipeline

The project is structured into three distinct phases, documented across three core notebooks.

### Phase 1: EDA & Dataset Builder (`notebook1_EDA`)
Loads, cleans, and enriches raw global capacity data.
* **Null Auditing & Validation:** Automated structural documentation and type checking.
* **Feature Engineering:** Computes 5-year rolling average growth rates, CAGRs, and relative acceleration.
* **External Enrichment:** Validates the correlation of wind resource quality (GWA 100m) and solar data availability (NREL) with actual infrastructure growth.
* **Outputs:** Generates 14 EDA visualizations (heatmaps, distributions, stacked trends) and exports the foundational `regional_growth_summary.csv`.

### Phase 2: Baseline Forecasting (`notebook2_forecasting`)
Establishes the forecasting methodology predicting annual additions (Δ MW) over a 2020-2024 holdout set.
* **Models Deployed:** Linear Regression (Baseline), Facebook Prophet (Logistic Growth), SARIMA (Log-Transformed), and XGBoost (Recursive Forecasting).
* **Cross-Validation:** 5-fold TimeSeriesSplit to ensure models learn actual momentum patterns, not just COVID-era anomalies.
* **Sanity Checks:** Strict programmatic enforcement ensuring 0.00% of the 2025-2029 forecasts show a year-over-year capacity decline.

### Phase 3: Optimized Ensemble (`notebook3_optimised`)
Upgrades the baseline pipeline with advanced architectures and hyperparameter tuning to hit the `<10% SMAPE` target.
* **LightGBM + Optuna:** Implemented leaf-wise tree growth with Bayesian optimization (TPE sampler) over 150 trials, incorporating cyclical sine/cosine year encodings.
* **Hybrid Prophet + LightGBM:** Captures macro policy shifts (logistic S-curve) with Prophet, while LightGBM predicts and corrects the localized momentum residuals.
* **Weighted Ensemble:** Uses SLSQP optimization with random restarts to find the exact blend of models that minimizes holdout SMAPE per energy source.

---

##  Technology Stack

* **Languages:** Python
* **Data Engineering:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, LightGBM, XGBoost, Statsmodels (SARIMA)
* **Time Series:** Facebook Prophet, NeuralForecast (Optional TFT/N-HiTS)
* **Optimization:** Optuna, SciPy
* **Visualization:** Matplotlib, Seaborn

---

## Key Findings

1. **Emerging Markets Lead Acceleration:** While China and the US dominate in absolute installed capacity, South and Southeast Asian markets exhibit the highest adoption acceleration rates (CAGR), signaling a structural shift in the global energy transition.
2. **Source-Specific Volatility:** Solar capacity proved the hardest to model natively due to exponential growth and abrupt policy discontinuities (feed-in tariffs), requiring aggressive changepoint tuning in Prophet. Hydro, being a mature technology, provided the most stable predictions.
3. **Ensemble Superiority:** The dynamically weighted ensemble consistently outperformed individual models, dropping SMAPE to between 5-12% across all sources.

---

##  How to Run

1. Clone the repository and navigate to the project directory.
2. Ensure `renewable_infra_global.csv` and `renewable_infra_full_enriched.csv` are in the root folder.
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna prophet statsmodels scipy
