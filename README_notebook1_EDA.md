# EDA-11 | Notebook 1 — Renewable Energy EDA & Dataset Builder

**Phase 1 of the Renewable Energy Infrastructure Growth Monitor**  
This notebook loads, cleans, analyses, and visualises global renewable energy capacity data, then exports the primary dataset deliverable used by Phase 2 (forecasting).

---

## What This Notebook Does — Step by Step

### Step 1: Setup & Configuration
Installs any missing libraries automatically. The configuration block at the top of the notebook is the **only place you need to edit** — set your file path, CAGR year window, and rapid growth threshold there.

### Step 2: Load the Dataset & Quality Audit
Loads `renewable_infra_full_enriched.csv` — a merged dataset built from IRENA capacity data, Global Wind Atlas wind speeds, and NREL solar resource flags.

Then runs a full **column-level null audit** that reports:
- Data type of every column
- How many values are missing and what percentage
- Number of unique values per column

This audit is the "dataset structure documentation" required by the assignment.

### Step 3: Global Capacity Trends (Plots 1–3)
Three views of the same global data:
- **Stacked area chart** — total GW across all sources by year. Shows how total renewable capacity has grown since 2000 and which sources dominate.
- **Line chart** — each source on its own line. Solar's exponential growth curve stands out clearly against the more linear Wind and steady Hydro.
- **Percentage mix chart** — same data but normalised to 100%, so you can see how the *share* of each source is shifting (Hydro share declining, Solar rising fast).

### Step 4: Country Comparisons (Plots 4–5)
Two different angles on country performance:
- **Plot 4** — Top 10 countries by *installed capacity* (MW) for each source. This shows who has built the most infrastructure. Large economies like China, USA, India dominate.
- **Plot 5** — Top 10 countries by *average annual growth rate* (last 5 years) for each source. Completely different picture — smaller economies with aggressive renewable policies often rank highest here. This is the early signal for where the next wave of growth is coming from.

### Step 5: Regional Heatmap (Plot 6)
Aggregates all country data to the **region level** (e.g. Asia, Europe, Americas). The heatmap shows installed GW per region per source type in the latest year. You can immediately spot which regions are mature (high Hydro, Wind) and which are just beginning Solar deployment.

### Step 6: Growth Rate Analysis (Plots 7–9)
Deep dive into how fast countries are growing:
- **Histogram + KDE (Plot 7)** — distribution of all annual growth rates. Tells you the "typical" experience and how many countries are growing very fast vs slowly.
- **Box plot by source (Plot 8)** — Solar has the highest median growth and widest spread. Hydro is the most mature and stable. Biomass is slow but consistent.
- **Median growth trend over time (Plot 9)** — is adoption accelerating or decelerating? If the line is trending up, the energy transition is gaining speed.

> Note: All growth rate plots clip values beyond the 1st–99th percentile to prevent extreme outliers (e.g. a country going from 1 MW to 500 MW = 49,900% growth) from compressing the axis.

### Step 7: External Data Enrichment — GWA & NREL (Plots 10–12)
Validates the two enrichment layers added to the dataset:

**GWA Wind Speed Analysis (Plot 10):**  
Does a country's wind resource quality (measured by GWA wind speed at 100m hub height) correlate with its wind capacity growth rate? Pearson r + scatter + trendline. A positive correlation confirms that wind resource quality drives adoption.

**NREL Solar Availability Analysis (Plot 11):**  
Countries with NREL solar irradiance data — do they grow their solar faster? NREL availability is a proxy for international data infrastructure attention. Mann-Whitney U statistical test checks if the difference between the two groups is significant.

**Correlation Matrix (Plot 12):**  
Heatmap of Pearson correlations between: capacity_mw, capacity_change_mw, growth_rate_pct, and gwa_wind_speed_100m. Identifies which numeric features move together — directly informs feature selection in Phase 2.

### Step 8: CAGR & Rapid Growth Identification (Plots 13–14)
CAGR (Compound Annual Growth Rate) gives a smoother, fairer measure of growth than a single year:

```
CAGR = (end_capacity / start_capacity) ^ (1 / years) − 1  × 100
```

Countries with CAGR above the threshold (default 15%) are flagged `is_rapid_growth = 1`.

- **Plot 13** — Top 15 fastest-growing country-source pairs (CAGR), colour-coded by source, with the threshold marked.
- **Plot 14** — Violin plot showing the distribution of CAGR across all countries per source. The width of the violin shows where most countries cluster.

### Step 9: Export `regional_growth_summary.csv`
The notebook merges three computed tables and saves one CSV with **one row per country-source pair**:

| Column | Description |
|---|---|
| country, iso3, region, sub_region, source_type | Identifiers |
| capacity_mw_{LATEST_YEAR} | Current installed capacity |
| avg_growth_rate_pct_5yr | 5-year average annual growth rate |
| capacity_mw_{CAGR_START} / _{CAGR_END} | Capacity at window endpoints |
| cagr_pct | Compound annual growth rate over window |
| is_rapid_growth | 1 if CAGR ≥ threshold, else 0 |

---

## Input Required

| File | Description |
|---|---|
| `renewable_infra_full_enriched.csv` | Enriched IRENA + GWA + NREL merged dataset |

---

## Outputs

| File | Description |
|---|---|
| `regional_growth_summary.csv` | Primary dataset deliverable — feeds into Notebook 2 |
| `plot_01_stacked_global_capacity.png` | Stacked global capacity by source |
| `plot_02_line_source_trends.png` | Year-on-year capacity per source |
| `plot_03_capacity_mix_share.png` | Percentage share by source |
| `plot_04_top10_countries_by_capacity.png` | Top 10 countries by installed MW |
| `plot_05_top10_countries_by_growth.png` | Top 10 countries by growth rate |
| `plot_06_region_source_heatmap.png` | Regional capacity heatmap |
| `plot_07_growth_rate_histogram.png` | Growth rate distribution |
| `plot_08_boxplot_growth_by_source.png` | Growth rate box plots by source |
| `plot_09_median_growth_trend.png` | Median growth rate over years |
| `plot_10_wind_speed_vs_growth.png` | GWA wind speed vs growth correlation |
| `plot_11_nrel_vs_solar_growth.png` | NREL enrichment vs solar growth |
| `plot_12_correlation_matrix.png` | Numeric feature correlation heatmap |
| `plot_13_top_rapid_growth.png` | Top 15 fastest CAGR country-source pairs |
| `plot_14_cagr_violin_by_source.png` | CAGR distribution violin plot |

---

## How to Run

1. Place `renewable_infra_full_enriched.csv` in the same folder as this notebook (or update `INPUT_PATH` in the configuration block).
2. Run all cells top to bottom. Libraries install automatically if missing.
3. All plots save as PNGs in the same directory. The summary CSV saves to `OUTPUT_DIR` (default: same directory).

---

## Key Findings (Printed at Runtime)

The final cell prints a live summary of:
- Which country has the largest installed capacity per source
- Which country has the fastest CAGR per source
- How many countries qualify as rapid growth per source

After this notebook completes, proceed to **Notebook 2 (Phase 2 — Forecasting)**.
