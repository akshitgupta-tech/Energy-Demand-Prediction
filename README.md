# India Hourly Energy Demand Forecasting
### VMD–XGBoost–LSTM Hybrid Ensemble Framework

A machine learning framework for 1-hour-ahead national electricity demand forecasting on Indian regional grid data. The pipeline combines exploratory analysis, Variational Mode Decomposition (VMD) for signal decomposition, classical ML regressors, a deep LSTM, and multiple ensemble strategies — with a rigorous emphasis on preventing data leakage.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Models](#models)
- [Ensemble Strategies](#ensemble-strategies)
- [Feature Engineering](#feature-engineering)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Outputs](#outputs)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

This project forecasts India's **National Hourly Electricity Demand (MW)** one hour ahead, using a hybrid approach that:

1. Decomposes the demand signal into intrinsic mode functions via **VMD** (8 modes)
2. Trains individual regressors: XGBoost, Random Forest, SVR, KNN, MLR
3. Trains a 2-layer **LSTM** on 168-hour (1 week) sliding windows
4. Combines XGBoost + LSTM into a **VMD–XGBoost–LSTM hybrid**
5. Evaluates multiple ensemble fusion methods (simple average, weighted average, inverse-MAPE weighting, stacking, voting)

All splits are **strictly chronological** (70/20/10 train/val/test) with no data leakage between splits.

---

## Pipeline Architecture

```
Raw Data (hourlyLoadDataIndia.xlsx)
        |
        v
Exploratory Data Analysis
  - Time series visualization (full, weekly, daily)
  - Distribution analysis (histogram, Q-Q, KDE)
  - Seasonal patterns (hour/day/month heatmaps)
  - ACF, trend decomposition (Savitzky-Golay)
  - Outlier detection (Z-score & IQR)
  - Regional demand comparison
        |
        v
Preprocessing
  - Chronological train/val/test split BEFORE scaling
  - MinMaxScaler fit on train only
  - IQR-based outlier removal
        |
        v
Feature Engineering
  - Lag features: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 168h, 336h
  - Rolling statistics: mean & std over 6h, 12h, 24h, 48h windows
  - Cyclic time encoding: sin/cos for hour, day-of-week, month
  - Binary flags: is_peak_hour, is_weekend
        |
        v
VMD Decomposition (K=8 modes)
  - Fit on full signal, indexed by split boundary
  - IMF columns appended to feature set for LSTM branch
        |
        v
Model Training
  - XGBoost / Random Forest / SVR / KNN / MLR (tabular features)
  - LSTM (sequence length = 168, on VMD-augmented features)
  - VMD–XGBoost–LSTM hybrid (weighted combination)
        |
        v
Ensemble Fusion
  - Simple average / Weighted average / Inverse-MAPE weights
  - Voting ensemble / Stacking ensemble
        |
        v
Evaluation & Export
  - MAE, RMSE, MAPE, R² on test set
  - Residual analysis (scatter, histogram, ACF, Q-Q)
  - Saved models (.pkl), predictions (.csv), plots (.png)
```

---

## Models

| Model | Type | Notes |
|---|---|---|
| XGBoost | Gradient Boosting | 1000 estimators, depth 8, hist tree method |
| Random Forest | Bagging Ensemble | 1000 estimators, depth 25 |
| SVR | Kernel Method | RBF kernel, C=10000; requires StandardScaler |
| KNN | Instance-Based | k=5 neighbors |
| MLR | Linear Regression | Baseline |
| LSTM | Deep Learning | 2-layer (128→64 units), seq_len=168, early stopping |
| VMD–XGBoost–LSTM | Hybrid | XGBoost on flattened VMD+lag features + LSTM on sequences, blended 50/50 |

---

## Ensemble Strategies

- **Simple Average** — equal weight across all base models
- **Weighted Average** — validation MAPE-based weights, normalized
- **Inverse-MAPE Weighting** — models with lower validation MAPE get higher weight
- **Voting Ensemble** — scikit-learn `VotingRegressor`
- **Stacking Ensemble** — meta-learner (Linear Regression) on base model predictions

The best ensemble is selected by MAPE on the held-out test set.

---

## Feature Engineering

Features are constructed with strict leakage-prevention: rolling stats and lags only reference `t-1` or earlier, and the scaler is fit exclusively on training data.

**Lag features** (hours back): `1, 2, 3, 6, 12, 24, 48, 72, 168, 336`

**Rolling statistics** (window sizes): `6h, 12h, 24h, 48h` — mean and std

**Cyclic encoding**:
```
sin_hour = sin(2π × hour / 24)
cos_hour = cos(2π × hour / 24)
sin_day  = sin(2π × day_of_week / 7)
cos_day  = cos(2π × day_of_week / 7)
sin_month = sin(2π × month / 12)
cos_month = cos(2π × month / 12)
```

**Binary flags**: `is_peak_hour` (9–18h), `is_weekend`

**VMD modes** (for LSTM branch): `IMF_1` through `IMF_8`

---

## Dataset

**File**: `hourlyLoadDataIndia.xlsx`

**Columns**:
- `datetime` — hourly timestamp
- `National Hourly Demand` — target variable (MW)
- `Northen Region Hourly Demand`
- `Western Region Hourly Demand`
- `Eastern Region Hourly Demand`
- `Southern Region Hourly Demand`
- `North-Eastern Region Hourly Demand`

The dataset is not included in this repository. Place the file at `/content/hourlyLoadDataIndia.xlsx` when running on Google Colab, or update the path in the script.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scipy
statsmodels
scikit-learn
xgboost
tensorflow
vmdpy
joblib
```

Install in one shot:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn xgboost tensorflow vmdpy joblib
```

On Google Colab, `vmdpy` needs an explicit install cell:

```python
!pip install vmdpy --quiet
```

---

## Usage

The script is structured as a linear Colab notebook exported to `.py`. Run it top-to-bottom after mounting Google Drive and placing the dataset:

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

Then execute the full script. All intermediate outputs are saved to `/content/` (or the local working directory if running locally).

To predict the next hour using the trained best ensemble model:

```python
next_hour_demand = predict_next_hour(
    last_200_hours,   # DataFrame with >= 168 rows of history
    best_ensemble_model,
    scaler,
    feature_cols
)
```

---

## Outputs

| File | Description |
|---|---|
| `01_timeseries_analysis.png` | Full / weekly / daily load time series |
| `02_distribution_analysis.png` | Histogram, box plot, Q-Q, KDE |
| `03_hourly_daily_patterns.png` | Hourly/daily/monthly averages and heatmap |
| `04_acf_trend_analysis.png` | ACF and Savitzky-Golay trend decomposition |
| `05_outliers_analysis.png` | Z-score and IQR outlier visualization |
| `06_regional_comparison.png` | Regional demand time series |
| `ensemble_comparison_1hr_ahead.png` | Predicted vs actual for all ensemble models |
| `residual_analysis_1hr_ahead.png` | Residual scatter, histogram, time plot, Q-Q |
| `model_comparison_1hr_ahead.csv` | MAE / RMSE / MAPE / R² for all models |
| `all_predictions_1hr_ahead.csv` | Actual vs predicted for all models on test set |
| `ensemble_summary_1hr_ahead.txt` | Text summary report with best model details |
| `*.pkl` | Saved models, scaler, and ensemble weights |

---

## Key Design Decisions

**No data leakage**: The train/val/test split happens before scaling. `MinMaxScaler` is fit only on training data and applied to val/test. Regional demand columns are excluded from features to avoid indirect leakage.

**Chronological split**: 70% train / 20% validation / 10% test, in time order. No shuffling.

**VMD on full signal, indexed at split**: VMD requires a contiguous signal to produce coherent modes. The full series is decomposed once, and the resulting IMFs are indexed to assign train/test portions.

**Sequence length = 168**: One week of hourly history is used as the LSTM lookback window, capturing both daily and weekly periodicity.

**Inverse-MAPE ensemble weighting**: Models that perform better on the validation set receive proportionally higher weight in the final ensemble, reducing overfitting to any single model's biases.
