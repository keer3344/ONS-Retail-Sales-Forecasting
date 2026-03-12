# ONS Retail Sales Forecasting (UK)

This project analyses and forecasts the **UK ONS Retail Sales Index** using
both classical time-series models and machine learning models.[web:50][file:64]

## 1. Dataset

- Source: **Office for National Statistics (ONS), UK Government** – Retail Sales Index time-series.[web:50]
- Filtered dataset: `retail-sales-index-time-series-v44-filtered-2026-02-23T12-44-00Z.xlsx`.[file:64]
- Frequency: Monthly.
- Main target series: **“All retailing excluding automotive fuel”** (seasonally adjusted, chained volume or % change – as used in the notebook).[file:64][file:67]

I only use **original government data**, no synthetic or AI-generated data.

## 2. Project goals

1. Clean the wide ONS Excel into an analysis-ready time series.
2. Explore trends, seasonality and correlations between key retail series.
3. Compare a **SARIMA** model and a **Random Forest regression with lag features**
   for forecasting the retail index.
4. Run a small **classification experiment** (up/down movement) to provide
   ROC–AUC and ROC curves, as requested in supervision meetings.[file:67]

## 3. Methods

### 3.1 Data preprocessing

- Use pandas to:
  - Convert the ONS “Dataset” sheet into tidy format.
  - Keep only Great Britain, seasonally adjusted series.
  - Focus on a small set of important categories and pivot to
    `Date × SeriesName`.[file:64][file:67]
- Handle missing values and ensure a continuous monthly index.

### 3.2 SARIMA model

- Perform grid search over `(p, d, q)` and `(P, D, Q, 12)` using **AIC**
  to select the best SARIMA specification.[file:67][web:71]
- Train on all data except the last 24 months.
- Forecast the final 24 months and evaluate with:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error).[file:67][web:70]

### 3.3 Random Forest regression (ML model)

- Engineer features from the target series:
  - 12 lag features (values at `t-1 … t-12`),
  - Rolling means and rolling standard deviation.[file:67][web:78]
- Use `TimeSeriesSplit` and `GridSearchCV` to tune:
  - `n_estimators`, `max_depth`, `min_samples_split`.
- Evaluate on the same 24‑month test horizon with MAE and RMSE.[file:67][web:70]

### 3.4 Classification + ROC (auxiliary experiment)

- Create a binary label **`increase_next_month`**:
  - 1 if the index increases more than a small threshold next month,
  - 0 if it decreases more than the threshold.
- Train:
  - `LogisticRegression` and
  - `RandomForestClassifier` on lag/rolling features.[file:67][web:56]
- Report:
  - Accuracy, Precision, Recall, F1,
  - ROC–AUC and ROC curves.
- This experiment is clearly **secondary**; the core evaluation is still
  based on regression metrics.

## 4. Results (summary)

- Both SARIMA and Random Forest capture the long‑term trend and seasonality
  of the retail index.
- I compare MAE and RMSE in a small results table and discuss where each
  model performs better (e.g. SARIMA for smooth seasonal pattern, RF for
  local non-linear changes).[file:67][web:70][web:80]
- The up/down classification task shows only modest ROC–AUC, which is
  expected given the noisy month‑to‑month changes in a macroeconomic index.

## 5. How to run

1. Clone this repository.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
