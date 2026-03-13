# ONS Retail Sales Index Forecasting with LSTM and SARIMA

**Author:** Keerthana  

This project implements an end‑to‑end forecasting pipeline for the UK Office for National Statistics (ONS) Retail Sales Index, comparing a tuned LSTM deep learning model against a SARIMA time‑series model on the same cleaned dataset. The workflow is designed to be fully reproducible and suitable for academic dissertation work and practical retail analytics.

---

## 1. Project Overview

- **Objective**: Forecast monthly percentage changes in the UK Retail Sales Index for “All retailing excluding automotive fuel – chained volume, seasonally adjusted”.  
- **Primary model**: Stacked LSTM (64–32 units with dropout) trained on a feature‑engineered univariate time series.  
- **Baseline model**: SARIMA with systematic hyperparameter search and full evaluation.  
- **Key deliverables**:  
  - Cleaned ONS monthly time series (2015–2025).  
  - Feature matrix with lags, rolling statistics, and seasonal encodings.  
  - Metrics and visualisations comparing LSTM, SARIMA and a simple ensemble.  

---

## 2. Data

### 2.1 Source

- **Provider**: Office for National Statistics (ONS), UK Government.  
- **Dataset**: Retail Sales Index Time Series – Version 44.  
- **Nature**: Official government statistics (not AI‑generated), widely used in economic and business research.

### 2.2 Files

- Excel file:  
  - `retail-sales-index-time-series-v44-filtered-2026-02-23T12-44-00Z.xlsx`  
  - Sheet: `Dataset`.  
- Notebook:  
  - `keerthana ONS Retail project.ipynb` (or similar).

### 2.3 Target Series

- Geography: **Great Britain** (`K03000001`).  
- Indicator: **All retailing excluding automotive fuel**.  
- Measure: **Chained volume – Percentage change, Seasonally Adjusted**.  
- Time span used: **Jan 2015 – Dec 2025** (132 monthly observations).

---

## 3. Environment & Dependencies

The notebook runs in Python 3 (Google Colab or local Jupyter).

**Core libraries**:

- Data & plotting: `pandas`, `numpy`, `matplotlib`, `seaborn`  
- ML & preprocessing: `scikit-learn` (metrics, scaling, splits)  
- Time series: `statsmodels` (SARIMAX, ACF/PACF)  
- Deep learning: `tensorflow` / `keras` (LSTM, BiLSTM, callbacks)  
- Misc: `warnings`  

All imports are grouped in the first code cell.

---

## 4. Methodology

### 4.1 Data Loading & Inspection

1. Load the Excel sheet `Dataset` using `pd.read_excel(..., header=None)` into `df_raw`.  
2. Print shape and first rows/columns to understand the mixed metadata layout.  
3. Promote row index 2 to the header row, then reset the DataFrame so each subsequent row corresponds to one series.

Result: A structured table with metadata columns plus monthly columns (`Jan-15`, `Feb-15`, …, `Dec-25`, with matching “Data Marking” fields).

### 4.2 Target Series Extraction

1. Identify **date columns** by matching month names and the hyphen, excluding all “Data Marking” columns.  
2. Search the first ~20 rows to find the one containing “All retailing excluding automotive fuel” with the correct price/measure description.  
3. Extract the numeric values from this row over all valid date columns.  
4. Parse the header strings (`Jan-15` etc.) into a `Date` column using `pd.to_datetime`.  
5. Build `df_clean = [Date, Value]`, drop missing values, and confirm date and value ranges.

Result: Clean monthly time series (132 points) ready for feature engineering.

---

### 4.3 Feature Engineering

On `df_clean`:

- Set `Date` as index and create the following features:

**Lag features**  
- `lag1`, `lag3`, `lag6`, `lag12`  

**Difference features**  
- `diff1`, `diff3`, `diff12`  

**Rolling statistics**  
- `rolling_mean_3`, `rolling_std_3`  
- `rolling_mean_12`, `rolling_std_12`  

**Seasonal / calendar features**  
- `month`, `quarter`  
- `month_sin`, `month_cos` (cyclical month encoding).

After dropping rows with insufficient history, the final dataset has shape `120 × 17` (Date, Value, 15 features).

---

### 4.4 Correlation Analysis

- Compute a full correlation matrix over all 17 columns.  
- Plot a comprehensive heatmap (`01FIXED_correlation_heatmap_all_features.png`) including **all engineered features**.  

Highlights:

- Strong correlation between `Value` and short lags (`lag1`) and `rolling_mean_3`, indicating strong short‑term persistence.

---

### 4.5 LSTM Forecasting Model

#### 4.5.1 Data Preparation

- Scale `Value` using `MinMaxScaler`.  
- Use a 12‑month sliding window to create supervised sequences:  
  - Input: 12 time steps of all features.  
  - Output: target value at the next month.  
- Split into **Train / Validation / Test** preserving time order:  
  - Train: 69 samples  
  - Validation: 17 samples  
  - Test: 22 samples.

#### 4.5.2 Model Architecture

Keras Sequential model:

- `LSTM(64, return_sequences=True)`  
- `Dropout(0.2)`  
- `LSTM(32)`  
- `Dense(16, activation='relu')`  
- `Dense(1)`  

Compile with:

- Optimizer: Adam (lr=0.001)  
- Loss: MSE  
- Metric: MAE  

Total trainable parameters: ~29,857.

#### 4.5.3 Training & Diagnostics

- Epochs: up to 100  
- Batch size: 8  
- EarlyStopping on `val_loss` with `patience=15`, `restore_best_weights=True`.  
- Training/validation **learning curves** (loss and MAE) saved as `02FIXED_lstm_learningcurves.png`.

Learning curves show smooth convergence and reasonable generalisation.

#### 4.5.4 LSTM Evaluation

After inverse scaling of predictions:

- MAE ≈ 4.40  
- RMSE ≈ 4.50  
- MAPE ≈ 418.30  
- R² ≈ −10.31  

(Exact values printed under “LSTM TEST RESULTS”.)

---

### 4.6 SARIMA Baseline

#### 4.6.1 Data & Split

- Use unscaled `Value` series with `Date` index.  
- Split into:  
  - Train: first 84 points  
  - Validation: next 21 points  
  - Test: final 27 points.

#### 4.6.2 Hyperparameter Tuning

Search grid:

- Non‑seasonal: `p ∈ {0,1,2}`, `d = 1`, `q ∈ {0,1,2}`  
- Seasonal: `P ∈ {0,1}`, `D = 1`, `Q ∈ {0,1}`, `s = 12`  

For each combination:

- Fit SARIMAX on `train_split_ts`.  
- Forecast validation segment.  
- Compute validation RMSE and track best AIC & RMSE.

Best model: **SARIMA(1,1,2) × (1,1,1,12)**.

#### 4.6.3 Final Evaluation

Trained on full train series and evaluated on test portion:

- MAE ≈ 5.60  
- RMSE ≈ 6.26  
- MAPE ≈ 546.65  
- R² ≈ −14.96  

LSTM clearly outperforms SARIMA on this dataset.

#### 4.6.4 Direction Classification

- Convert test set to **Up/Down** labels based on sign of month‑to‑month change for both actual and predicted values.  
- Compute accuracy, confusion matrix, and classification report.  
- Direction accuracy ≈ 42.31%; detailed precision/recall per class reported.

An ROC–AUC calculation is attempted using residuals; if not feasible, a clear message is printed.

---

### 4.7 Ensemble and Final Ranking

- Simple ensemble:  
  - `Ensemble = 0.4 * LSTM + 0.4 * BiLSTM + 0.2 * SARIMA` (when BiLSTM is available).  
- Compute MAE, RMSE, R² for all models (LSTM, BiLSTM, SARIMA, Ensemble).  
- Define an **OverallScore** combining normalised MAE, RMSE and R², then rank models by score.

Result: LSTM and the ensemble rank best; SARIMA is clearly weaker.

Visual comparison is saved as `08_FINAL_MODEL_comparison_all.png` with bar plots for MAE, RMSE, R² and Overall Score.

---

## 5. Outputs

Key artefacts produced by the notebook:

- Clean series: `df_clean` and feature‑engineered `df_features` (in‑memory).  
- Plots:  
  - `01FIXED_correlation_heatmap_all_features.png`  
  - `02FIXED_lstm_learningcurves.png`  
  - `04_PREDICTIONS_comparison.png` (test actual vs LSTM vs SARIMA)  
  - `08_FINAL_MODEL_comparison_all.png` (all models).  
- Printed evaluation tables comparing metrics across models.

---

## 6. Running the Notebook

1. Place these files together:  
   - `keerthana ONS Retail project.ipynb`  
   - `retail-sales-index-time-series-v44-filtered-2026-02-23T12-44-00Z.xlsx`.  

2. Open the notebook in **Google Colab** or **Jupyter Notebook** (Python 3).  

3. Run all cells sequentially from top to bottom:  
   - STEP 1: Data loading and inspection  
   - STEP 2–3: Target extraction and feature engineering  
   - STEP 4: Correlation analysis  
   - STEP 5: LSTM modelling and evaluation  
   - STEP 6: SARIMA modelling and evaluation  
   - STEP 7–9: Model comparison, ranking, and visualisation.  

4. Check the working directory for generated plots and any saved documentation.

---

## 7. Future Work

Potential extensions:

- Build a dedicated LSTM/BiLSTM classifier for Up/Down direction with ROC–AUC and confusion matrices.  
- Extend from univariate to multivariate forecasting by adding other ONS retail series as exogenous features.  
- Use SHAP or permutation importance for feature interpretability.  
- Explore more advanced ensembles (e.g. LSTM + XGBoost + SARIMA) under the same evaluation framework.
