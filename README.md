# ONS Retail Sales Forecasting

This repository contains the code and notebooks for my MSc Data Science final project at the University of Hertfordshire: **“Forecasting UK Retail Sales Using LSTM, SARIMAX and Exogenous Macroeconomic Indicators.”**

The project builds an end‑to‑end forecasting pipeline for the UK **Retail Sales Index** published by the Office for National Statistics (ONS), comparing a compact LSTM model with classical SARIMAX baselines and simple ensembles for one‑step and multi‑horizon (3, 6, 12 months ahead) forecasts.

## Repository structure

- `notebooks/`
  - `ONS_retail_LSTM_baseline.ipynb`  
    One‑step LSTM baseline: feature engineering, leakage‑aware scaling, sequence construction, model training with early stopping, and evaluation on train/validation/test splits.
  - `ONS_retail_multi_horizon_LSTM_SARIMAX_ensemble.ipynb`  
    Extended pipeline with exogenous GDP/CPI features, direct multi‑horizon targets (t+3, t+6, t+12), SARIMAX with exogenous variables, LSTM hyperparameter search, and LSTM–SARIMAX ensembles.
  

- `data/`
  - Scripts or notes for downloading and preparing the **Retail Sales Index** time‑series dataset (version 44) and monthly UK GDP/CPI series from the Office for National Statistics (ONS).  
    > Raw ONS Excel files are not committed to this repo; they should be downloaded separately from the ONS website.

- `figures/`
  - Key plots used in the report, including learning curves, actual vs predicted plots, metric comparison charts, multi‑horizon forecast comparisons, and correlation heatmaps.

## Data

The main target series is:

- **All retailing excluding automotive fuel – value of retail sales at current prices, seasonally adjusted, Great Britain**, from the ONS **Retail Sales Index** time‑series dataset (version 44).

Monthly macroeconomic indicators used as exogenous variables:

- UK monthly **GDP** index.
- UK monthly **CPI** (inflation) index.

All data are official UK government statistics obtained from the Office for National Statistics.

## Methods

The pipeline is implemented in Python using `pandas`, `numpy`, `scikit-learn`, `statsmodels` and `tensorflow/keras` and follows these steps:

1. **Data cleaning and feature engineering**  
   - Construct a tidy monthly `Date–Value` series (2010‑01 to 2024‑12).  
   - Engineer time‑series features: lags, rolling means/standard deviations, differences, seasonal calendar features, and percentage changes.  
   - Add exogenous features: GDP, CPI, and their lags and differences.

2. **Leakage‑aware splitting and scaling**  
   - Chronological train/validation/test split (70/15/15) applied after feature creation.  
   - `MinMaxScaler` fitted on the training period only and applied to validation/test to avoid leakage.

3. **Sequence construction and LSTM models**  
   - Convert tabular features into fixed‑length sequences (typically 12 months) for sequence‑to‑one LSTM forecasting.  
   - Compact LSTM architectures with dropout and early stopping; small hyperparameter grids for units, dropout, learning rate, and lookback window.

4. **SARIMAX and ensembles**  
   - SARIMAX models with exogenous GDP/CPI features for each horizon.  
   - Simple convex‑weight ensembles combining LSTM and SARIMAX based on validation RMSE.

5. **Evaluation**  
   - Metrics: MAE, RMSE, R² (and MAPE in multi‑horizon experiments).  
   - Visual diagnostics: learning curves, residual plots, and actual vs predicted plots for all sets and horizons.

## How to run

1. Create a Python environment (e.g. `conda` or `venv`) and install the main dependencies:

   ```bash
   pip install numpy pandas scikit-learn statsmodels tensorflow matplotlib seaborn
   ```

2. Download the ONS Retail Sales Index and macroeconomic series (GDP and CPI) from the ONS website and place them in the `data/` folder using the filenames expected in the notebooks.

3. Open the notebooks in Jupyter/Colab:

   - Run `ONS_retail_LSTM_baseline.ipynb` to reproduce the one‑step LSTM results.  
   - Run `ONS_retail_multi_horizon_LSTM_SARIMAX_ensemble.ipynb` to reproduce the multi‑horizon and ensemble experiments.

## Project context

This repository accompanies the report:

> Keerthana Koluguri (2026), **“Forecasting UK Retail Sales Using LSTM, SARIMAX and Exogenous Macroeconomic Indicators”**, MSc Data Science, University of Hertfordshire.

The report describes the methodology, experimental design, results, limitations and future work in detail. This repo focuses on the reproducible code and main figures used in that report.

## License

This code is provided for academic and learning purposes. Official ONS datasets remain subject to the terms specified by the Office for National Statistics.
