# ONS-Retail-Sales-Forecasting
Time series forecasting analysis of ONS Retail Sales Index using ARIMA, SARIMA, ETS, VAR, and LSTM models. Includes parameter tuning, diagnostics, and comprehensive model comparison.


## Project Overview

This project implements a comprehensive time series forecasting analysis on the ONS Retail Sales Index dataset. It compares five different time series models: ARIMA, SARIMA, ETS, VAR, and LSTM to determine which provides the best forecasting accuracy and reliability.

## Dataset

The data is sourced from the Office for National Statistics (ONS) Retail Sales Index. The dataset contains monthly retail sales data that is preprocessed and normalized for model training.

## Models Implemented

### 1. ARIMA (AutoRegressive Integrated Moving Average)
ARIMA is a statistical model that combines autoregression, differencing, and moving average components. It's effective for univariate time series with clear temporal patterns.
- **Parameters**: p (AR), d (I), q (MA)
- **Use Case**: Traditional stationary and non-stationary time series

### 2. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
SARIMA extends ARIMA by incorporating seasonal components, making it ideal for data with clear seasonal patterns.
- **Parameters**: (p, d, q) × (P, D, Q, s)
- **Use Case**: Data with seasonal fluctuations (e.g., retail sales)

### 3. ETS (Error, Trend, Seasonal)
ETS decomposes time series into error, trend, and seasonal components using exponential smoothing methods.
- **Use Case**: Flexible modeling of trend and seasonal variations

### 4. VAR (Vector AutoRegression)
VAR models multivariate relationships in time series data, capturing dynamic dependencies between multiple variables.
- **Use Case**: When you have multiple related time series

### 5. LSTM (Long Short-Term Memory)
LSTM is a deep learning model that can capture long-term dependencies in sequential data through its memory cells.
- **Use Case**: Complex nonlinear patterns and large datasets

## Project Features

- **Model Implementation**: Human-written implementations of all five models
- **Parameter Tuning**: Automated hyperparameter optimization for each model
- **Diagnostics**: Comprehensive residual analysis and model diagnostics
- **Comparison Framework**: Standardized evaluation metrics across all models
- **Visualization**: Plots for forecasts, residuals, and model comparisons
- **Documentation**: Detailed comments explaining all steps

## Dependencies

```
python >= 3.8
pandas
numpy
scikit-learn
statsmodels
tensorflow / keras
matplotlib
seaborn
pytest
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main notebook or script to execute all models
4. Review outputs, visualizations, and comparison table

## Model Performance

Each model is evaluated using:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)

A comprehensive comparison table summarizes performance across all metrics.

## Author

Developed for academic research in time series forecasting and statistical modeling.

## License

MIT License
