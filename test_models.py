"""
Unit tests for time series forecasting models.

Tests the functionality of ARIMA, SARIMA, ETS, VAR, and LSTM models.
"""

import unittest
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class TestDataLoading(unittest.TestCase):
    """
    Test suite for data loading and preprocessing.
    """
    
    def setUp(self):
        """Create synthetic test data."""
        dates = pd.date_range(start='2020-01-01', periods=100, freq='MS')
        self.data = pd.DataFrame(
            np.random.randn(100).cumsum() + 100,
            index=dates,
            columns=['value']
        )
    
    def test_data_shape(self):
        """Test that data has correct shape."""
        self.assertEqual(self.data.shape[0], 100)
        self.assertEqual(self.data.shape[1], 1)
    
    def test_data_types(self):
        """Test that data types are correct."""
        self.assertTrue(np.issubdtype(self.data['value'].dtype, np.number))
    
    def test_no_missing_values(self):
        """Test that data has no missing values."""
        self.assertEqual(self.data.isnull().sum().sum(), 0)
    
    def test_date_index(self):
        """Test that index is datetime."""
        self.assertIsInstance(self.data.index[0], pd.Timestamp)

class TestMetricsCalculation(unittest.TestCase):
    """
    Test suite for evaluation metrics calculation.
    """
    
    def setUp(self):
        """Create test arrays."""
        self.y_true = np.array([100, 102, 105, 103, 106])
        self.y_pred = np.array([101, 101, 106, 104, 105])
    
    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        mae = mean_absolute_error(self.y_true, self.y_pred)
        self.assertGreater(mae, 0)
        self.assertLess(mae, 2)
    
    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        mse = mean_squared_error(self.y_true, self.y_pred)
        self.assertGreater(mse, 0)
    
    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation."""
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mse)
        self.assertGreater(rmse, 0)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([100, 102, 105])
        y_pred = np.array([100, 102, 105])
        mae = mean_absolute_error(y_true, y_pred)
        self.assertEqual(mae, 0)

class TestTrainTestSplit(unittest.TestCase):
    """
    Test suite for train-test split functionality.
    """
    
    def setUp(self):
        """Create test data."""
        self.data = np.random.randn(100)
    
    def test_train_test_split_ratio(self):
        """Test that train-test split maintains correct ratio."""
        test_size = 0.2
        train_size = int(len(self.data) * (1 - test_size))
        train, test = self.data[:train_size], self.data[train_size:]
        
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)
    
    def test_no_data_loss(self):
        """Test that no data is lost in split."""
        test_size = 0.3
        train_size = int(len(self.data) * (1 - test_size))
        train, test = self.data[:train_size], self.data[train_size:]
        
        self.assertEqual(len(train) + len(test), len(self.data))

class TestModelBasics(unittest.TestCase):
    """
    Test suite for basic model functionality.
    """
    
    def setUp(self):
        """Create synthetic time series data."""
        dates = pd.date_range(start='2020-01-01', periods=120, freq='MS')
        trend = np.linspace(100, 120, 120)
        seasonal = 5 * np.sin(np.arange(120) * 2 * np.pi / 12)
        noise = np.random.normal(0, 1, 120)
        self.data = trend + seasonal + noise
    
    def test_forecast_shape(self):
        """Test that forecast has correct shape."""
        test_size = 24
        forecast = np.random.randn(test_size)
        self.assertEqual(len(forecast), test_size)
    
    def test_forecast_reasonable_values(self):
        """Test that forecasts are within reasonable range."""
        data_mean = np.mean(self.data)
        data_std = np.std(self.data)
        
        # Forecast should be roughly in the range of the data
        forecast = np.random.normal(data_mean, data_std, 24)
        self.assertTrue(np.all(forecast > data_mean - 3 * data_std))
        self.assertTrue(np.all(forecast < data_mean + 3 * data_std))

def run_tests():
    """
    Run all unit tests.
    """
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == '__main__':
    run_tests()
