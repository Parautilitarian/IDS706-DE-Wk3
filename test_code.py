# test_code.py
import unittest
import polars as pl
import numpy as np
from unittest.mock import patch, mock_open
from code import (
    load_data, inspect_data, clean_data, compute_decision_time_by_income,
    reformat_purchase_amount, compute_amount_by_education, prepare_ml_data,
    train_model, evaluate_model
)
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# unit tests
class TestCode(unittest.TestCase):
    def setUp(self):
        self.test_csv = (
            "Customer_ID,Income_Level,Education_Level,Purchase_Amount,Time_to_Decision,Return_Rate,Feature1\n"
            "1,High,Graduate,$100.00,5,0.1,10\n"
            "2,Low,Undergraduate,$50.00,3,0.2,20\n"
            "3,Medium,Graduate,$200.00,7,0.15,15\n"
            "4,High,Postgraduate,$150.00,6,0.3,25\n"
        )
        self.df = pl.DataFrame({
            "Customer_ID": [1, 2, 3, 4],
            "Income_Level": ["High", "Low", "Medium", "High"],
            "Education_Level": ["Graduate", "Undergraduate", "Graduate", "Postgraduate"],
            "Purchase_Amount": ["$100.00", "$50.00", "$200.00", "$150.00"],
            "Time_to_Decision": [5, 3, 7, 6],
            "Return_Rate": [0.1, 0.2, 0.15, 0.3],
            "Feature1": [10, 20, 15, 25]
        })

    def test_load_data(self):
        with patch("builtins.open", mock_open(read_data=self.test_csv)):
            df = load_data("test_data.csv")
            self.assertIsInstance(df, pl.DataFrame)
            self.assertEqual(len(df), 4)
            self.assertEqual(list(df.columns), [
                "Customer_ID", "Income_Level", "Education_Level", "Purchase_Amount",
                "Time_to_Decision", "Return_Rate", "Feature1"
            ])

    def test_load_data_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent.csv")

    def test_inspect_data(self):
        inspection = inspect_data(self.df)
        self.assertEqual(len(inspection["head"]), 4)
        self.assertEqual(inspection["columns"], self.df.columns)
        self.assertEqual(inspection["missing_values"].sum_horizontal().item(), 0)

    def test_clean_data(self):
        df_with_duplicates = self.df.vstack(self.df.head(1))
        cleaned_df = clean_data(df_with_duplicates)
        self.assertEqual(len(cleaned_df), 4)

    def test_compute_decision_time_by_income(self):
        result = compute_decision_time_by_income(self.df)
        self.assertEqual(len(result), 3)  # 3 income levels
        self.assertTrue("Decision_Time_by_Income" in result.columns)

    def test_reformat_purchase_amount(self):
        df = reformat_purchase_amount(self.df)
        self.assertTrue("price_float" in df.columns)
        self.assertEqual(df["price_float"].dtype, pl.Float64)
        self.assertEqual(df["price_float"][0], 100.0)

    def test_compute_amount_by_education(self):
        df = reformat_purchase_amount(self.df)
        result = compute_amount_by_education(df)
        self.assertEqual(len(result), 3)  # 3 education levels
        self.assertTrue("Amount_by_Education" in result.columns)

    def test_prepare_ml_data(self):
        X, y = prepare_ml_data(self.df)
        self.assertIsInstance(X, pl.DataFrame)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[1], 3)  # Customer_ID, Time_to_Decision, Feature1
        self.assertEqual(len(y), 4)
        self.assertFalse("Return_Rate" in X.columns)

    def test_train_model(self):
        X, y = prepare_ml_data(self.df)
        model = train_model(X, y, n_estimators=100, random_state=42)
        self.assertIsInstance(model, XGBRegressor)
        self.assertTrue(hasattr(model, "feature_importances_"))

    def test_evaluate_model(self):
        X, y = prepare_ml_data(self.df)
        model = train_model(X, y, n_estimators=100, random_state=42)
        metrics = evaluate_model(model, X, y)
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)

if __name__ == "__main__":
    unittest.main()


# system test
def test_workflow(mocker, test_csv):
    mocker.patch("builtins.open", mocker.mock_open(read_data=test_csv))
    df = load_data("test_data.csv")
    df = clean_data(df)
    df = reformat_purchase_amount(df)
    X, y = prepare_ml_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train, n_estimators=100, random_state=42)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["r2"] >= -1.0
    assert metrics["mse"] > 0.0