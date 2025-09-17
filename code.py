import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data(file_path: str) -> pl.DataFrame:
    #Load e-commerce dataset from a CSV file.
    try:
        return pl.read_csv(file_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e

def inspect_data(df: pl.DataFrame) -> dict:
    #Inspect the DataFrame and return key information.
    return {
        "head": df.head(),
        "columns": df.columns,
        "size": df.estimated_size(),
        "description": df.describe(),
        "missing_values": df.null_count()
    }

def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    #Remove duplicates from the DataFrame.
    return df.unique()

def compute_decision_time_by_income(df: pl.DataFrame) -> pl.DataFrame:
    #Group by Income_Level and compute median Time_to_Decision.
    return df.group_by("Income_Level").agg(
        pl.col("Time_to_Decision").median().alias("Decision_Time_by_Income")
    )

def reformat_purchase_amount(df: pl.DataFrame) -> pl.DataFrame:
    #Convert Purchase_Amount from string to float.
    return df.with_columns(
        pl.col("Purchase_Amount")
        .str.strip_chars()
        .str.replace_all(r"[\$,]", "")
        .cast(pl.Float64)
        .alias("price_float")
    )

def compute_amount_by_education(df: pl.DataFrame) -> pl.DataFrame:
    #Group by Education_Level and compute mean Purchase_Amount.
    return df.group_by("Education_Level").agg(
        pl.col("price_float").mean().round(2).alias("Amount_by_Education")
    )

def prepare_ml_data(df: pl.DataFrame, target_col: str = "Return_Rate") -> tuple:
    #Prepare numerical features and target for ML.
    numerical_df = df.select(pl.col(pl.Float64, pl.Int64))
    y = numerical_df.select(target_col).to_numpy().ravel()
    X = numerical_df.drop(target_col)
    return X, y

def train_model(X: pl.DataFrame, y: np.ndarray, **kwargs) -> XGBRegressor:
    #Train an XGBoost regressor model.
    model = XGBRegressor(**kwargs)
    model.fit(X, y)
    return model

def evaluate_model(model: XGBRegressor, X_test: pl.DataFrame, y_test: np.ndarray) -> dict:
    #Evaluate the model and return performance metrics.
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return {"mse": mse, "rmse": rmse, "r2": r2}

def plot_feature_importance(model: XGBRegressor, features: list, save_path: str = None):
    #Plot feature importance for the trained model.
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(features, importances)
    plt.xlabel("Importance Score")
    plt.title("Feature Importance in Predicting Return Rate")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main(file_path: str):
    #Main workflow for data processing and machine learning
    # Load and inspect data
    ecommerce = load_data(file_path)
    inspection = inspect_data(ecommerce)
    print("E-Commerce Customer Behavior Data:")
    print(inspection["head"])
    print("Columns include:", inspection["columns"])
    print("Size:", inspection["size"])
    print("Description:", inspection["description"])
    print("Missing values:\n", inspection["missing_values"])

    # Clean data
    ecommerce = clean_data(ecommerce)
    print("Cleaned shape:", ecommerce.shape)

    # Compute groupings
    grouping1 = compute_decision_time_by_income(ecommerce)
    print(grouping1)

    ecommerce = reformat_purchase_amount(ecommerce)
    grouping2 = compute_amount_by_education(ecommerce)
    print(grouping2)

    # ML pipeline
    X, y = prepare_ml_data(ecommerce)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train, n_estimators=200, learning_rate=0.1, random_state=42)
    metrics = evaluate_model(model, X_test, y_test)
    print("MSE:", metrics["mse"])
    print("RMSE:", metrics["rmse"])
    print("R²:", metrics["r2"])

    # Plot feature importance
    plot_feature_importance(model, X.columns)

if __name__ == "__main__":
    file_path = "/workspaces/IDS706-DE-Wk3/Ecommerce_Consumer_Behavior_Analysis_Data.csv"
    main(file_path)