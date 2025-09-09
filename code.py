import polars as pl
# pyright: ignore[reportShadowedImports]

# Load e-commerce consuumer behavior dataset (CSV)
# Source URL https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data?resource=download

ecommerce = pl.read_csv("/workspaces/IDS706-DE-Wk3/Ecommerce_Consumer_Behavior_Analysis_Data.csv")

# Data inspection
# print("E-Commerce Customer Behavir Data:")
# print(ecommerce.head())

# # Check column names
# print(ecommerce.columns)

# # print("Size:")
# print(ecommerce.estimated_size())

# # print("Description:")
# print(ecommerce.describe())

# # Check for missing values
# print("Missing values:\n", ecommerce.null_count())

# # Drop duplicates
# ecommerce = ecommerce.unique()
# print("Cleaned shape:", ecommerce.shape)

# # Find Time to Decision by Income Level
# grouping1 = ecommerce.group_by("Income_Level").agg(
#     pl.col("Time_to_Decision").median().alias("Decision_Time_by_Income")
# )
# print(grouping1)

# # Reformat the Income Amount data type from string values in dollar amount to float
# ecommerce = ecommerce.with_columns(
#     pl.col("Purchase_Amount")
#     .str.strip_chars()              # remove leading/trailing spaces
#     .str.replace_all(r"[\$,]", "")  # remove $ and ,
#     .cast(pl.Float64)               # convert to float
#     .alias("price_float")
# )

# # Find Purchase Amount by Education Level
# grouping2 = ecommerce.group_by("Education_Level").agg(
#     pl.col("price_float").mean().round(2).alias("Amount_by_Education")
# )
# print(grouping2)

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Keep only numerical columns
numerical_df = ecommerce.select([
    pl.col(pl.Float64, pl.Int64)
])

# Target column
y = numerical_df.select("Return_Rate")

# Features: all columns except the target
X = numerical_df.drop("Return_Rate")

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model & predictions
model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R²:", r2)

# Feature importance when predicting the return rate
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.title("Feature Importance in Predicting House Value")
plt.show()