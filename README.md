# IDS706-DE-Wk3
IDS706 basic dataset analysis

# E-Commerce Consumer Behavior Analysis

This project analyzes an **e-commerce consumer behavior dataset** using **Polars** for data processing and **XGBoost** for machine learning. It demonstrates data cleaning, feature engineering, exploratory analysis, and predictive modeling.

### Dataset source: [Kaggle - E-commerce Consumer Behavior Analysis]
(**URL**: https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data)

---

## Features

- Load and inspect the dataset with **Polars**
  - Import from csv
  - Understand data types and summary statistics
- Clean and preprocess data:
  - Remove duplicates
  - Handle missing values
  - Convert purchase amounts (`$`) into floats
- Exploratory analysis:
  - Median *Time to Decision* by **Income Level**
  - Average *Purchase Amount* by **Education Level**
- Train an **XGBoost Regressor** to predict **Return Rate**
- Evaluate performance with:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² score
- Plot feature importance to understand key drivers

---

## Setup

### Requirements

- Python 3.12
- [Polars](https://pola-rs.github.io/polars/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)

### Install dependencies:

```bash
pip install polars scikit-learn xgboost matplotlib
```

## Usage

### Place the dataset in your working directory: 
```bash 
Ecommerce_Consumer_Behavior_Analysis_Data.csv 
```

### Run the script:
```bash
python ecommerce_analysis.py
```

### Example outputs:
- Dataset summary & missing values
- Cleaned dataset shape
- Aggregations by Income Level and Education Level
- Model evaluation metrics (MSE, RMSE, R²)
- Feature importance bar chart


