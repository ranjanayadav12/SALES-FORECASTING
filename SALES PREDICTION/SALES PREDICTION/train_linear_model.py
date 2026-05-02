import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import os

def train_linear_regression():
    # Load data
    train_path = "training_data.csv"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found")
        return

    df = pd.read_csv(train_path)
    
    # Features: rate, sales_in_first_month, sales_in_second_month
    X = df[['rate', 'sales_in_first_month', 'sales_in_second_month']].copy()
    X['sales_in_first_month'] = X['sales_in_first_month'] / 100000.0
    X['sales_in_second_month'] = X['sales_in_second_month'] / 100000.0
    y = df['sales_in_third_month'] / 100000.0
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save model
    with open("linear_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # print coefficients for verification
    print("Linear Regression Model Trained Successfully!")
    print(f"Intercept (b0): {model.intercept_}")
    print(f"Coefficients (b1, b2, b3): {model.coef_}")
    print(f"Formula: Sales = {model.intercept_:.4f} + ({model.coef_[0]:.4f} * rate) + ({model.coef_[1]:.4f} * sales1) + ({model.coef_[2]:.4f} * sales2)")

if __name__ == "__main__":
    train_linear_regression()
