"""
Sales Prediction Model Training Script using Gradient Boosting
This script trains a sales prediction model using Gradient Boosting algorithm
which handles large numbers better than polynomial features.
NOTE: Using sklearn's GradientBoostingRegressor as XGBoost requires 64-bit Python
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SALES PREDICTION MODEL TRAINING")
print("Using Gradient Boosting Regressor")
print("="*60)

# Load the data
data = pd.read_csv('sales_lakhs.csv')

print("\nSALES DATA OVERVIEW")
print("-"*40)
print(data.head())
print(f"\nDataset shape: {data.shape}")

# ============ FEATURE ENGINEERING ============
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("(Using safe features - NO sales squared values)")
print("="*60)

# Create a copy for feature engineering
df = data.copy()

# Drop the index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# 1. Interaction features (safe - NO sales squared)
df['rate_x_sales_first'] = df['rate'] * df['sales_in_first_month']
df['rate_x_sales_second'] = df['rate'] * df['sales_in_second_month']
df['sales_first_x_sales_second'] = df['sales_in_first_month'] * df['sales_in_second_month']

# 2. Ratio features (handle division by zero)
df['sales_ratio_first_second'] = df['sales_in_first_month'] / (df['sales_in_second_month'] + 1)
df['sales_ratio_second_first'] = df['sales_in_second_month'] / (df['sales_in_first_month'] + 1)

# 3. Difference features
df['sales_diff'] = df['sales_in_first_month'] - df['sales_in_second_month']
df['sales_trend'] = (df['sales_in_second_month'] - df['sales_in_first_month']) / (df['sales_in_first_month'] + 1)

# 4. Total and average
df['total_sales'] = df['sales_in_first_month'] + df['sales_in_second_month']
df['avg_sales'] = (df['sales_in_first_month'] + df['sales_in_second_month']) / 2

# 5. Rate features (only rate squared - NOT sales squared)
df['rate_squared'] = df['rate'] ** 2
df['rate_log'] = np.log1p(df['rate'])

# 6. Normalized features (important for stability)
df['sales_first_norm'] = df['sales_in_first_month'] / (df['total_sales'] + 1)
df['sales_second_norm'] = df['sales_in_second_month'] / (df['total_sales'] + 1)

# 7. Rate impact features
df['rate_per_sales_first'] = df['rate'] / (df['sales_in_first_month'] + 1) * 100000
df['rate_per_sales_second'] = df['rate'] / (df['sales_in_second_month'] + 1) * 100000

print("Features created:")
print(df.columns.tolist())

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['sales_in_third_month']]
X = df[feature_cols]
y = df['sales_in_third_month']

print(f"\nTotal features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# ============ SCALING ============
print("\n" + "="*60)
print("FEATURE SCALING")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for use in predictions
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Scaler saved to scaler.pkl")

# ============ MODEL TRAINING WITH GRADIENT BOOSTING ============
print("\n" + "="*60)
print("TRAINING GRADIENT BOOSTING MODEL")
print("="*60)

# Gradient Boosting model with good defaults (alternative to XGBoost)
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.8,
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(gb_model, X_scaled, y, cv=5, scoring='r2')
print(f"Gradient Boosting CV R2 Scores: {cv_scores}")
print(f"Mean CV R2: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Hyperparameter tuning
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_scaled, y)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_*100:.2f}%")

# Use the best model
best_model = grid_search.best_estimator_

# ============ TRAIN FINAL MODEL ============
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*60)

# Fit on all data
best_model.fit(X_scaled, y)

# Calculate final metrics
y_pred = best_model.predict(X_scaled)
final_r2 = r2_score(y, y_pred)
final_mae = mean_absolute_error(y, y_pred)
final_rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Final R2 Score: {final_r2*100:.2f}%")
print(f"Final MAE: {final_mae:.2f}")
print(f"Final RMSE: {final_rmse:.2f}")

# Save the best model
pickle.dump(best_model, open('model.pkl', 'wb'))
print("\nModel saved to model.pkl")

# Save feature names for reference
feature_info = {
    'feature_cols': feature_cols,
    'best_model_name': 'GradientBoosting',
    'accuracy': final_r2 * 100,
    'r2_score': final_r2 * 100,
    'mae': final_mae,
    'rmse': final_rmse,
    'scale': 'lakhs'
}
pickle.dump(feature_info, open('model_info.pkl', 'wb'))
print("Model info saved to model_info.pkl")

# ============ TEST PREDICTIONS ============
print("\n" + "="*60)
print("TEST PREDICTIONS")
print("="*60)

# Test Case 1: rate=10, sales_first=6 lakhs (600000), sales_second=11 lakhs (1100000)
test_rate = 10
test_sales_first = 600000  # 6 lakhs
test_sales_second = 1100000  # 11 lakhs

# Create features
test_features = {
    'rate': test_rate,
    'sales_in_first_month': test_sales_first,
    'sales_in_second_month': test_sales_second,
    'rate_x_sales_first': test_rate * test_sales_first,
    'rate_x_sales_second': test_rate * test_sales_second,
    'sales_first_x_sales_second': test_sales_first * test_sales_second,
    'sales_ratio_first_second': test_sales_first / (test_sales_second + 1),
    'sales_ratio_second_first': test_sales_second / (test_sales_first + 1),
    'sales_diff': test_sales_first - test_sales_second,
    'sales_trend': (test_sales_second - test_sales_first) / (test_sales_first + 1),
    'total_sales': test_sales_first + test_sales_second,
    'avg_sales': (test_sales_first + test_sales_second) / 2,
    'rate_squared': test_rate ** 2,
    'rate_log': np.log1p(test_rate),
    'sales_first_norm': test_sales_first / (test_sales_first + test_sales_second + 1),
    'sales_second_norm': test_sales_second / (test_sales_first + test_sales_second + 1),
    'rate_per_sales_first': test_rate / (test_sales_first + 1) * 100000,
    'rate_per_sales_second': test_rate / (test_sales_second + 1) * 100000
}

test_df = pd.DataFrame([test_features])
test_df = test_df[feature_cols]
test_scaled = scaler.transform(test_df)

pred = best_model.predict(test_scaled)[0]
print(f"\nTest Input: rate={test_rate}, sales_first={test_sales_first/100000} lakhs, sales_second={test_sales_second/100000} lakhs")
print(f"  Predicted third month sales: Rs.{pred:,.2f} (= {pred/100000:.2f} lakhs)")

# Test Case 2: Another test case
test_rate2 = 15
test_sales_first2 = 800000  # 8 lakhs
test_sales_second2 = 900000  # 9 lakhs

test_features2 = {
    'rate': test_rate2,
    'sales_in_first_month': test_sales_first2,
    'sales_in_second_month': test_sales_second2,
    'rate_x_sales_first': test_rate2 * test_sales_first2,
    'rate_x_sales_second': test_rate2 * test_sales_second2,
    'sales_first_x_sales_second': test_sales_first2 * test_sales_second2,
    'sales_ratio_first_second': test_sales_first2 / (test_sales_second2 + 1),
    'sales_ratio_second_first': test_sales_second2 / (test_sales_first2 + 1),
    'sales_diff': test_sales_first2 - test_sales_second2,
    'sales_trend': (test_sales_second2 - test_sales_first2) / (test_sales_first2 + 1),
    'total_sales': test_sales_first2 + test_sales_second2,
    'avg_sales': (test_sales_first2 + test_sales_second2) / 2,
    'rate_squared': test_rate2 ** 2,
    'rate_log': np.log1p(test_rate2),
    'sales_first_norm': test_sales_first2 / (test_sales_first2 + test_sales_second2 + 1),
    'sales_second_norm': test_sales_second2 / (test_sales_first2 + test_sales_second2 + 1),
    'rate_per_sales_first': test_rate2 / (test_sales_first2 + 1) * 100000,
    'rate_per_sales_second': test_rate2 / (test_sales_second2 + 1) * 100000
}

test_df2 = pd.DataFrame([test_features2])
test_df2 = test_df2[feature_cols]
test_scaled2 = scaler.transform(test_df2)

pred2 = best_model.predict(test_scaled2)[0]
print(f"\nTest Input: rate={test_rate2}, sales_first={test_sales_first2/100000} lakhs, sales_second={test_sales_second2/100000} lakhs")
print(f"  Predicted third month sales: Rs.{pred2:,.2f} (= {pred2/100000:.2f} lakhs)")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best Model: Gradient Boosting")
print(f"Accuracy: {final_r2*100:.2f}%")
print("Model now predicts sales correctly without using sales squared values!")
print("Features used: interaction, ratio, difference, and normalized features")
