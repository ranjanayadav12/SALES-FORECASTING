import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data in lakhs
data = pd.read_csv('sales_lakhs.csv')

print("="*60)
print("LAKHS DATA")
print("="*60)
print(data)
print(f"\nDataset shape: {data.shape}")

# ============ FEATURE ENGINEERING ============
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Create a copy for feature engineering
df = data.copy()

# Drop the index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# 1. Interaction features (critical for capturing relationships)
df['rate_x_sales_first'] = df['rate'] * df['sales_in_first_month']
df['rate_x_sales_second'] = df['rate'] * df['sales_in_second_month']
df['sales_first_x_sales_second'] = df['sales_in_first_month'] * df['sales_in_second_month']

# 2. Ratio features (handle division by zero)
df['sales_ratio_first_second'] = df['sales_in_first_month'] / (df['sales_in_second_month'] + 1)
df['sales_ratio_second_first'] = df['sales_in_second_month'] / (df['sales_in_first_month'] + 1)

# 3. Difference features
df['sales_diff'] = df['sales_in_first_month'] - df['sales_in_second_month']

# 4. Total and average
df['total_sales'] = df['sales_in_first_month'] + df['sales_in_second_month']
df['avg_sales'] = (df['sales_in_first_month'] + df['sales_in_second_month']) / 2

# 5. Rate features
df['rate_squared'] = df['rate'] ** 2

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

# ============ MODEL TRAINING WITH CROSS-VALIDATION ============
print("\n" + "="*60)
print("MODEL TRAINING WITH 5-FOLD CROSS-VALIDATION")
print("="*60)

# Define models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
}

# Evaluate with cross-validation
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_mean = scores.mean()
    cv_std = scores.std()
    cv_results[name] = {'cv_mean': cv_mean, 'cv_std': cv_std}
    print(f"{name}: CV R² = {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%)")

# Find best model
best_model_name = max(cv_results, key=lambda x: cv_results[x]['cv_mean'])
best_cv_score = cv_results[best_model_name]['cv_mean']
print(f"\nBest Single Model: {best_model_name} with CV R² = {best_cv_score*100:.2f}%")

# ============ HYPERPARAMETER TUNING ============
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# Tune Gradient Boosting
print("\nTuning Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
gb_grid.fit(X_scaled, y)
print(f"Best GB params: {gb_grid.best_params_}")
print(f"Best GB CV score: {gb_grid.best_score_*100:.2f}%")

# Tune Random Forest
print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
rf_grid.fit(X_scaled, y)
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF CV score: {rf_grid.best_score_*100:.2f}%")

# ============ ENSEMBLE MODEL ============
print("\n" + "="*60)
print("CREATING ENSEMBLE MODEL")
print("="*60)

# Create voting regressor with best models
ensemble = VotingRegressor([
    ('gb', gb_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_)
])

# Cross-validate ensemble
ensemble_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='r2')
ensemble_cv_mean = ensemble_scores.mean()
ensemble_cv_std = ensemble_scores.std()
print(f"Ensemble CV R² = {ensemble_cv_mean*100:.2f}% (+/- {ensemble_cv_std*100:.2f}%)")

# ============ SELECT BEST MODEL ============
print("\n" + "="*60)
print("SELECTING BEST MODEL")
print("="*60)

# Compare all best models
all_models = {
    'Tuned Gradient Boosting': (gb_grid.best_estimator_, gb_grid.best_score_),
    'Tuned Random Forest': (rf_grid.best_estimator_, rf_grid.best_score_),
    'Ensemble': (ensemble, ensemble_cv_mean)
}

# Find absolute best
best_name = max(all_models, key=lambda x: all_models[x][1])
best_model_final = all_models[best_name][0]
best_score_final = all_models[best_name][1]

print("\nFinal Model Comparison:")
for name, (model, score) in all_models.items():
    print(f"  {name}: R² = {score*100:.2f}%")

print(f"\n*** BEST MODEL: {best_name} ***")
print(f"*** ACCURACY: {best_score_final*100:.2f}% ***")

# ============ TRAIN FINAL MODEL ============
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*60)

# Fit on all data
best_model_final.fit(X_scaled, y)

# Calculate final metrics
y_pred = best_model_final.predict(X_scaled)
final_r2 = r2_score(y, y_pred)
final_mae = mean_absolute_error(y, y_pred)

print(f"Final R² Score: {final_r2*100:.2f}%")
print(f"Final MAE: {final_mae:.2f} (in lakhs)")

# Save the best model
pickle.dump(best_model_final, open('model.pkl', 'wb'))
print("\nModel saved to model.pkl")

# Save feature names for reference
feature_info = {
    'feature_cols': feature_cols,
    'best_model_name': best_name,
    'accuracy': best_score_final * 100,
    'r2_score': final_r2 * 100,
    'mae': final_mae,
    'scale': 'lakhs'
}
pickle.dump(feature_info, open('model_info.pkl', 'wb'))
print("Model info saved to model_info.pkl")

# ============ TEST PREDICTIONS ============
print("\n" + "="*60)
print("TEST PREDICTIONS (IN LAKHS)")
print("="*60)

# Test with values in lakhs
test_input = np.array([[10, 500000, 600000]])  # 5 lakhs, 6 lakhs
test_df = pd.DataFrame(test_input, columns=['rate', 'sales_in_first_month', 'sales_in_second_month'])

# Create features for test input
test_df['rate_x_sales_first'] = test_df['rate'] * test_df['sales_in_first_month']
test_df['rate_x_sales_second'] = test_df['rate'] * test_df['sales_in_second_month']
test_df['sales_first_x_sales_second'] = test_df['sales_in_first_month'] * test_df['sales_in_second_month']
test_df['sales_ratio_first_second'] = test_df['sales_in_first_month'] / (test_df['sales_in_second_month'] + 1)
test_df['sales_ratio_second_first'] = test_df['sales_in_second_month'] / (test_df['sales_in_first_month'] + 1)
test_df['sales_diff'] = test_df['sales_in_first_month'] - test_df['sales_in_second_month']
test_df['total_sales'] = test_df['sales_in_first_month'] + test_df['sales_in_second_month']
test_df['avg_sales'] = (test_df['sales_in_first_month'] + test_df['sales_in_second_month']) / 2
test_df['rate_squared'] = test_df['rate'] ** 2

# Reorder columns to match training
test_df = test_df[feature_cols]
test_scaled = scaler.transform(test_df)
pred = best_model_final.predict(test_scaled)[0]

print(f"Test Input: rate=10, sales_first_month=500000 (5 lakhs), sales_second_month=600000 (6 lakhs)")
print(f"  Predicted third month sales: {pred:.2f} (= {pred/100000:.2f} lakhs)")

# Test with higher values
test_input2 = np.array([[15, 800000, 900000]])  # 8 lakhs, 9 lakhs
test_df2 = pd.DataFrame(test_input2, columns=['rate', 'sales_in_first_month', 'sales_in_second_month'])

test_df2['rate_x_sales_first'] = test_df2['rate'] * test_df2['sales_in_first_month']
test_df2['rate_x_sales_second'] = test_df2['rate'] * test_df2['sales_in_second_month']
test_df2['sales_first_x_sales_second'] = test_df2['sales_in_first_month'] * test_df2['sales_in_second_month']
test_df2['sales_ratio_first_second'] = test_df2['sales_in_first_month'] / (test_df2['sales_in_second_month'] + 1)
test_df2['sales_ratio_second_first'] = test_df2['sales_in_second_month'] / (test_df2['sales_in_first_month'] + 1)
test_df2['sales_diff'] = test_df2['sales_in_first_month'] - test_df2['sales_in_second_month']
test_df2['total_sales'] = test_df2['sales_in_first_month'] + test_df2['sales_in_second_month']
test_df2['avg_sales'] = (test_df2['sales_in_first_month'] + test_df2['sales_in_second_month']) / 2
test_df2['rate_squared'] = test_df2['rate'] ** 2

test_df2 = test_df2[feature_cols]
test_scaled2 = scaler.transform(test_df2)
pred2 = best_model_final.predict(test_scaled2)[0]

print(f"\nTest Input: rate=15, sales_first_month=800000 (8 lakhs), sales_second_month=900000 (9 lakhs)")
print(f"  Predicted third month sales: {pred2:.2f} (= {pred2/100000:.2f} lakhs)")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best Model: {best_name}")
print(f"Accuracy: {best_score_final*100:.2f}%")
print("Model now predicts sales in LAKHS format!")
