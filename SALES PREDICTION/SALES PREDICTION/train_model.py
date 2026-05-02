import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# STEP 1: Generate High-Quality Synthetic Training Data
# ============================================================
print("="*60)
print("STEP 1: Generating Synthetic Training Data")
print("="*60)

# Generate 500+ realistic sales records
n_samples = 500

# Random but realistic values
rates = np.random.uniform(3, 20, n_samples)  # Rate between 3-20%
sales_first = np.random.uniform(100000, 1500000, n_samples)  # 1-15 lakhs
sales_second = np.random.uniform(100000, 1500000, n_samples)  # 1-15 lakhs

# Create realistic third month sales based on patterns
# Third month sales = weighted avg of first two + rate effect + some noise
base_third = (sales_first * 0.4 + sales_second * 0.5)
rate_effect = rates * 5000  # Higher rate = slightly lower expected sales
noise = np.random.normal(0, 50000, n_samples)  # Random noise

sales_third = base_third - rate_effect + noise
sales_third = np.maximum(sales_third, 50000)  # Minimum 50k

# Create DataFrame
df = pd.DataFrame({
    'rate': rates,
    'sales_in_first_month': sales_first,
    'sales_in_second_month': sales_second,
    'sales_in_third_month': sales_third
})

# Round values
df = df.round(2)

print(f"Generated {len(df)} training samples")
print(f"Rate range: {df['rate'].min():.2f} - {df['rate'].max():.2f}")
print(f"Sales 1st range: ₹{df['sales_in_first_month'].min():,.0f} - ₹{df['sales_in_first_month'].max():,.0f}")
print(f"Sales 2nd range: ₹{df['sales_in_second_month'].min():,.0f} - ₹{df['sales_in_second_month'].max():,.0f}")
print(f"Sales 3rd range: ₹{df['sales_in_third_month'].min():,.0f} - ₹{df['sales_in_third_month'].max():,.0f}")

# Save training data
df.to_csv('training_data.csv', index=False)
print("Training data saved to training_data.csv")

# ============================================================
# STEP 2: Feature Engineering
# ============================================================
print("\n" + "="*60)
print("STEP 2: Feature Engineering")
print("="*60)

# Create features
df['rate_x_sales_first'] = df['rate'] * df['sales_in_first_month']
df['rate_x_sales_second'] = df['rate'] * df['sales_in_second_month']
df['sales_first_x_sales_second'] = df['sales_in_first_month'] * df['sales_in_second_month']
df['sales_ratio_first_second'] = df['sales_in_first_month'] / (df['sales_in_second_month'] + 1)
df['sales_ratio_second_first'] = df['sales_in_second_month'] / (df['sales_in_first_month'] + 1)
df['sales_diff'] = df['sales_in_first_month'] - df['sales_in_second_month']
df['total_sales'] = df['sales_in_first_month'] + df['sales_in_second_month']
df['avg_sales'] = (df['sales_in_first_month'] + df['sales_in_second_month']) / 2
df['rate_squared'] = df['rate'] ** 2
df['sales_growth_rate'] = (df['sales_in_second_month'] - df['sales_in_first_month']) / (df['sales_in_first_month'] + 1)

feature_cols = [
    'rate', 'sales_in_first_month', 'sales_in_second_month',
    'rate_x_sales_first', 'rate_x_sales_second', 'sales_first_x_sales_second',
    'sales_ratio_first_second', 'sales_ratio_second_first',
    'sales_diff', 'total_sales', 'avg_sales', 'rate_squared', 'sales_growth_rate'
]

X = df[feature_cols]
y = df['sales_in_third_month']

print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# ============================================================
# STEP 3: Feature Scaling
# ============================================================
print("\n" + "="*60)
print("STEP 3: Feature Scaling")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Scaler saved to scaler.pkl")

# ============================================================
# STEP 4: Model Training with Cross-Validation
# ============================================================
print("\n" + "="*60)
print("STEP 4: Model Training with 5-Fold Cross-Validation")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    mae_scores = -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mean = scores.mean()
    cv_std = scores.std()
    cv_results[name] = {'cv_mean': cv_mean, 'cv_std': cv_std, 'mae': mae_scores.mean()}
    print(f"{name}: CV R² = {cv_mean*100:.2f}% (+/- {cv_std*100:.2f}%), MAE = ₹{mae_scores.mean():,.0f}")

best_model_name = max(cv_results, key=lambda x: cv_results[x]['cv_mean'])
best_cv_score = cv_results[best_model_name]['cv_mean']
print(f"\nBest Single Model: {best_model_name} with CV R² = {best_cv_score*100:.2f}%")

# ============================================================
# STEP 5: Hyperparameter Tuning
# ============================================================
print("\n" + "="*60)
print("STEP 5: Hyperparameter Tuning")
print("="*60)

print("\nTuning Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.15],
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

print("\nTuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
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

# ============================================================
# STEP 6: Create Ensemble
# ============================================================
print("\n" + "="*60)
print("STEP 6: Creating Ensemble Model")
print("="*60)

ensemble = VotingRegressor([
    ('gb', gb_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_)
])

ensemble_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='r2')
ensemble_cv_mean = ensemble_scores.mean()
ensemble_cv_std = ensemble_scores.std()
print(f"Ensemble CV R² = {ensemble_cv_mean*100:.2f}% (+/- {ensemble_cv_std*100:.2f}%)")

# ============================================================
# STEP 7: Select Best Model
# ============================================================
print("\n" + "="*60)
print("STEP 7: Selecting Best Model")
print("="*60)

all_models = {
    'Tuned Gradient Boosting': (gb_grid.best_estimator_, gb_grid.best_score_),
    'Tuned Random Forest': (rf_grid.best_estimator_, rf_grid.best_score_),
    'Ensemble': (ensemble, ensemble_cv_mean)
}

best_name = max(all_models, key=lambda x: all_models[x][1])
best_model_final = all_models[best_name][0]
best_score_final = all_models[best_name][1]

print("\nFinal Model Comparison:")
for name, (model, score) in all_models.items():
    print(f"  {name}: R² = {score*100:.2f}%")

print(f"\n*** BEST MODEL: {best_name} ***")
print(f"*** ACCURACY: {best_score_final*100:.2f}% ***")

# ============================================================
# STEP 8: Train Final Model
# ============================================================
print("\n" + "="*60)
print("STEP 8: Training Final Model on All Data")
print("="*60)

best_model_final.fit(X_scaled, y)

y_pred = best_model_final.predict(X_scaled)
final_r2 = r2_score(y, y_pred)
final_mae = mean_absolute_error(y, y_pred)
final_rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Final R² Score: {final_r2*100:.2f}%")
print(f"Final MAE: ₹{final_mae:,.2f}")
print(f"Final RMSE: ₹{final_rmse:,.2f}")

pickle.dump(best_model_final, open('model.pkl', 'wb'))
print("\nModel saved to model.pkl")

# Save model info
model_info = {
    'feature_cols': feature_cols,
    'best_model_name': best_name,
    'accuracy': best_score_final * 100,
    'r2_score': final_r2 * 100,
    'mae': final_mae,
    'rmse': final_rmse
}
pickle.dump(model_info, open('model_info.pkl', 'wb'))
print("Model info saved to model_info.pkl")

# ============================================================
# STEP 9: Test Predictions
# ============================================================
print("\n" + "="*60)
print("STEP 9: Test Predictions")
print("="*60)

def create_features_and_predict(rate, sales_first, sales_second):
    """Helper function to create features and predict"""
    # Create feature dataframe
    test_df = pd.DataFrame({
        'rate': [rate],
        'sales_in_first_month': [sales_first],
        'sales_in_second_month': [sales_second]
    })
    
    # Create engineered features
    test_df['rate_x_sales_first'] = test_df['rate'] * test_df['sales_in_first_month']
    test_df['rate_x_sales_second'] = test_df['rate'] * test_df['sales_in_second_month']
    test_df['sales_first_x_sales_second'] = test_df['sales_in_first_month'] * test_df['sales_in_second_month']
    test_df['sales_ratio_first_second'] = test_df['sales_in_first_month'] / (test_df['sales_in_second_month'] + 1)
    test_df['sales_ratio_second_first'] = test_df['sales_in_second_month'] / (test_df['sales_in_first_month'] + 1)
    test_df['sales_diff'] = test_df['sales_in_first_month'] - test_df['sales_in_second_month']
    test_df['total_sales'] = test_df['sales_in_first_month'] + test_df['sales_in_second_month']
    test_df['avg_sales'] = (test_df['sales_in_first_month'] + test_df['sales_in_second_month']) / 2
    test_df['rate_squared'] = test_df['rate'] ** 2
    test_df['sales_growth_rate'] = (test_df['sales_in_second_month'] - test_df['sales_in_first_month']) / (test_df['sales_in_first_month'] + 1)
    
    # Select features
    test_X = test_df[feature_cols]
    test_X_scaled = scaler.transform(test_X)
    
    # Predict
    prediction = best_model_final.predict(test_X_scaled)[0]
    return prediction

# Test cases
tests = [
    (5, 500000, 600000),   # 5 lakhs, 6 lakhs
    (10, 800000, 700000),  # 8 lakhs, 7 lakhs
    (15, 1000000, 1200000), # 10 lakhs, 12 lakhs
    (8, 300000, 400000),   # 3 lakhs, 4 lakhs
]

print("\nTest Predictions:")
for rate, s1, s2 in tests:
    pred = create_features_and_predict(rate, s1, s2)
    print(f"Rate: {rate}%, Sales: ₹{s1/100000:.1f}L, ₹{s2/100000:.1f}L -> Predicted: ₹{pred/100000:.2f}L")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best Model: {best_name}")
print(f"Accuracy (CV R²): {best_score_final*100:.2f}%")
print(f"Final R²: {final_r2*100:.2f}%")
print(f"MAE: ₹{final_mae:,.2f}")
