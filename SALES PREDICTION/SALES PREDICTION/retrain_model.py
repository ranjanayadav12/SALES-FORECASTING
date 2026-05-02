import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the combined data
data = pd.read_csv('combined_sales_data.csv')

print("="*60)
print("COMBINED SALES DATA")
print("="*60)
print(data.head(10))
print(f"\nDataset shape: {data.shape}")
print(f"\nData Statistics:")
print(data.describe())

# Drop unnamed index column if present
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# ============ ADVANCED FEATURE ENGINEERING ============
print("\n" + "="*60)
print("ADVANCED FEATURE ENGINEERING")
print("="*60)

df = data.copy()

# 1. Basic features
df['rate_x_sales_first'] = df['rate'] * df['sales_in_first_month']
df['rate_x_sales_second'] = df['rate'] * df['sales_in_second_month']
df['sales_first_x_sales_second'] = df['sales_in_first_month'] * df['sales_in_second_month']

# 2. Ratio features (with smoothing to avoid division by zero)
df['sales_ratio_first_second'] = df['sales_in_first_month'] / (df['sales_in_second_month'] + 1)
df['sales_ratio_second_first'] = df['sales_in_second_month'] / (df['sales_in_first_month'] + 1)

# 3. Difference and trend features
df['sales_diff'] = df['sales_in_first_month'] - df['sales_in_second_month']
df['sales_trend'] = (df['sales_in_second_month'] - df['sales_in_first_month']) / (df['sales_in_first_month'] + 1)

# 4. Total and average
df['total_sales'] = df['sales_in_first_month'] + df['sales_in_second_month']
df['avg_sales'] = (df['sales_in_first_month'] + df['sales_in_second_month']) / 2

# 5. Rate features
df['rate_squared'] = df['rate'] ** 2
df['rate_log'] = np.log1p(df['rate'])

# 6. Sales polynomial features
df['sales_first_squared'] = df['sales_in_first_month'] ** 2
df['sales_second_squared'] = df['sales_in_second_month'] ** 2
df['sales_first_log'] = np.log1p(df['sales_in_first_month'])
df['sales_second_log'] = np.log1p(df['sales_in_second_month'])

# 7. Normalized features (important for prediction stability)
df['sales_first_norm'] = df['sales_in_first_month'] / (df['total_sales'] + 1)
df['sales_second_norm'] = df['sales_in_second_month'] / (df['total_sales'] + 1)

# 8. Rate impact features
df['rate_per_sales_first'] = df['rate'] / (df['sales_in_first_month'] + 1) * 100000
df['rate_per_sales_second'] = df['rate'] / (df['sales_in_second_month'] + 1) * 100000

print("Features created:")
print(df.columns.tolist())

# Prepare features and target
feature_cols = [col for col in df.columns if col not in ['sales_in_third_month']]
X = df[feature_cols]
y = df['sales_in_third_month']

print(f"\nTotal features: {len(feature_cols)}")

# ============ ROBUST SCALING ============
print("\n" + "="*60)
print("FEATURE SCALING (ROBUST)")
print("="*60)

scaler = RobustScaler()  # More robust to outliers than StandardScaler
X_scaled = scaler.fit_transform(X)

# Save scaler
pickle.dump(scaler, open('scaler.pkl', 'wb'))
print("Scaler saved to scaler.pkl")

# ============ MULTIPLE MODEL EVALUATION ============
print("\n" + "="*60)
print("EVALUATING MULTIPLE MODELS")
print("="*60)

# Define models to evaluate
models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf', C=1000, gamma='scale'),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    mae_scores = -cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
    cv_results[name] = {
        'r2_mean': scores.mean(),
        'r2_std': scores.std(),
        'mae_mean': mae_scores.mean()
    }
    print(f"{name}: R² = {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%), MAE = {mae_scores.mean():,.0f}")

# Find best single model
best_single_name = max(cv_results, key=lambda x: cv_results[x]['r2_mean'])
print(f"\nBest Single Model: {best_single_name} with R² = {cv_results[best_single_name]['r2_mean']*100:.2f}%")

# ============ HYPERPARAMETER TUNING FOR TOP MODELS ============
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

# Tune Gradient Boosting (usually best for this type of data)
print("\nTuning Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 150, 200, 250],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 0.9, 1.0]
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
    'n_estimators': [150, 200, 250],
    'max_depth': [8, 10, 12, None],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2]
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

# ============ STACKING ENSEMBLE ============
print("\n" + "="*60)
print("CREATING STACKING ENSEMBLE")
print("="*60)

# Create a stacking regressor with the best models
estimators = [
    ('gb', gb_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_),
    ('ridge', Ridge(alpha=1.0))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=0.5),
    cv=5,
    n_jobs=-1
)

stacking_scores = cross_val_score(stacking, X_scaled, y, cv=5, scoring='r2')
print(f"Stacking Ensemble CV R² = {stacking_scores.mean()*100:.2f}% (+/- {stacking_scores.std()*100:.2f}%)")

# ============ FINAL MODEL SELECTION ============
print("\n" + "="*60)
print("FINAL MODEL SELECTION")
print("="*60)

all_models = {
    'Tuned Gradient Boosting': (gb_grid.best_estimator_, gb_grid.best_score_),
    'Tuned Random Forest': (rf_grid.best_estimator_, rf_grid.best_score_),
    'Stacking Ensemble': (stacking, stacking_scores.mean())
}

print("\nFinal Model Comparison:")
for name, (model, score) in all_models.items():
    print(f"  {name}: R² = {score*100:.2f}%")

# Select the best model
best_name = max(all_models, key=lambda x: all_models[x][1])
best_model_final = all_models[best_name][0]
best_score_final = all_models[best_name][1]

print(f"\n*** SELECTED MODEL: {best_name} ***")
print(f"*** ACCURACY: {best_score_final*100:.2f}% ***")

# ============ TRAIN FINAL MODEL ============
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*60)

best_model_final.fit(X_scaled, y)

# Calculate final metrics
y_pred = best_model_final.predict(X_scaled)
final_r2 = r2_score(y, y_pred)
final_mae = mean_absolute_error(y, y_pred)
final_rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"Final R² Score: {final_r2*100:.2f}%")
print(f"Final MAE: {final_mae:,.2f} (in rupees)")
print(f"Final RMSE: {final_rmse:,.2f} (in rupees)")

# Save the model
pickle.dump(best_model_final, open('model.pkl', 'wb'))
print("\nModel saved to model.pkl")

# Save feature names and model info
feature_info = {
    'feature_cols': feature_cols,
    'best_model_name': best_name,
    'accuracy': best_score_final * 100,
    'r2_score': final_r2 * 100,
    'mae': final_mae,
    'rmse': final_rmse,
    'scale': 'raw_rupees'
}
pickle.dump(feature_info, open('model_info.pkl', 'wb'))
print("Model info saved to model_info.pkl")

# ============ TEST PREDICTIONS ============
print("\n" + "="*60)
print("TEST PREDICTIONS")
print("="*60)

def test_prediction(rate, sales_first_lakhs, sales_second_lakhs):
    """Test prediction with inputs in lakhs"""
    # Convert lakhs to actual rupees
    sales_first = sales_first_lakhs * 100000
    sales_second = sales_second_lakhs * 100000
    
    # Create features
    test_df = pd.DataFrame([[rate, sales_first, sales_second]], 
                          columns=['rate', 'sales_in_first_month', 'sales_in_second_month'])
    
    # Create all features
    test_df['rate_x_sales_first'] = test_df['rate'] * test_df['sales_in_first_month']
    test_df['rate_x_sales_second'] = test_df['rate'] * test_df['sales_in_second_month']
    test_df['sales_first_x_sales_second'] = test_df['sales_in_first_month'] * test_df['sales_in_second_month']
    test_df['sales_ratio_first_second'] = test_df['sales_in_first_month'] / (test_df['sales_in_second_month'] + 1)
    test_df['sales_ratio_second_first'] = test_df['sales_in_second_month'] / (test_df['sales_in_first_month'] + 1)
    test_df['sales_diff'] = test_df['sales_in_first_month'] - test_df['sales_in_second_month']
    test_df['sales_trend'] = (test_df['sales_in_second_month'] - test_df['sales_in_first_month']) / (test_df['sales_in_first_month'] + 1)
    test_df['total_sales'] = test_df['sales_in_first_month'] + test_df['sales_in_second_month']
    test_df['avg_sales'] = (test_df['sales_in_first_month'] + test_df['sales_in_second_month']) / 2
    test_df['rate_squared'] = test_df['rate'] ** 2
    test_df['rate_log'] = np.log1p(test_df['rate'])
    test_df['sales_first_squared'] = test_df['sales_in_first_month'] ** 2
    test_df['sales_second_squared'] = test_df['sales_in_second_month'] ** 2
    test_df['sales_first_log'] = np.log1p(test_df['sales_in_first_month'])
    test_df['sales_second_log'] = np.log1p(test_df['sales_in_second_month'])
    test_df['sales_first_norm'] = test_df['sales_in_first_month'] / (test_df['total_sales'] + 1)
    test_df['sales_second_norm'] = test_df['sales_in_second_month'] / (test_df['total_sales'] + 1)
    test_df['rate_per_sales_first'] = test_df['rate'] / (test_df['sales_in_first_month'] + 1) * 100000
    test_df['rate_per_sales_second'] = test_df['rate'] / (test_df['sales_in_second_month'] + 1) * 100000
    
    # Reorder columns
    test_df = test_df[feature_cols]
    
    # Scale
    test_scaled = scaler.transform(test_df)
    
    # Predict
    prediction = best_model_final.predict(test_scaled)[0]
    
    # Convert to lakhs
    prediction_lakhs = prediction / 100000
    
    return prediction, prediction_lakhs

# Test the case that was giving wrong results
print("\n--- Test Case 1: User enters 3 and 4 lakhs ---")
pred_raw, pred_lakhs = test_prediction(rate=10, sales_first_lakhs=3, sales_second_lakhs=4)
print(f"Input: rate=10, sales_first=3 lakhs, sales_second=4 lakhs")
print(f"Prediction: ₹{pred_raw:,.0f} = {pred_lakhs:.2f} lakhs")

# Test with training data similar values
print("\n--- Test Case 2: Similar to training data ---")
pred_raw2, pred_lakhs2 = test_prediction(rate=10, sales_first_lakhs=8, sales_second_lakhs=6)
print(f"Input: rate=10, sales_first=8 lakhs, sales_second=6 lakhs")
print(f"Prediction: ₹{pred_raw2:,.0f} = {pred_lakhs2:.2f} lakhs")

print("\n--- Test Case 3: Lower values ---")
pred_raw3, pred_lakhs3 = test_prediction(rate=5, sales_first_lakhs=3, sales_second_lakhs=2.5)
print(f"Input: rate=5, sales_first=3 lakhs, sales_second=2.5 lakhs")
print(f"Prediction: ₹{pred_raw3:,.0f} = {pred_lakhs3:.2f} lakhs")

print("\n" + "="*60)
print("MODEL RETRAINING COMPLETE!")
print("="*60)
print(f"Model: {best_name}")
print(f"Accuracy: {best_score_final*100:.2f}%")
print("The model now predicts based on actual patterns in the data!")
