import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Load the data
data = pd.read_csv('clean_sales.csv')

# Prepare features and target
X = data.iloc[:, 1:4]  # rate, sales_in_first_month, sales_in_second_month
y = data.iloc[:, -1]   # sales_in_third_month

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X, y)

# Check accuracy
score = gb_model.score(X, y)
print(f"Gradient Boosting Accuracy: {score*100:.2f}%")

# Save the model
pickle.dump(gb_model, open('model.pkl', 'wb'))
print("Model saved to model.pkl")

# Test prediction
test_input = np.array([[4, 300, 500]])
prediction = gb_model.predict(test_input)
print(f"\nTest Prediction for input {test_input.tolist()}:")
print(f"Predicted sales: {prediction[0]:.2f}")
