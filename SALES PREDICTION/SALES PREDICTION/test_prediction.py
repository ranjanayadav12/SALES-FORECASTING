# Test script to verify prediction works correctly
import pickle
import numpy as np
import feature_utils

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("="*60)
print("TESTING PREDICTION LOGIC")
print("="*60)

# Test Case 1: User enters in Lakhs mode (6 for 6 lakhs = 600000)
print("\nTest 1: Lakhs mode - User enters 6 for 6 lakhs")
sales_first = 6  # User enters 6 (meaning 6 lakhs)
sales_second = 11  # User enters 11 (meaning 11 lakhs)
rate = 10

# Convert to actual
sales_first_actual = sales_first * 100000  # 6 * 100000 = 600000
sales_second_actual = sales_second * 100000  # 11 * 100000 = 1100000

print(f"Input: rate={rate}, sales_first={sales_first} lakhs, sales_second={sales_second} lakhs")
print(f"After conversion: rate={rate}, sales_first={sales_first_actual}, sales_second={sales_second_actual}")

# Create features
input_features = feature_utils.create_features([rate, sales_first_actual, sales_second_actual])
feature_cols = feature_utils.get_feature_columns()
input_df = input_features[feature_cols]

# Scale
input_data = scaler.transform(input_df)

# Predict
prediction = float(model.predict(input_data)[0])
print(f"Raw prediction: {prediction}")

# Convert back to lakhs for display
output_lakhs = prediction / 100000
print(f"Prediction in lakhs: {output_lakhs:.2f} Lakhs")

# Test Case 2: User enters in Raw mode (600000 for ₹600000)
print("\n" + "="*60)
print("Test 2: Raw mode - User enters 600000 for ₹600000")
sales_first = 600000
sales_second = 1100000
rate = 10

# No conversion needed for raw mode
sales_first_actual = sales_first
sales_second_actual = sales_second

print(f"Input: rate={rate}, sales_first={sales_first}, sales_second={sales_second}")

# Create features
input_features = feature_utils.create_features([rate, sales_first_actual, sales_second_actual])
input_df = input_features[feature_cols]

# Scale
input_data = scaler.transform(input_df)

# Predict
prediction = float(model.predict(input_data)[0])
print(f"Raw prediction: {prediction}")

# Convert back to lakhs for display
output_lakhs = prediction / 100000
print(f"Prediction in lakhs: {output_lakhs:.2f} Lakhs")

print("\n" + "="*60)
print("Both tests should give similar results if inputs represent same value!")
print("="*60)
