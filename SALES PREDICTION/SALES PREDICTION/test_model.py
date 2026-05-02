import pickle
import numpy as np

try:
    with open("linear_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)
    
    # Test a typical case: Rate=10, Sales1=1, Sales2=1 (in Lakhs)
    X = np.array([[10, 1, 1]])
    pred = model.predict(X)[0]
    print(f"Test Prediction (Rate=10, S1=1, S2=1): {pred}")
    
    # Test a small case: Rate=1, Sales1=0.1, Sales2=0.1
    X2 = np.array([[1, 0.1, 0.1]])
    pred2 = model.predict(X2)[0]
    print(f"Test Prediction (Rate=1, S1=0.1, S2=0.1): {pred2}")
except Exception as e:
    print(f"Error: {e}")
