import pandas as pd
import numpy as np

# Read the current data
df = pd.read_csv('clean_sales.csv')

print("Original data:")
print(df)

# Convert sales values to lakhs (multiply by 1000 to make 500 = 5 lakhs)
# This assumes the original values are in thousands and we want to convert to lakhs
# 500 * 1000 = 500000 = 5 lakhs

# First, handle the rate column - convert words to numbers
rate_mapping = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}

# Convert rate to numeric (handle missing and string values)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce').fillna(0)

# Convert sales to lakhs (multiply by 1000)
# This converts 500 to 500000 (5 lakhs)
df['sales_in_first_month'] = df['sales_in_first_month'] * 1000
df['sales_in_second_month'] = df['sales_in_second_month'] * 1000
df['sales_in_third_month'] = df['sales_in_third_month'] * 1000

print("\nConverted to lakhs:")
print(df)

# Save the lakhs data
df.to_csv('sales_lakhs.csv', index=False)
print("\nSaved to sales_lakhs.csv")
