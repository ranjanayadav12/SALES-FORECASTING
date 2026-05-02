import requests

# Using the correct endpoint and parameter names
url = 'http://localhost:5000/predict'
data = {
    'rate': 5,
    'sales_first_month': 200,
    'sales_second_month': 400
}

# Note: The /predict route requires authentication (session)
# For testing without auth, we would need to modify the app
# or use the web interface directly

# This is a simple test - but it will redirect to login since no session
r = requests.post(url, data=data)

print(f"Status Code: {r.status_code}")
print(f"Response: {r.text}")
