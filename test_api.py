import requests

# Test health
r = requests.get("http://127.0.0.1:5000/health")
print("HEALTH:", r.json())

# Test predict
payload = {
    "CreditScore": 600,
    "Gender": "Female",
    "Age": 42,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000,
    "Geography": "France"
}
r = requests.post("http://127.0.0.1:5000/predict", json=payload)
print("PREDICT:", r.json())

# Test model info
r = requests.get("http://127.0.0.1:5000/model-info")
print("MODEL INFO:", r.json())