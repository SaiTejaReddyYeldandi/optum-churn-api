import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['status'] == 'healthy'
    assert 'AUC' in data
    assert 'F1' in data

def test_predict_valid_input(client):
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
    r = client.post('/predict',
        data=json.dumps(payload),
        content_type='application/json')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'churn_prediction' in data
    assert 'churn_probability' in data
    assert 'risk_level' in data
    assert data['churn_prediction'] in [0, 1]
    assert 0 <= data['churn_probability'] <= 1

def test_predict_missing_fields(client):
    payload = {"CreditScore": 600}
    r = client.post('/predict',
        data=json.dumps(payload),
        content_type='application/json')
    assert r.status_code == 400
    data = json.loads(r.data)
    assert 'error' in data

def test_predict_high_risk(client):
    payload = {
        "CreditScore": 400,
        "Gender": "Female",
        "Age": 55,
        "Tenure": 1,
        "Balance": 120000,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 30000,
        "Geography": "Germany"
    }
    r = client.post('/predict',
        data=json.dumps(payload),
        content_type='application/json')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['churn_probability'] > 0.5

def test_model_info(client):
    r = client.get('/model-info')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'features' in data
    assert 'metrics' in data