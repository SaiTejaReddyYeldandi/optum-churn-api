from flask import Flask, request, jsonify
import pickle
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)
with open('metrics.json', 'r') as f:
    metrics = json.load(f)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'XGBoost Churn Predictor',
        'AUC': metrics['AUC'],
        'F1': metrics['F1']
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data'}), 400

        # Validate required fields
        required = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                    'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                    'EstimatedSalary', 'Geography']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        # Build feature vector
        gender = 1 if data['Gender'].lower() == 'male' else 0
        geography = data['Geography'].lower()

        features = {
            'CreditScore': data['CreditScore'],
            'Gender': gender,
            'Age': data['Age'],
            'Tenure': data['Tenure'],
            'Balance': data['Balance'],
            'NumOfProducts': data['NumOfProducts'],
            'HasCrCard': data['HasCrCard'],
            'IsActiveMember': data['IsActiveMember'],
            'EstimatedSalary': data['EstimatedSalary'],
            'Geography_France': 1 if geography == 'france' else 0,
            'Geography_Germany': 1 if geography == 'germany' else 0,
            'Geography_Spain': 1 if geography == 'spain' else 0,
        }

        X = np.array([[features[f] for f in feature_names]])
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = int(prob >= 0.5)

        risk = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'

        log.info(f"Prediction: {prediction}, Probability: {prob:.4f}, Risk: {risk}")

        return jsonify({
            'churn_prediction': prediction,
            'churn_probability': round(float(prob), 4),
            'risk_level': risk,
            'interpretation': 'Will churn' if prediction == 1 else 'Will not churn'
        })

    except Exception as e:
        log.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'XGBoost Classifier',
        'features': feature_names,
        'metrics': metrics,
        'smote_applied': True,
        'training_rows': 12740
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

