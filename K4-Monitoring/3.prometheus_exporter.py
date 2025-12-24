"""
Prometheus Exporter for ML Model Inference Metrics
Author: Anwar-Rohmadi
Model: HistGradientBoostingRegressor (House Price Prediction)
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Flask, Response, request, jsonify
import joblib
import numpy as np
import json
import time
import os

app = Flask(__name__)

# ===== LOAD TRAINED MODEL =====
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pkl')
FEATURES_PATH = os.environ.get('FEATURES_PATH', 'feature_names.json')

print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, 'r') as f:
    feature_config = json.load(f)
FEATURE_NAMES = feature_config['features']
N_FEATURES = feature_config['n_features']
print(f"Model loaded successfully! ({N_FEATURES} features)")

# ===== PROMETHEUS METRICS =====
REQUEST_COUNT = Counter('model_request_total', 'Total model prediction requests', ['status'])
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Request latency', 
                            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0])
PREDICTION_VALUE = Gauge('model_prediction_value', 'Last prediction value')
PREDICTION_MEAN = Gauge('model_prediction_mean', 'Mean of predictions')
PREDICTION_STD = Gauge('model_prediction_std', 'Std of predictions')
ERROR_COUNT = Counter('model_error_total', 'Total prediction errors')
ACTIVE_REQUESTS = Gauge('model_active_requests', 'Currently processing requests')

predictions_history = []

@app.route('/predict', methods=['POST'])
def predict():
    """Predict house price from features using trained HGBR model"""
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    try:
        data = request.get_json()
        features = data.get('features', [])
        
        # Ensure correct number of features
        if len(features) != N_FEATURES:
            if len(features) < N_FEATURES:
                features = features + [0] * (N_FEATURES - len(features))
            else:
                features = features[:N_FEATURES]
        
        # Make prediction with REAL trained model
        features_array = np.array(features).reshape(1, -1)
        prediction = float(model.predict(features_array)[0])
        
        # Update prediction history
        predictions_history.append(prediction)
        if len(predictions_history) > 100:
            predictions_history.pop(0)
        
        # Update Prometheus metrics
        PREDICTION_VALUE.set(prediction)
        PREDICTION_MEAN.set(np.mean(predictions_history))
        PREDICTION_STD.set(np.std(predictions_history) if len(predictions_history) > 1 else 0)
        REQUEST_COUNT.labels(status='success').inc()
        
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        
        ACTIVE_REQUESTS.dec()
        return jsonify({
            'prediction': prediction,
            'latency': latency,
            'model': 'HistGradientBoostingRegressor'
        })
    
    except Exception as e:
        ERROR_COUNT.inc()
        REQUEST_COUNT.labels(status='error').inc()
        ACTIVE_REQUESTS.dec()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'HistGradientBoostingRegressor',
        'n_features': N_FEATURES
    })

@app.route('/model-info')
def model_info():
    """Model information endpoint"""
    try:
        with open('model_info.json', 'r') as f:
            info = json.load(f)
        return jsonify(info)
    except:
        return jsonify({'error': 'Model info not found'}), 404

@app.route('/')
def index():
    """Root endpoint with API documentation"""
    return jsonify({
        'name': 'House Price Prediction API',
        'author': 'Anwar-Rohmadi',
        'model': 'HistGradientBoostingRegressor',
        'endpoints': {
            '/predict': 'POST - Predict house price',
            '/health': 'GET - Health check',
            '/metrics': 'GET - Prometheus metrics',
            '/model-info': 'GET - Model information'
        }
    })

if __name__ == '__main__':
    print("=" * 50)
    print("HOUSE PRICE PREDICTION API")
    print(f"Model: HistGradientBoostingRegressor")
    print(f"Features: {N_FEATURES}")
    print("Starting server on port 8000...")
    print("=" * 50)
    app.run(host='0.0.0.0', port=8000)
