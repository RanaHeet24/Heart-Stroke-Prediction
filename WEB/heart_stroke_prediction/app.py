# Flask backend for heart stroke prediction

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Helper to load model/scaler if present
def load_artifact(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            return pickle.load(open(path, 'rb'))
        except Exception:
            return None
    return None

# Try to load model and scaler (may be created by train_model.py)
model = load_artifact('model.pkl')
scaler = load_artifact('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not found. Run train_model.py to create artifacts.'}), 500

    data = request.get_json() or {}

    # Accept either a `features` array or named fields (matching the form)
    if 'features' in data:
        try:
            features = np.array(data['features'], dtype=float).reshape(1, -1)
        except Exception:
            return jsonify({'error': 'Invalid features format'}), 400
    else:
        # Expect named fields used in index.html: age, hypertension, heart_disease, avg_glucose_level, bmi
        try:
            fields = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
            features = np.array([float(data.get(f, 0)) for f in fields]).reshape(1, -1)
        except Exception:
            return jsonify({'error': 'Invalid input fields'}), 400

    try:
        features_scaled = scaler.transform(features)
    except Exception as e:
        return jsonify({'error': 'Scaler transform failed', 'details': str(e)}), 500

    try:
        pred = model.predict(features_scaled)
        result = int(pred[0])

        # If model supports predict_proba, include probability for class 1
        prob = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features_scaled)
            prob = float(probs[0][1])

        human = 'Stroke' if result == 1 else 'No Stroke'
        resp = {'result': human, 'prediction': result}
        if prob is not None:
            resp['probability'] = prob

        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
