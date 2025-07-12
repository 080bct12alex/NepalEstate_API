import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000","https://nepalestates.vercel.app"]}})

# Load the ML model
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'realstate_prices_mlp_model.pickle')
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return "NepalEstate backend is running!"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        floors = float(data.get('floors', 1))
        area = float(data.get('area', 10))
        road_width = float(data.get('road_width', 10))
        city_bhaktapur = int(data.get('city_bhaktapur', 0))
        city_kathmandu = int(data.get('city_kathmandu', 0))
        city_lalitpur = int(data.get('city_lalitpur', 0))
        road_type_blacktopped = int(data.get('road_type_blacktopped', 0))
        road_type_gravelled = int(data.get('road_type_gravelled', 0))
        road_type_soil_stabilized = int(data.get('road_type_soil_stabilized', 0))

        # Load model
        model = load_model()
        if model is None:
            return jsonify({"error": "Failed to load model"}), 500

        # Prepare input features
        features = np.array([[ 
            floors, area, road_width, city_bhaktapur, city_kathmandu, city_lalitpur,
            road_type_blacktopped, road_type_gravelled, road_type_soil_stabilized
        ]])

        # Make prediction
        prediction = model.predict(features)[0]

        return jsonify({"predictedPrice": float(prediction)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
