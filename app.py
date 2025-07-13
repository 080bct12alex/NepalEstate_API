import os
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)  
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://nepalestates.vercel.app"
]}})

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'realstate_prices_mlp_model.pickle')
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model once at app startup , store globally (memory) :Loading the model once at startup and reusing it is the best practice for ML APIs. It greatly improves performance and resource utilization.
model = load_model()

# If model loading fails, the app exits early to avoid runtime errors.
if model is None:
    print("Failed to load model. Exiting app.")
    exit(1)       # Stop the app if model fails to load        
                  # cannot return a Flask response here, because no request context exists yet. as  return   in predict route 
                                                                                                   #  if model is None:
                                                                                                         #return jsonify({"error": "Model not loaded"}), 500                      

@app.route('/', methods=['GET'])
def index():
    if model is not None:
        return jsonify({
            "message": "NepalEstate backend is running!",
            "model_status": "Model loaded successfully"
        })
    else:
        return jsonify({
            "message": "NepalEstate backend is running!",
            "model_status": "Model failed to load"
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.get_json()
        
        
        # Extract features from request with default values
        floors = float(data.get('floors', 1))
        area = float(data.get('area', 10))
        road_width = float(data.get('road_width', 10))
        city_bhaktapur = int(data.get('city_bhaktapur', 0))
        city_kathmandu = int(data.get('city_kathmandu', 0))
        city_lalitpur = int(data.get('city_lalitpur', 0))
        road_type_blacktopped = int(data.get('road_type_blacktopped', 0))
        road_type_gravelled = int(data.get('road_type_gravelled', 0))
        road_type_soil_stabilized = int(data.get('road_type_soil_stabilized', 0))

        features = np.array([[
            floors, area, road_width,
            city_bhaktapur, city_kathmandu, city_lalitpur,
            road_type_blacktopped, road_type_gravelled, road_type_soil_stabilized
        ]])

        prediction = model.predict(features)[0]

        return jsonify({"predictedPrice": float(prediction)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
     # debug=True for development only; use False or production server in production    # when deploying your Flask app (e.g., to Render, Heroku, or any cloud provider), you should not use the app.run(...) block like this:  
    app.run(debug=True, port=5001)    # if you include app.run(...), it's not needed and can cause issues or conflicts in production (especially with port binding or process management).
                                        # it's only appropriate for local development, not for production deployments like Render.
                                        # Why it works locally:   app.run() uses the Flask development server . It's great for quick testing and debugging on your machine You control the port (e.g., 5001)

                                  #Note : f you include app.run(...), it's ignored on Render — unless misconfigured — but it can cause confusion, conflicts, or prevent your app from starting if the Render environment doesn't allow custom ports like 5001.

