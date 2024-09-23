import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS to communicate with frontend

# Load scaler and classifier models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
classifier_path = os.path.join(BASE_DIR, 'classifier.pkl')

# Load the models from the pickle files
scaler = joblib.load(scaler_path)
classifier = joblib.load(classifier_path)

# Define the prediction function
def predict(input_features):
    # Convert the input features into a NumPy array and reshape
    input_features = np.array(input_features).reshape(1, -1)
    
    # Scale the features using the pre-loaded scaler
    scaled_features = scaler.transform(input_features)
    
    # Use the classifier to make a prediction
    prediction = classifier.predict(scaled_features)
    
    return prediction[0]

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Get the input features from the request (assumed to be a JSON object)
        data = request.json
        input_features = data.get('features', [0] * 134)  # Default to [0] * 134 if not provided
        
        # Validate that the input has 134 elements
        if len(input_features) != 134:
            return jsonify({"error": "Input feature list must have exactly 134 elements"}), 400
        
        # Get the prediction result
        return predict(input_features)
         
        # Return the prediction as a JSON response
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
