from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)
CORS(app)  # Ensures CORS is enabled

# Load the model from MLflow model registry
model_name = "Cancer_Severity_Model"
model_uri = f"models:/{model_name}/production"
model = mlflow.pyfunc.load_model(model_uri)

label_map = {0: 'Low', 1: 'Medium', 2: 'High'}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON from POST request
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])

        # Predict the result
        prediction = model.predict(input_data)

        # Decode numeric predictions into labels
        decoded_predictions = [label_map.get(int(pred), "Unknown") for pred in prediction]

        return jsonify({"predictions": decoded_predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=9000)
