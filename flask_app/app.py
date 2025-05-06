from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for requests from React

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    print("Received data:", data)

    # Dummy logic – you can replace with ML model
    severity_class = 2  # Example fixed response

    return jsonify({
        "message": "Form received successfully!",
        "receivedData": data,
        "severityClass": severity_class
    })

if __name__ == '__main__':
    app.run(debug=True)
