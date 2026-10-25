from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "student_model.pkl")

# Load model safely
model, scaler, feature_names = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = [data.get(feat, 0) for feat in feature_names]
    input_scaled = scaler.transform([input_data])

    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])

    return jsonify({
        "dropout_risk": prediction,
        "probability": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


@app.route("/analytics", methods=["GET"])
def analytics():
    data = pd.read_csv("../dataset/student_data.csv")
    data["dropout"] = data["G3"].apply(lambda x: 1 if x < 10 else 0)

    return jsonify({
        "total_students": len(data),
        "dropout": int(data["dropout"].sum()),
        "safe": int(len(data) - data["dropout"].sum())
    })

if __name__ == "__main__":
    app.run(debug=True)
