from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("../ml/student_model.pkl", "rb") as f:
    model, scaler, feature_names = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.json

    features = [input_data[feature] for feature in feature_names]
    features = np.array(features).reshape(1, -1)

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    risk = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"

    return jsonify({
        "dropout_prediction": int(prediction),
        "probability": round(probability, 2),
        "risk_level": risk
    })

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
