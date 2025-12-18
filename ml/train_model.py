import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ------------------------------
# FOOLPROOF DATASET PATH
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /ml
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "student_data.csv")

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# Load dataset
data = pd.read_csv(DATASET_PATH)

# Optional: strip column names to avoid hidden spaces
data.columns = [c.strip() for c in data.columns]

# ------------------------------
# Create target column
# ------------------------------
if "dropout" not in data.columns:
    data["dropout"] = data["G3"].apply(lambda x: 1 if x < 10 else 0)

# Features and target
X = data.drop(["G3", "dropout"], axis=1)
y = data["dropout"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", accuracy)

# Save model and scaler
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
joblib.dump((model, scaler, X.columns.tolist()), MODEL_PATH)
print("✅ Model trained and saved successfully")
