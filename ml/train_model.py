import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------------
# PATH SETUP (Codespaces safe)
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "student_data.csv")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

# ---------------------------------
# LOAD DATA
# ---------------------------------
data = pd.read_csv(DATASET_PATH)
data.columns = [c.strip() for c in data.columns]

# ---------------------------------
# CREATE DROPOUT COLUMN
# ---------------------------------
if "G3" not in data.columns:
    raise ValueError("Dataset must contain column 'G3'")

data["dropout"] = data["G3"].apply(lambda x: 1 if x < 10 else 0)

# ---------------------------------
# ENCODE CATEGORICAL COLUMNS
# ---------------------------------
categorical_cols = data.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# ---------------------------------
# FEATURES & TARGET
# ---------------------------------
X = data.drop(["G3", "dropout"], axis=1)
y = data["dropout"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------
# TRAIN MODEL
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model Accuracy:", accuracy)

# ---------------------------------
# SAVE MODEL (NO PICKLE ERRORS)
# ---------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
joblib.dump((model, scaler, X.columns.tolist()), MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)
