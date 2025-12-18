import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Get absolute path safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "student_data.csv")

# Load dataset
data = pd.read_csv(DATASET_PATH)

X = data.drop("dropout", axis=1)
y = data["dropout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Model accuracy:", accuracy)

# Save model
MODEL_PATH = os.path.join(BASE_DIR, "student_model.pkl")
joblib.dump(model, MODEL_PATH)

print("Model trained and saved successfully")
