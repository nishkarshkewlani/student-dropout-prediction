import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load real dataset
data = pd.read_csv("../dataset/student_data.csv")

# Create dropout label (real logic)
data["dropout"] = data["G3"].apply(lambda x: 1 if x < 10 else 0)

# Encode categorical columns
for col in data.select_dtypes(include="object").columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Features and target
X = data.drop(["G3", "dropout"], axis=1)
y = data["dropout"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("student_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X.columns.tolist()), f)

print("✅ Model trained on real dataset and saved")
