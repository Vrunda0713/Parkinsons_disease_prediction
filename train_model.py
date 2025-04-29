import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("cleaned_parkinsons_dataset.csv")  # Ensure the correct file path

# Define features and target
X = data.drop(columns=['status'])  # Features
y = data['status']  # Target (binary: 0 = Healthy, 1 = Parkinson's)

# Handle class imbalance dynamically
class_weights = len(y) / (2 * np.bincount(y))
scale_pos_weight = class_weights[0] / class_weights[1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=2,
    gamma=0.4,
    subsample=0.95,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(X_train_scaled, y_train)

# Predictions and Evaluation
y_pred = xgb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Save model and scaler
joblib.dump(xgb_model, "xgb_parkinsons_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("🚀 Model and scaler saved successfully!")
import os
print(os.getcwd())  # This will show exactly where the files are saved
