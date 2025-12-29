"""
Simple training script to create a StandardScaler and a KNeighborsClassifier
from the provided CSV. Saves `model.pkl` and `scaler.pkl` in the project root.

Assumptions:
- CSV columns include at least: Age, Heart_Disease, Hypertension (or High_BP), Avg_Glucose_Level, BMI, Stroke
- The dataset in this repo is `stroke_prediction_custom_dataset.csv` and contains header row.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

CSV = 'stroke_prediction_custom_dataset.csv'
MODEL_OUT = 'model.pkl'
SCALER_OUT = 'scaler.pkl'

if not os.path.exists(CSV):
    raise SystemExit(f"CSV file not found: {CSV}")

# Read CSV
df = pd.read_csv(CSV)

# Basic mappings for yes/no and gender if present
map_yes_no = lambda x: 1 if str(x).strip().lower() in ['yes', 'y', '1', 'true', 'male'] else 0

# Inspect columns and choose features present in our frontend
# We'll try to build features: age, hypertension, heart_disease, avg_glucose_level, bmi
cols = df.columns.str.lower()

# Create a simple robust extractor
def col_exists(names):
    for n in names:
        for c in df.columns:
            if c.lower() == n:
                return c
    return None

age_col = col_exists(['age'])
hypertension_col = col_exists(['hypertension', 'high_bp', 'high_bp'])
heart_col = col_exists(['heart_disease', 'heart disease', 'heart_dz'])
glucose_col = col_exists(['avg_glucose_level', 'avg_glucose', 'average_glucose_level'])
bmi_col = col_exists(['bmi'])
label_col = col_exists(['stroke', 'stroke ']) or 'Stroke'

# If some columns missing, try to guess from dataset (case-insensitive)
if age_col is None:
    raise SystemExit('Age column not found in CSV')

# Build features dataframe with defaults when missing
X = pd.DataFrame()
X['age'] = df[age_col]

if hypertension_col and hypertension_col in df.columns:
    X['hypertension'] = df[hypertension_col].map(lambda v: 1 if str(v).strip().lower() in ['yes', '1', 'true'] else 0)
else:
    X['hypertension'] = 0

if heart_col and heart_col in df.columns:
    X['heart_disease'] = df[heart_col].map(lambda v: 1 if str(v).strip().lower() in ['yes', '1', 'true'] else 0)
else:
    X['heart_disease'] = 0

if glucose_col and glucose_col in df.columns:
    X['avg_glucose_level'] = pd.to_numeric(df[glucose_col], errors='coerce').fillna(df[glucose_col].median())
else:
    X['avg_glucose_level'] = df.iloc[:,0]*0 + df.shape[0]

if bmi_col and bmi_col in df.columns:
    X['bmi'] = pd.to_numeric(df[bmi_col], errors='coerce').fillna(df[bmi_col].median())
else:
    X['bmi'] = df.iloc[:,0]*0 + df.shape[0]

# Label
label_candidates = [c for c in df.columns if c.lower().strip() == 'stroke']
if label_candidates:
    y = df[label_candidates[0]].map(lambda v: 1 if str(v).strip().lower() in ['1','yes','true'] else 0)
else:
    raise SystemExit('Label column "Stroke" not found in CSV')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Evaluate
acc = model.score(X_test_scaled, y_test)
print(f"Trained KNN accuracy on holdout: {acc:.3f}")

# Save artifacts
pickle.dump(model, open(MODEL_OUT, 'wb'))
pickle.dump(scaler, open(SCALER_OUT, 'wb'))
print(f"Saved model -> {MODEL_OUT}")
print(f"Saved scaler -> {SCALER_OUT}")
