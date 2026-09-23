import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# ---------- LOAD DATA ----------
df = pd.read_csv("loan_data.csv")

print("📄 Columns in CSV:", df.columns.tolist())
print("🔍 Sample Data:\n", df.head())

# ---------- CLEANING ----------
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# ---------- ENCODE CATEGORICAL COLUMNS ----------
encode_cols = ['Gender', 'Married', 'Education', 'Property_Area']
le = LabelEncoder()
for col in encode_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# ---------- RENAME COLUMNS ----------
rename_map = {
    'Loan_Status': 'loan_status',
    'ApplicantIncome': 'ApplicantIncome',
    'CoapplicantIncome': 'CoapplicantIncome',
    'LoanAmount': 'LoanAmount',
    'Credit_History': 'Credit_History',
    'Gender': 'Gender',
    'Married': 'Married',
    'Education': 'Education',
    'Property_Area': 'Property_Area',
    'Age': 'Age'  # extra feature
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# ---------- SELECT FINAL FEATURES (Dependents removed) ----------
features = [
    'Age',                # applicant age
    'Gender',             # male/female
    'Married',            # yes/no
    'Education',          # graduate/non
    'ApplicantIncome',    # applicant monthly income
    'CoapplicantIncome',  # coapplicant income
    'LoanAmount',         # requested loan amount
    'Credit_History',     # 1 good / 0 poor
    'Property_Area'       # rural/semiurban/urban
]

# Keep only columns that exist
features = [f for f in features if f in df.columns]
X = df[features]
y = df['loan_status']

print("\n✅ Final Features used for training:", features)

# ---------- FEATURE SCALING ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------- TRAIN MODEL ----------
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# ---------- EVALUATE ----------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {acc * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------- SAVE MODEL ----------
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model_features.pkl', 'wb') as f:
    pickle.dump(features, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n💾 Model, scaler, and features saved successfully!")
