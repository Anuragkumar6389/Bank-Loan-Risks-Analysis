import pandas as pd
import numpy as np
import random

# ---------- CONFIG ----------
n = 2000  # total number of records

# ---------- GENERATE RANDOM DATA ----------
np.random.seed(42)

data = {
    "Age": np.random.randint(21, 60, n),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Married": np.random.choice(["Yes", "No"], n, p=[0.7, 0.3]),
    "Education": np.random.choice(["Graduate", "Not Graduate"], n, p=[0.75, 0.25]),
    "Dependents": np.random.randint(0, 4, n),
    "ApplicantIncome": np.random.randint(10000, 100000, n),
    "CoapplicantIncome": np.random.randint(0, 40000, n),
    "LoanAmount": np.random.randint(50000, 500000, n),
    "Credit_History": np.random.choice([1, 0], n, p=[0.8, 0.2]),
    "Property_Area": np.random.choice(["Urban", "Rural", "Semiurban"], n, p=[0.4, 0.3, 0.3]),
}

df = pd.DataFrame(data)

# ---------- DEFINE APPROVAL LOGIC ----------
def approve(row):
    score = 0

    # credit history
    if row["Credit_History"] == 1:
        score += 3

    # income vs loan ratio
    total_income = row["ApplicantIncome"] + row["CoapplicantIncome"]
    if total_income / (row["LoanAmount"] + 1) > 0.25:
        score += 2
    elif total_income / (row["LoanAmount"] + 1) > 0.15:
        score += 1

    # education and marital status
    if row["Education"] == "Graduate":
        score += 1
    if row["Married"] == "Yes":
        score += 1

    # property area
    if row["Property_Area"] == "Urban":
        score += 1
    elif row["Property_Area"] == "Semiurban":
        score += 0.5

    # age-based factor
    if 25 <= row["Age"] <= 45:
        score += 1

    # loan approval decision (threshold)
    approval = 1 if score >= 5 else 0
    return approval

df["Loan_Status"] = df.apply(approve, axis=1)

# ---------- BALANCE DATA SLIGHTLY ----------
# ensure roughly 60-70% approvals for realistic dataset
approved = df[df["Loan_Status"] == 1]
rejected = df[df["Loan_Status"] == 0]

if len(approved) > len(rejected) * 1.5:
    approved = approved.sample(len(rejected) * 1.3, random_state=42)
df = pd.concat([approved, rejected]).sample(frac=1).reset_index(drop=True)

# ---------- SAVE FILE ----------
df.to_csv("loan_data.csv", index=False)
print("✅ 2000+ record realistic 'loan_data.csv' generated successfully!")
print("📊 Approved:", df['Loan_Status'].value_counts().to_dict())
print("\n📁 Saved as: loan_data.csv")
