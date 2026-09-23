import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ---------- PAGc:\Users\panch\Desktop\PythonProject\HospitalManagement\loan_analysis.pyE CONFIG ----------
st.set_page_config(page_title="🏦 Bank Loan Approval Dashboard", page_icon="💰", layout="wide")

# ---------- HEADER ----------
st.markdown("""
    <h1 style='text-align: center; color: #1ABC9C;'>🏦 Bank Loan Approval & Analysis Dashboard</h1>
    <p style='text-align: center; color: gray;'>
    A realistic ML + Rule-based Loan Approval Simulator built using <b>Streamlit, Scikit-learn, and Plotly</b>.
    </p>
    <hr>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
try:
    model = pickle.load(open('models/loan_model.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    features = pickle.load(open('models/model_features.pkl', 'rb'))
    model_status = "✅ ML Model Loaded Successfully (with Scaler)"
except Exception as e:
    model = None
    scaler = None
    model_status = f"⚠️ Model not found or error: {e}"

st.info(model_status)

# ---------- SIDEBAR INPUT ----------
st.sidebar.markdown("## 👇 Enter Applicant Details")

age = st.sidebar.slider("Age", 18, 70, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Not Graduate", " Graduate"])
applicantincome = st.sidebar.number_input("Applicant Monthly Income (₹)", min_value=0, value=30000, step=1000)
coapplicantincome = st.sidebar.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=1000)
loanamount = st.sidebar.number_input("Requested Loan Amount (₹)", min_value=0, value=150000, step=1000)
credit_history = st.sidebar.selectbox("Credit History (1 = Good, 0 = Poor)", [1, 0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------- ENCODING ----------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Not Graduate" else 0
property_area_encoded = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ---------- CUSTOM SCORING LOGIC ----------
score = 0

# 1️⃣ Age: increases till 40–45, then decreases smoothly
if age < 22:
    score += 4        # very young = less stable
elif 22 <= age <= 30:
    score += 8        # early working years
elif 31 <= age <= 40:
    score += 12       # most stable earning phase
elif 41 <= age <= 50:
    score += 10       # still strong profile
elif 51 <= age <= 55:
    score += 7        # nearing retirement
else:
    score += 5        # reduced repayment capacity


# 2️⃣ Gender: Female slightly higher (govt priority)
score += 8 if gender == 1 else 10

# 3️⃣ Marital Status: Married = More stable
score += 10 if married == 1 else 8

# 4️⃣ Education: graduate ko clear advantage do
if education == 1:  # Graduate
    if applicantincome >= 60000:
        score += 15   # educated + strong income = premium profile
    elif applicantincome >= 20000:
        score += 11
    else:
        score += 7
else:  # Non-Graduate
    if applicantincome >= 60000:
        score += 7    # high income saves a bit
    elif applicantincome >= 20000:
        score += 4
    else:
        score += 1    # very weak profile




# 5️⃣ Applicant Income
if applicantincome < 20000:
    score += 5
elif 20000 <= applicantincome <= 60000:
    score += 12
else:
    score += 10

# 6️⃣ Coapplicant Income
score += 10 if coapplicantincome > 10000 else 6

# 7️⃣ Loan Amount
if loanamount < 250000:
    score += 12
elif 250000 <= loanamount <= 400000:
    score += 8
else:
    score += 4

# 8️⃣ Credit History
score += 20 if credit_history == 1 else -10

# 9️⃣ Property Area
score += {2: 10, 1: 8, 0: 6}[property_area_encoded]

# ---------- STRICT REJECTION CONDITIONS ----------
reject_reason = None
if credit_history == 0 and education == 0:
    reject_reason = "❌ Poor credit history and not a graduate"
elif education == 0 and applicantincome < 30000:
    reject_reason = "❌ Low income and not a graduate"
elif score < 50:
    reject_reason = "⚠️ Overall profile weak"

# ---------- APPROVAL % ----------
approval_percent = round((score / 120) * 100, 2)
approval_percent = max(0, min(100, approval_percent))

# ---------- FEATURE IMPACT SUMMARY ----------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧩 Feature Impact Summary")
impacts = {
    "Age": age,
    "Gender": 10 if gender == 0 else 8,
    "Marital Status": 10 if married == 1 else 8,
    "Education": 8,
    "Applicant Income": 12 if 20000 <= applicantincome <= 60000 else (10 if applicantincome > 60000 else 5),
    "Coapplicant Income": 10 if coapplicantincome > 10000 else 6,
    "Loan Amount": 12 if loanamount < 250000 else (8 if loanamount <= 400000 else 4),
    "Credit History": 20 if credit_history == 1 else -10,
    "Property Area": {2: 10, 1: 8, 0: 6}[property_area_encoded]
}
for k, v in impacts.items():
    st.sidebar.write(f"**{k}:** {'+' if v > 0 else ''}{v}")

# ---------- PREDICTION ----------
st.markdown("## 🔍 Loan Approval Prediction")

if st.button("Predict Loan Status 🚀"):
    with st.spinner("Evaluating your loan application..."):
        total_income = applicantincome + coapplicantincome
        if applicantincome < 10000:
            st.error("❌ Applicant income too low for loan consideration.")
        elif loanamount > total_income * 8:
            st.error("❌ Requested loan amount too high compared to total income.")
        else:
            if model and scaler:
                features_arr = np.array([[age, gender, married, education, 
                                          applicantincome, coapplicantincome,
                                          loanamount, credit_history, property_area_encoded]])
                scaled_features = scaler.transform(features_arr)
                prediction = model.predict(scaled_features)[0]
                prob = model.predict_proba(scaled_features)[0][1] * 100
                final_status = "Approved" if prediction == 1 else "Rejected"
            else:
                prob = min(max(score + np.random.randint(10, 20), 0), 100)
                if score >= 80:
                    final_status = "Approved"
                elif score >= 60:
                    final_status = "Under Review"
                else:
                    final_status = "Rejected"

            st.markdown("---")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(label="Loan Decision", value=final_status, delta=f"{prob:.2f}% Confidence")

            with col2:
                st.progress(int(prob))
                if final_status == "Approved":
                    st.success(f"✅ Loan Approved! (Confidence: {prob:.2f}%)")
                elif final_status == "Under Review":
                    st.warning(f"🕒 Application Under Review (Confidence: {prob:.2f}%)")
                else:
                    st.error(f"❌ Loan Rejected! (Confidence: {100 - prob:.2f}%)")

            st.markdown("### 📊 Loan Evaluation Breakdown")
            st.write(f"**Approval Score:** `{score}/120`")
            st.write(f"**Total Monthly Income:** ₹{total_income:,}")
            st.write(f"**Loan Amount Requested:** ₹{loanamount:,}")
            st.write(f"**Credit History:** {'Good' if credit_history == 1 else 'Poor'}")

# ---------- FOOTER ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: gray;'>
Developed by <b> Anurag kumar 💻</b> |
Realistic Loan Simulation Dashboard | Streamlit + Scikit-learn + Plotly
</p>
""", unsafe_allow_html=True)
