import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("models/Logistic_Regression_heart.pkl")
scaler = joblib.load("models/scaler.pkl")
expected_columns = joblib.load("models/columns.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Prediction ")
st.markdown("Provide the following patient details to check the risk of heart disease:")

st.markdown("---")

# **Personal Info**
st.subheader("👤 Personal Information")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40, help="Patient's age in years")
    sex = st.selectbox("Sex", ["M", "F"], help="Male or Female")
with col2:
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Blood pressure at rest")
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200, help="Cholesterol level")

# **Heart Measurements**
st.subheader("❤️ Heart Measurements")
col3, col4 = st.columns(2)
with col3:
    max_hr = st.slider("Max Heart Rate", 60, 220, 150, help="Maximum heart rate achieved")
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1, help="ST depression induced by exercise")
with col4:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], help="0 = False, 1 = True")
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], help="Y = Yes, N = No")

# **ECG & Chest Pain Info**
st.subheader("🫀 ECG and Chest Pain Info")
col5, col6 = st.columns(2)
with col5:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"], help="Type of chest pain experienced")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], help="Resting electrocardiographic result")
with col6:
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of the peak exercise ST segment")

st.markdown("---")

# Predict button
if st.button("Predict"):
    # Create input dict
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # DataFrame and missing columns
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Display results with styling
    st.markdown("### Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease. Probability: {probability:.2f}")