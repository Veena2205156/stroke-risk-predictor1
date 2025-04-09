
import streamlit as st
import numpy as np
import pickle

# Load model and encoder
model = pickle.load(open("stroke_rf_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
st.title("🧠 Stroke Risk Level Predictor")
st.markdown("Enter patient details to predict the risk level of stroke.")

# User Inputs
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ['Yes', 'No'])
work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence = st.selectbox("Residence Type", ['Urban', 'Rural'])
avg_glucose = st.number_input("Average Glucose Level", 50.0, 300.0, step=0.1)
bmi = st.number_input("BMI", 10.0, 60.0, step=0.1)
smoking_status = st.selectbox("Smoking Status", ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

# Mappings for encoded inputs
gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
ever_married_map = {'Yes': 1, 'No': 0}
work_type_map = {'Govt_job': 0, 'children': 1, 'Private': 2, 'Self-employed': 3, 'Never_worked': 4}
residence_map = {'Urban': 1, 'Rural': 0}
smoking_map = {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}

# Final input array
input_data = np.array([[
    gender_map[gender], age, hypertension, heart_disease,
    ever_married_map[ever_married], work_type_map[work_type],
    residence_map[residence], avg_glucose, bmi, smoking_map[smoking_status]
]])

# Predict and show result
if st.button("Predict Stroke Risk"):
    prediction = model.predict(input_data)
    risk_label = le.inverse_transform(prediction)[0]
    
    st.subheader(f"🩺 Predicted Stroke Risk Level: **{risk_label}**")
    if risk_label == "High":
        st.error("⚠️ High Risk! Please consult a doctor.")
    elif risk_label == "Medium":
        st.warning("⚠️ Medium Risk. Monitor health regularly.")
    else:
        st.success("✅ Low Risk. Maintain a healthy lifestyle.")
