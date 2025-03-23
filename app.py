import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Title of the app
st.title("Heart Failure Prediction")

# Input fields
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase Level", min_value=0, max_value=8000, value=500)
diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=40)
high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", [0, 1])
platelets = st.number_input("Platelet Count", min_value=0.0, max_value=900000.0, value=250000.0)
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, max_value=10.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium", min_value=100, max_value=160, value=140)
sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
smoking = st.selectbox("Smoking (0: No, 1: Yes)", [0, 1])

# Convert inputs into an array for prediction
input_features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking]])

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        st.error("The patient is at risk of heart failure.")
    else:
        st.success("The patient is NOT at risk of heart failure.")
