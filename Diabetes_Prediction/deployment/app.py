import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="supriyasaragade/Diabetes-Prediction", filename="best_diabetes_prediction.joblib")
model = joblib.load(model_path)

# Suppress the warning
st.set_page_config(page_title="Diabetes Prediction", layout="wide", initial_sidebar_state="expanded")
# Processing BMI to Nutritional Status
def calculate_nutritional_status(x): 
    if x == 0.0: 
        return np.nan
    elif x < 18.5: 
        return "Underweight"
    elif x < 25: 
        return "Normal"
    elif x >= 25 and x < 30: 
        return "Overweight"
    elif x >= 30: 
        return "Obese"

st.title("🏥 Diabetes Prediction Model")
st.write("Enter patient information to predict diabetes risk")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, value=20.0)

with col2:
    insulin = st.number_input("Insulin Level (µU/mL)", min_value=0.0, value=80.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age (years)", min_value=18, max_value=120, value=30)
    BMI = st.number_input("BMI", min_value=0.0, value=50.0)
    nutritional_status = calculate_nutritional_status(BMI)

if st.button("🔍 Predict Diabetes Risk"):
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age],
        'Nutritional_Status': [nutritional_status]
    })
    
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ **High Risk of Diabetes**")
        else:
            st.success("✅ **Low Risk of Diabetes**")
    
    with col2:
        st.metric("Confidence", f"{max(prediction_proba)*100:.1f}%")
    
    st.markdown("### Prediction Probability")
    col1, col2 = st.columns(2)
    col1.metric("No Diabetes", f"{prediction_proba[0]*100:.1f}%")
    col2.metric("Diabetes", f"{prediction_proba[1]*100:.1f}%")
