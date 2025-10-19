import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('smoker_status_pipeline.pkl')

# Streamlit app
st.title('🚬 Predicating Smoker Status using bio-signals')
st.write('Predict the smoker status using bio-signals. Link: [Predicating Smoker Status using bio-signals](https://www.kaggle.com/competitions/predicating-smoker-status-using-bio-signals)')

# Input features
st.header('📊 Enter Your Bio-signals')

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader('Physical Measurements')
    
    # Age
    age = st.slider('Age (years)', min_value=0, max_value=100, value=30, step=1)
    
    # Height
    height = st.slider('Height (cm)', min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    
    # Weight
    weight = st.slider('Weight (kg)', min_value=30.0, max_value=200.0, value=70.0, step=0.1)
    
    # Waist
    waist = st.slider('Waist (cm)', min_value=50.0, max_value=150.0, value=80.0, step=0.1)
    
    # Eyesight
    eyesight_left = st.slider('Eyesight (Left)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    eyesight_right = st.slider('Eyesight (Right)', min_value=0.0, max_value=2.0, value=1.0, step=0.1)

with col2:
    st.subheader('Blood Pressure & Health Metrics')
    
    # Blood pressure
    systolic = st.slider('Systolic Blood Pressure (mmHg)', min_value=70, max_value=200, value=120, step=1)
    relaxation = st.slider('Diastolic Blood Pressure (mmHg)', min_value=40, max_value=120, value=80, step=1)
    
    # Blood sugar
    fasting_blood_sugar = st.slider('Fasting Blood Sugar (mg/dL)', min_value=50.0, max_value=300.0, value=90.0, step=1.0)
    
    # Lipids
    triglyceride = st.slider('Triglyceride (mg/dL)', min_value=30.0, max_value=500.0, value=100.0, step=1.0)
    hdl = st.slider('HDL (mg/dL)', min_value=20.0, max_value=100.0, value=50.0, step=1.0)
    
    # Other health metrics
    hemoglobin = st.slider('Hemoglobin (g/dL)', min_value=8.0, max_value=20.0, value=14.0, step=0.1)
    serum_creatinine = st.slider('Serum Creatinine (mg/dL)', min_value=0.3, max_value=3.0, value=1.0, step=0.01)
    alt = st.slider('ALT (U/L)', min_value=5.0, max_value=200.0, value=25.0, step=1.0)
    gtp = st.slider('GTP (U/L)', min_value=5.0, max_value=200.0, value=30.0, step=1.0)
    dental_caries = st.slider('Dental Caries', min_value=0.0, max_value=10.0, value=0.0, step=1.0)


# Prediction button
if st.button('🔮 Predict Smoker Status', type='primary'):
    # Convert to DataFrame
    input_df = pd.DataFrame([{
        'age': age,
        'height(cm)': height,
        'weight(kg)': weight,
        'waist(cm)': waist,
        'eyesight(left)': eyesight_left,
        'eyesight(right)': eyesight_right,
        'systolic': systolic,
        'relaxation': relaxation,
        'fasting blood sugar': fasting_blood_sugar,
        'triglyceride': triglyceride,
        'HDL': hdl,
        'hemoglobin': hemoglobin,
        'serum creatinine': serum_creatinine,
        'ALT': alt,
        'Gtp': gtp,
        'dental caries': dental_caries
    }])
    
    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]
    
    # Display results
    st.header('🎯 Prediction Results')
    
    if prediction == 1:
        st.error(f'🚬 **Predicted as SMOKER** (Probability: {prediction_proba[1]:.2%})')
    else:
        st.success(f'🚭 **Predicted as NON-SMOKER** (Probability: {prediction_proba[0]:.2%})')
    
    # Show probability breakdown
    st.subheader('📈 Probability Breakdown')
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        st.metric('🚭 Non-Smoker Probability', f'{prediction_proba[0]:.2%}', border=True)
    
    with prob_col2:
        st.metric('🚬 Smoker Probability', f'{prediction_proba[1]:.2%}', border=True)