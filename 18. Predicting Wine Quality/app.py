import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('wine_quality_pipeline.pkl')
encoder = joblib.load('wine_quality_encoder.pkl')

# Streamlit app
st.title('🍷 Predicting Wine Quality')
st.write('Predict the quality of wine based on the given dataset. Link: [Predicting Wine Quality](https://www.kaggle.com/competitions/wine-quality-ordinal)')

# Input features
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider('fixed acidity', min_value=4.0, max_value=16.0, value=7.0, step=0.1)
    volatile_acidity = st.slider('volatile acidity', min_value=0.00, max_value=1.50, value=0.30, step=0.01)
    citric_acid = st.slider('citric acid', min_value=0.00, max_value=1.50, value=0.30, step=0.01)
    residual_sugar = st.slider('residual sugar', min_value=0.5, max_value=20.0, value=2.5, step=0.1)
    chlorides = st.slider('chlorides', min_value=0.000, max_value=0.200, value=0.080, step=0.001)
    free_sulfur_dioxide = st.slider('free sulfur dioxide', min_value=1, max_value=100, value=30, step=1)

with col2:
    total_sulfur_dioxide = st.slider('total sulfur dioxide', min_value=6, max_value=300, value=115, step=1)
    density = st.slider('density', min_value=0.9900, max_value=1.0050, value=0.9968, step=0.0001)
    pH = st.slider('pH', min_value=2.50, max_value=4.50, value=3.30, step=0.01)
    sulphates = st.slider('sulphates', min_value=0.30, max_value=2.00, value=0.60, step=0.01)
    alcohol = st.slider('alcohol', min_value=8.0, max_value=15.0, value=10.0, step=0.1)

if st.button('🔮 Predict Quality', type='primary', use_container_width=True):
    input_df = pd.DataFrame([{
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }])
    prediction = pipeline.predict(input_df)[0]
    prediction = encoder.inverse_transform(prediction.reshape(-1, 1))[0]
    st.metric('Predicted Quality', f"{prediction[0]}", border=True)