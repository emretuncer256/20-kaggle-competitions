import streamlit as st
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('crab_age_prediction_pipeline.pkl')

# Streamlit app
st.title('🦀 Crab Age Prediction')
st.write('Predict the age of crabs based on the given dataset. Link: [Regression with a Crab Age Dataset](https://www.kaggle.com/competitions/playground-series-s3e16)')
st.image('https://www.kaggle.com/competitions/51983/images/header', width='stretch')

# Input features

col1, col2 = st.columns(2)

with col1:
    length = st.slider('Length', min_value=0.01, max_value=5.0, value=1.3, step=0.01)
    height = st.slider('Height', min_value=0.01, max_value=5.0, value=0.35, step=0.01)
    shucked_weight = st.slider('Shucked Weight', min_value=0.01, max_value=60.0, value=10.1, step=0.01)
    shell_weight = st.slider('Shell Weight', min_value=0.01, max_value=40.0, value=6.7, step=0.01)

with col2:
    diameter = st.slider('Diameter', min_value=0.01, max_value=3.0, value=1.0, step=0.01)
    weight = st.slider('Weight', min_value=0.01, max_value=120.0, value=23.4, step=0.01)
    viscera_weight = st.slider('Viscera Weight', min_value=0.01, max_value=30.0, value=5.1, step=0.01)
    sex = st.selectbox('Sex', options=['F', 'I', 'M'], index=0)

if st.button('Predict Age', type='primary', use_container_width=True):
    input_df = pd.DataFrame([{
        'Sex': sex,
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Weight': weight,
        'Shucked Weight': shucked_weight,
        'Viscera Weight': viscera_weight,
        'Shell Weight': shell_weight,
    }])

    prediction = pipeline.predict(input_df)[0]
    st.metric(label='Predicted Age', value=f'{prediction:.2f}')