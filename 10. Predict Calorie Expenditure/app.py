import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('calorie_prediction_pipeline.pkl')

# Streamlit app
st.title('🍔 Predict Calorie Expenditure')
st.write('Predict the calorie expenditure of a person based on the given dataset. Link: [Predict Calorie Expenditure](https://www.kaggle.com/competitions/playground-series-s5e5)')
st.image('https://www.kaggle.com/competitions/91716/images/header', width='stretch')

# Input features
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox('Sex', options=['female', 'male'], index=0)
    age = st.slider('Age', min_value=18, max_value=80, value=30, step=1)
    duration = st.slider('Duration', min_value=1.0, max_value=60.0, value=15.0, step=0.5)

with col2:
    heart_rate = st.slider('Heart_Rate', min_value=65, max_value=150, value=80, step=1)
    body_temp = st.slider('Body_Temp', min_value=36.0, max_value=42.0, value=37.8, step=0.1)

if st.button('Predict Calories', type='primary', use_container_width=True):
    input_df = pd.DataFrame([{
        'Sex': sex,
        'Age': age,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp
    }])
    prediction = pipeline.predict(input_df)[0]
    st.metric(label='Predicted Calories', value=f'{prediction:.2f}')