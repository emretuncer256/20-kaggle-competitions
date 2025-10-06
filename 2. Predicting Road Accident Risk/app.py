import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('road_accident_risk_pipeline.pkl')

# Function to extract categorical choices from the pipeline
def extract_categorical_choices_from_pipeline(pipeline):
    pre = pipeline.named_steps['preprocessor']  # ColumnTransformer
    choices = {}

    for name, trans, cols in pre.transformers_:
        if trans.__class__.__name__ == 'OneHotEncoder':
            ohe = trans
            for col_name, cats in zip(cols, ohe.categories_):
                choices[col_name] = list(cats)

    return choices

# Load the categorical choices
categorical_choices = extract_categorical_choices_from_pipeline(pipeline)

# Streamlit app
st.title('🛣️ Road Accident Risk Prediction')
st.write('Predict the risk of road accidents based on the given dataset. You can find the competition and the dataset [here](https://www.kaggle.com/competitions/playground-series-s5e10).')
st.image('https://www.kaggle.com/competitions/91721/images/header', width='stretch')

# Input features
col1, col2 = st.columns(2)
with col1:
    road_type = st.selectbox(
        'Road Type',
        categorical_choices['road_type']
    )
    num_lanes = st.number_input('Number of Lanes', min_value=1, max_value=6, value=2, step=1)
    curvature = st.slider('Curvature', min_value=0.0, max_value=1.0, value=0.25, step=0.05, format='%.2f')
    speed_limit = st.slider('Speed Limit (km/h)', min_value=10, max_value=120, value=50, step=5)
    lighting = st.selectbox(
        'Lighting',
        categorical_choices['lighting']
    )
    num_reported_accidents = st.slider('Number of Reported Accidents', min_value=0, max_value=15, value=1, step=1)

with col2:
    weather = st.selectbox(
        'Weather',
        categorical_choices['weather']
    )
    road_signs_present = st.selectbox('Road Signs Present?', ['No', 'Yes'])
    public_road = st.selectbox('Public Road?', ['No', 'Yes'])
    time_of_day = st.selectbox(
        'Time of Day',
        categorical_choices['time_of_day']
    )
    holiday = st.selectbox('Holiday?', ['No', 'Yes'])
    school_season = st.selectbox('School Season?', ['No', 'Yes'])


if st.button('Predict Risk', type='primary', use_container_width=True):
    try:
        input_df = pd.DataFrame({
            'road_type': [road_type],
            'num_lanes': [num_lanes],
            'curvature': [curvature],
            'speed_limit': [speed_limit],
            'lighting': [lighting],
            'weather': [weather],
            'road_signs_present': [1 if road_signs_present == 'Yes' else 0],
            'public_road': [1 if public_road == 'Yes' else 0],
            'time_of_day': [time_of_day],
            'holiday': [1 if holiday == 'Yes' else 0],
            'school_season': [1 if school_season == 'Yes' else 0],
            'num_reported_accidents': [num_reported_accidents],
        })
        prediction = pipeline.predict(input_df)
        st.metric('Predicted Accident Risk (probability)', f"{prediction[0]:.3f}")
    except Exception:
        st.error('Error while predicting the risk')
