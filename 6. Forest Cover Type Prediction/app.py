import streamlit as st
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('forest_cover_type_prediction.pkl')

# Streamlit app
st.title('🌳 Forest Cover Type Prediction')
st.write('Predict the cover type of a forest based on the given dataset. Link: [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction)')
st.image('https://storage.googleapis.com/kaggle-datasets-images/4215545/7271876/e97e3508f549c939f93778433776a2d3/dataset-cover.jpg?t=2023-12-24-11-45-39', width='stretch')

# Input features
st.header('Input Features')

# Continuous features
st.subheader('🗺️ Geographic Features')
col1, col2 = st.columns(2)

with col1:
    elevation = st.number_input('Elevation', min_value=0, max_value=5000, value=2000, step=1)
    aspect = st.number_input('Aspect', min_value=0, max_value=360, value=180, step=1)
    slope = st.number_input('Slope', min_value=0, max_value=90, value=20, step=1)
    horizontal_distance_to_hydrology = st.number_input('Horizontal Distance To Hydrology', min_value=0, max_value=2000, value=200, step=1)
    vertical_distance_to_hydrology = st.number_input('Vertical Distance To Hydrology', min_value=-200, max_value=200, value=0, step=1)
    horizontal_distance_to_roadways = st.number_input('Horizontal Distance To Roadways', min_value=0, max_value=2000, value=300, step=1)

with col2:
    hillshade_9am = st.number_input('Hillshade 9am', min_value=0, max_value=255, value=200, step=1)
    hillshade_noon = st.number_input('Hillshade Noon', min_value=0, max_value=255, value=220, step=1)
    hillshade_3pm = st.number_input('Hillshade 3pm', min_value=0, max_value=255, value=180, step=1)
    horizontal_distance_to_fire_points = st.number_input('Horizontal Distance To Fire Points', min_value=0, max_value=2000, value=400, step=1)

# Wilderness Areas (Boolean)
st.subheader('🐾 Wilderness Areas')
wilderness_cols = st.columns(4)
wilderness_areas = {}
with wilderness_cols[0]:
    wilderness_areas['Wilderness_Area1'] = st.checkbox('Wilderness Area 1')
with wilderness_cols[1]:
    wilderness_areas['Wilderness_Area2'] = st.checkbox('Wilderness Area 2')
with wilderness_cols[2]:
    wilderness_areas['Wilderness_Area3'] = st.checkbox('Wilderness Area 3')
with wilderness_cols[3]:
    wilderness_areas['Wilderness_Area4'] = st.checkbox('Wilderness Area 4')

# Soil Types (Boolean)
st.subheader('🌱 Soil Types')
soil_cols = st.columns(8)
soil_types = {}
for i in range(1, 41):
    col_idx = (i - 1) % 8
    with soil_cols[col_idx]:
        soil_types[f'Soil_Type{i}'] = st.checkbox(f'Soil Type {i}')

if st.button('Predict Cover Type', type='primary', use_container_width=True):
    input_data = pd.DataFrame({
        'Elevation': [elevation],
        'Aspect': [aspect],
        'Slope': [slope],
        'Horizontal_Distance_To_Hydrology': [horizontal_distance_to_hydrology],
        'Vertical_Distance_To_Hydrology': [vertical_distance_to_hydrology],
        'Horizontal_Distance_To_Roadways': [horizontal_distance_to_roadways],
        'Hillshade_9am': [hillshade_9am],
        'Hillshade_Noon': [hillshade_noon],
        'Hillshade_3pm': [hillshade_3pm],
        'Horizontal_Distance_To_Fire_Points': [horizontal_distance_to_fire_points],
        **wilderness_areas,
        **soil_types
    })
    prediction = pipeline.predict(input_data)
    st.metric('Predicted Cover Type', f"{prediction[0]}", border=True)