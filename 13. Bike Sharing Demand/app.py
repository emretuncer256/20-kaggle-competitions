import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('bike_sharing_demand_pipeline.pkl')

# Streamlit app
st.title('🚲 Bike Sharing Demand Prediction')
st.write('Predict the demand for bike sharing based on the given dataset. Link: [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)')

# Input features
st.header('📊 Input Features')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Weather & Environmental')
    
    # Season (1:spring, 2:summer, 3:fall, 4:winter)
    season = st.selectbox(
        'Season',
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}[x]
    )
    
    # Weather (1: Clear, 2: Mist + Cloudy, 3: Light Snow/Rain, 4: Heavy Rain/Snow)
    weather = st.selectbox(
        'Weather',
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: 'Clear', 2: 'Mist + Cloudy', 3: 'Light Snow/Rain', 4: 'Heavy Rain/Snow'}[x]
    )
    
    # Temperature (normalized to 0-1)
    temp = st.slider('Temperature (°C)', min_value=-8.0, max_value=39.0, value=20.0, step=0.1)
    temp_normalized = (temp + 8) / 47  # Normalize to 0-1 range
    
    # "Feels like" temperature (normalized to 0-1)
    atemp = st.slider('Feels Like Temperature (°C)', min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    atemp_normalized = (atemp + 10) / 60  # Normalize to 0-1 range
    
    # Humidity (0-100%)
    humidity = st.slider('Humidity (%)', min_value=0, max_value=100, value=50, step=1)
    humidity_normalized = humidity / 100  # Normalize to 0-1 range
    
    # Windspeed (0-67 km/h)
    windspeed = st.slider('Windspeed (km/h)', min_value=0.0, max_value=67.0, value=10.0, step=0.1)
    windspeed_normalized = windspeed / 67  # Normalize to 0-1 range

with col2:
    st.subheader('Time & Calendar')
    
    # Hour (0-23)
    hour = st.slider('Hour of Day', min_value=0, max_value=23, value=12, step=1)
    
    # Day (1-31)
    day = st.slider('Day of Month', min_value=1, max_value=31, value=15, step=1)
    
    # Month (1-12)
    month = st.selectbox('Month', options=list(range(1, 13)), 
                        format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1])
    
    # Year (2011 or 2012 based on the dataset)
    year = st.selectbox('Year', options=[2011, 2012])
    
    # Day of week (0=Monday, 6=Sunday)
    dayofweek = st.selectbox('Day of Week', options=list(range(7)),
                            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                                 'Friday', 'Saturday', 'Sunday'][x])
    
    st.subheader('Calendar Events')
    
    # Holiday (0: No, 1: Yes)
    holiday = st.radio('Holiday', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
    
    # Working day (0: No, 1: Yes)
    workingday = st.radio('Working Day', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Create input dataframe
input_data = pd.DataFrame({
    'season': [season],
    'holiday': [holiday],
    'workingday': [workingday],
    'weather': [weather],
    'temp': [temp_normalized],
    'atemp': [atemp_normalized],
    'humidity': [humidity_normalized],
    'windspeed': [windspeed_normalized],
    'hour': [hour],
    'day': [day],
    'month': [month],
    'year': [year],
    'dayofweek': [dayofweek]
})

# Display input summary
st.header('📋 Input Summary')
st.dataframe(input_data, use_container_width=True)

# Make prediction
if st.button('🔮 Predict Bike Demand', type='primary', use_container_width=True):
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success(f'🚲 Predicted Bike Demand: **{prediction:.0f} bikes**')
        
        # Add some context
        if prediction < 50:
            st.info('💡 Low demand - Good time for maintenance or special promotions')
        elif prediction < 150:
            st.info('💡 Moderate demand - Normal operations')
        elif prediction < 300:
            st.info('💡 High demand - Consider increasing bike availability')
        else:
            st.info('💡 Very high demand - Peak usage time!')
            
    except Exception as e:
        st.error(f'Error making prediction: {str(e)}')