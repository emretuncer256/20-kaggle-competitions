import streamlit as st
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('machine_failure_prediction.pkl')

# Streamlit app
st.title('🎛️ Machine Failure Prediction')
st.write('Predict the failure of a machine based on the given dataset. Link: [Binary Classification of Machine Failures](https://www.kaggle.com/competitions/playground-series-s3e17)')
st.image('https://www.kaggle.com/competitions/53376/images/header', width='stretch')

# Input features
st.subheader('🔧 Machine Parameters')

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.write("**Temperature & Speed Parameters**")
    air_temp = st.slider(
        'Air temperature [K]', 
        min_value=200.0, 
        max_value=750.0, 
        value=310.0, 
        step=0.1,
        help="Air temperature in Kelvin"
    )
    
    process_temp = st.slider(
        'Process temperature [K]', 
        min_value=250.0, 
        max_value=450.0, 
        value=300.0, 
        step=0.1,
        help="Process temperature in Kelvin"
    )
    
    rotational_speed = st.slider(
        'Rotational speed [rpm]', 
        min_value=1000, 
        max_value=5000, 
        value=1100, 
        step=1,
        help="Rotational speed in revolutions per minute"
    )
    
    torque = st.slider(
        'Torque [Nm]', 
        min_value=1.0, 
        max_value=120.0, 
        value=45.0, 
        step=0.1,
        help="Torque in Newton meters"
    )
    
    tool_wear = st.slider(
        'Tool wear [min]', 
        min_value=0, 
        max_value=500, 
        value=120, 
        step=1,
        help="Tool wear in minutes"
    )

with col2:
    st.write("**Failure Indicators**")
    st.write("Check the boxes for any detected failures:")
    
    twf = st.checkbox(
        'TWF (Tool Wear Failure)', 
        value=False,
        help="Tool Wear Failure detected"
    )
    
    hdf = st.checkbox(
        'HDF (Heat Dissipation Failure)', 
        value=False,
        help="Heat Dissipation Failure detected"
    )
    
    pwf = st.checkbox(
        'PWF (Power Failure)', 
        value=False,
        help="Power Failure detected"
    )
    
    osf = st.checkbox(
        'OSF (Overstrain Failure)', 
        value=False,
        help="Overstrain Failure detected"
    )
    
    rnf = st.checkbox(
        'RNF (Random Failure)', 
        value=False,
        help="Random Failure detected"
    )

# Create prediction button
if st.button('Predict Machine Failure', type='primary', use_container_width=True):
    input_df = pd.DataFrame([{
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rotational_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'TWF': int(twf),
        'HDF': int(hdf),
        'PWF': int(pwf),
        'OSF': int(osf),
        'RNF': int(rnf)
    }])
    
    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]
    
    # Display results
    st.subheader('Prediction Results')
    
    if prediction == 1:
        st.error(f'⚠️ **Machine Failure Predicted**')
        st.write(f'Probability of failure: {prediction_proba[1]:.2%}')
    else:
        st.success(f'✅ **No Machine Failure Predicted**')
        st.write(f'Probability of failure: {prediction_proba[1]:.2%}')
    
    # Show detailed probabilities
    st.write("**Detailed Probabilities:**")
    col_prob1, col_prob2 = st.columns(2)
    with col_prob1:
        st.metric("🟢 No Failure", f"{prediction_proba[0]:.1%}", border=True)
    with col_prob2:
        st.metric("🔴 Failure", f"{prediction_proba[1]:.1%}", border=True)