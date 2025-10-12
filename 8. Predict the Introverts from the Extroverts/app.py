import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('extrovert_introvert_classifier.pkl')

# Streamlit app
st.title('👥 Predict the Introverts from the Extroverts')
st.write('Predict the introverts from the extroverts based on the given dataset. Link: [Predict the Introverts from the Extroverts](https://www.kaggle.com/competitions/playground-series-s5e7)')
st.image('https://www.kaggle.com/competitions/91718/images/header', width='stretch')

# Input features
st.subheader('📊 Input Features')

col1, col2 = st.columns(2)

with col1:
    # Time spent alone (float64)
    time_spent_alone = st.slider(
        'Time Spent Alone (hours per week)',
        min_value=0.0,
        max_value=15.0,  # 24*7 = 168 hours in a week
        value=3.0,
        step=0.5,
        help='How many hours per week do you spend alone?'
    )
    
    # Stage fear (object - Yes/No)
    stage_fear = st.selectbox(
        'Stage Fear',
        options=['Yes', 'No'],
        help='Do you experience stage fear or performance anxiety?'
    )
    
    # Social event attendance (float64)
    social_event_attendance = st.number_input(
        'Social Event Attendance (events per month)',
        min_value=0,
        max_value=15,
        value=5,
        step=1,
        help='How many social events do you attend per month?'
    )
    
    # Going outside (float64)
    going_outside = st.number_input(
        'Going Outside (times per week)',
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help='How many times per week do you go outside?'
    )

with col2:
    # Drained after socializing (object - Yes/No)
    drained_after_socializing = st.selectbox(
        'Drained After Socializing',
        options=['Yes', 'No'],
        help='Do you feel drained after socializing?'
    )
    
    # Friends circle size (float64)
    friends_circle_size = st.number_input(
        'Friends Circle Size',
        min_value=0,
        max_value=50,
        value=8,
        step=1,
        help='How many close friends do you have?'
    )
    
    # Post frequency (float64)
    post_frequency = st.slider(
        'Post Frequency (posts per week)',
        min_value=0,
        max_value=100,
        value=5,
        step=1,
        help='How many social media posts do you make per week?'
    )

# Create a button to make prediction
if st.button('🔮 Predict Personality Type', type='primary', use_container_width=True):
    # Prepare input data
    input_data = {
        'Time_spent_Alone': time_spent_alone,
        'Stage_fear': stage_fear,
        'Social_event_attendance': social_event_attendance,
        'Going_outside': going_outside,
        'Drained_after_socializing': drained_after_socializing,
        'Friends_circle_size': friends_circle_size,
        'Post_frequency': post_frequency
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]
    
    # Display results
    st.subheader('🎯 Prediction Results')
    
    if prediction == 'Introvert':
        st.success('🔴 **Introvert** - You are predicted to be an introvert!')
    else:
        st.success('🟢 **Extrovert** - You are predicted to be an extrovert!')
    
    # Show confidence scores
    introvert_prob = prediction_proba[1] * 100
    extrovert_prob = prediction_proba[0] * 100
    
    st.write(f'**Confidence Scores:**')
    c1, c2 = st.columns(2)
    c1.metric('😳 Introvert:', round(introvert_prob, 2), border=True)
    c2.metric('🤠 Extrovert:', round(extrovert_prob, 2), border=True)
    
    # Show input summary
    st.subheader('📋 Input Summary')
    for feature, value in input_data.items():
        st.write(f'• **{feature.replace("_", " ").title()}**: {value}')