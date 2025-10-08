import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('bpm_pipeline.pkl')

# Streamlit app
st.title(':musical_score: Beats per Minute Prediction')
st.write('Predict the beats per minute of a song based on the given dataset. Link: [Predicting the Beats-per-Minute of Songs](https://www.kaggle.com/competitions/playground-series-s5e9)')
st.image('https://www.kaggle.com/competitions/91720/images/header', width='stretch')

# Input features
col1, col2 = st.columns(2)

with col1:
    rhythm_score = st.slider('RhythmScore', min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    audio_loudness = st.slider('AudioLoudness (dB)', min_value=-60.0, max_value=5.0, value=-8.0, step=0.1)
    vocal_content = st.slider('VocalContent', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    acoustic_quality = st.slider('AcousticQuality', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    instrumental_score = st.slider('InstrumentalScore', min_value=0.0, max_value=1.0, value=0.1, step=0.01)

with col2:
    live_performance_likelihood = st.number_input('LivePerformanceLikelihood', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    mood_score = st.number_input('MoodScore', min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    track_duration_ms = st.slider('TrackDurationMs', min_value=30000.0, max_value=10000000.0, value=250000.0, step=500.0, format='%f')
    energy = st.slider('Energy', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button('Predict BPM', type='primary', use_container_width=True):
    X_input = pd.DataFrame({
        'RhythmScore': [rhythm_score],
        'AudioLoudness': [audio_loudness],
        'VocalContent': [vocal_content],
        'AcousticQuality': [acoustic_quality],
        'InstrumentalScore': [instrumental_score],
        'LivePerformanceLikelihood': [live_performance_likelihood],
        'MoodScore': [mood_score],
        'TrackDurationMs': [track_duration_ms],
        'Energy': [energy]
    })
    prediction = pipeline.predict(X_input)[0]
    st.metric('Predicted BeatsPerMinute', f"{prediction:.2f}")