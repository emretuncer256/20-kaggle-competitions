import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('breast_cancer_pipeline.pkl')

# Streamlit app
st.title('🔬 Breast Cancer Classification')
st.write('Predict the breast cancer label (Malignant or Benign) based on the given dataset. Link: [Breast Cancer Classification](https://www.kaggle.com/competitions/breast-cancer-classification-prototype-fall-2025)')

# Input features
st.subheader('📊 Input Features')

# Create tabs for different feature groups
tab1, tab2, tab3 = st.tabs(['Mean Features', 'Standard Error Features', 'Worst Features'])

with tab1:
    st.write('**Mean Features**')
    col1, col2 = st.columns(2)
    
    with col1:
        radius_mean = st.slider('Radius Mean', 6.0, 30.0, 14.0, 0.1)
        texture_mean = st.slider('Texture Mean', 9.0, 40.0, 19.0, 0.1)
        perimeter_mean = st.slider('Perimeter Mean', 43.0, 190.0, 92.0, 0.1)
        area_mean = st.slider('Area Mean', 143.0, 2500.0, 650.0, 1.0)
        smoothness_mean = st.slider('Smoothness Mean', 0.05, 0.17, 0.10, 0.001)
    
    with col2:
        compactness_mean = st.slider('Compactness Mean', 0.02, 0.35, 0.10, 0.001)
        concavity_mean = st.slider('Concavity Mean', 0.0, 0.43, 0.09, 0.001)
        concave_points_mean = st.slider('Concave Points Mean', 0.0, 0.20, 0.05, 0.001)
        symmetry_mean = st.slider('Symmetry Mean', 0.11, 0.30, 0.18, 0.001)
        fractal_dimension_mean = st.slider('Fractal Dimension Mean', 0.05, 0.10, 0.06, 0.001)

with tab2:
    st.write('**Standard Error Features**')
    col1, col2 = st.columns(2)
    
    with col1:
        radius_se = st.slider('Radius SE', 0.1, 2.9, 0.4, 0.01)
        texture_se = st.slider('Texture SE', 0.4, 4.9, 1.2, 0.01)
        perimeter_se = st.slider('Perimeter SE', 0.8, 22.0, 2.9, 0.01)
        area_se = st.slider('Area SE', 6.0, 542.0, 40.0, 1.0)
        smoothness_se = st.slider('Smoothness SE', 0.001, 0.031, 0.007, 0.0001)
    
    with col2:
        compactness_se = st.slider('Compactness SE', 0.002, 0.135, 0.025, 0.001)
        concavity_se = st.slider('Concavity SE', 0.0, 0.40, 0.03, 0.001)
        concave_points_se = st.slider('Concave Points SE', 0.0, 0.05, 0.012, 0.001)
        symmetry_se = st.slider('Symmetry SE', 0.007, 0.079, 0.020, 0.001)
        fractal_dimension_se = st.slider('Fractal Dimension SE', 0.0009, 0.030, 0.004, 0.0001)

with tab3:
    st.write('**Worst Features**')
    col1, col2 = st.columns(2)
    
    with col1:
        radius_worst = st.slider('Radius Worst', 7.9, 36.0, 16.0, 0.1)
        texture_worst = st.slider('Texture Worst', 12.0, 50.0, 25.0, 0.1)
        perimeter_worst = st.slider('Perimeter Worst', 50.0, 250.0, 107.0, 0.1)
        area_worst = st.slider('Area Worst', 185.0, 4250.0, 880.0, 1.0)
        smoothness_worst = st.slider('Smoothness Worst', 0.07, 0.22, 0.13, 0.001)
    
    with col2:
        compactness_worst = st.slider('Compactness Worst', 0.03, 1.06, 0.25, 0.001)
        concavity_worst = st.slider('Concavity Worst', 0.0, 1.25, 0.27, 0.001)
        concave_points_worst = st.slider('Concave Points Worst', 0.0, 0.29, 0.11, 0.001)
        symmetry_worst = st.slider('Symmetry Worst', 0.16, 0.66, 0.29, 0.001)
        fractal_dimension_worst = st.slider('Fractal Dimension Worst', 0.06, 0.21, 0.11, 0.001)

# Prediction button
if st.button('🔮 Predict Cancer Type', type='primary'):
    # Convert features to DataFrame
    df = pd.DataFrame([{
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }])
    
    # Make prediction
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0]
    
    # Display results
    st.subheader('🎯 Prediction Results')
    
    if prediction == 'M':
        st.error(f'**Malignant** (Probability: {probability[1]:.2%})')
        st.write('⚠️ This indicates a malignant (cancerous) tumor.')
    elif prediction == 'B':
        st.success(f'**Benign** (Probability: {probability[0]:.2%})')
        st.write('✅ This indicates a benign (non-cancerous) tumor.')
    
    # Show probability breakdown
    st.write('**Probability Breakdown:**')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('🟢 Benign', f'{probability[0]:.2%}', border=True)
    with col2:
        st.metric('🔴 Malignant', f'{probability[1]:.2%}', border=True)