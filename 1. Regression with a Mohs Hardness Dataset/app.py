import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('mohs_hardness_regressor.pkl')

# Streamlit app
st.title('Mohs Hardness Prediction')
st.write('Predict the Mohs hardness from Kaggle Competition. Link: [Regression with a Mohs Hardness Dataset](https://www.kaggle.com/competitions/playground-series-s3e25)')
st.image('https://www.kaggle.com/competitions/60892/images/header', width='stretch')

# Input features
col1, col2 = st.columns(2)

with col1:
    density_Total = st.number_input('density_Total', value=5.0, step=0.1, min_value=0.0, max_value=750.0)
    allelectrons_Average = st.number_input('allelectrons_Average', value=50.0, step=1.0, min_value=0.0, max_value=100.0)
    val_e_Average = st.number_input('val_e_Average', value=4.0, step=0.1, min_value=0.0, max_value=8.0)
    atomicweight_Average = st.number_input('atomicweight_Average', value=50.0, step=0.1, min_value=0.0, max_value=200.0)

with col2:
    ionenergy_Average = st.number_input('ionenergy_Average', value=7.0, step=0.1, min_value=0.0, max_value=20.0)
    el_neg_chi_Average = st.number_input('el_neg_chi_Average', value=1.5, step=0.1, min_value=0.0, max_value=5.0)
    R_cov_element_Average = st.number_input('R_cov_element_Average', value=1.2, step=0.1, min_value=0.0, max_value=3.0)
    density_Average = st.number_input('density_Average', value=5.0, step=0.1, min_value=0.0, max_value=15.0)


if st.button('Predict', type='primary', use_container_width=True):
    X = pd.DataFrame({
        'density_Total': [density_Total],
        'allelectrons_Average': [allelectrons_Average],
        'val_e_Average': [val_e_Average],
        'atomicweight_Average': [atomicweight_Average],
        'ionenergy_Average': [ionenergy_Average],
        'el_neg_chi_Average': [el_neg_chi_Average],
        'R_cov_element_Average': [R_cov_element_Average],
        'density_Average': [density_Average]
    })
    pred = model.predict(X)
    st.success(f'Predicted Mohs Hardness: **{float(pred[0]):.3f}**')