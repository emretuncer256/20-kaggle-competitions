import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('academic_success_pipeline.pkl')

# Streamlit app
st.title('🎓 Academic Success Dataset')
st.write('Predict the academic success of students based on the given dataset. Link: [Academic Success Dataset](https://www.kaggle.com/competitions/playground-series-s4e6)')

# Input features
st.header('📊 Student Information')

# Personal Information
st.subheader('Personal Information')
col1, col2 = st.columns(2)

with col1:
    marital_status = st.selectbox('Marital status', options=[1, 2, 3, 4, 5, 6], index=0)
    application_mode = st.slider('Application mode', min_value=1, max_value=20, value=1)
    application_order = st.slider('Application order', min_value=0, max_value=10, value=1)
    course = st.slider('Course', min_value=1, max_value=200, value=1)
    daytime_evening = st.selectbox('Daytime/evening attendance', options=[0, 1], format_func=lambda x: 'Daytime' if x == 1 else 'Evening')
    previous_qualification = st.slider('Previous qualification', min_value=1, max_value=20, value=1)
    previous_qualification_grade = st.slider('Previous qualification (grade)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    nationality = st.slider('Nacionality', min_value=1, max_value=100, value=1)

with col2:
    mothers_qualification = st.slider("Mother's qualification", min_value=1, max_value=50, value=1)
    fathers_qualification = st.slider("Father's qualification", min_value=1, max_value=50, value=1)
    mothers_occupation = st.slider("Mother's occupation", min_value=1, max_value=50, value=1)
    fathers_occupation = st.slider("Father's occupation", min_value=1, max_value=50, value=1)
    admission_grade = st.slider('Admission grade', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    displaced = st.selectbox('Displaced', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    educational_special_needs = st.selectbox('Educational special needs', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    debtor = st.selectbox('Debtor', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Academic Information
st.subheader('Academic Information')
col3, col4 = st.columns(2)

with col3:
    tuition_fees_up_to_date = st.selectbox('Tuition fees up to date', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    gender = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    scholarship_holder = st.selectbox('Scholarship holder', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    age_at_enrollment = st.slider('Age at enrollment', min_value=17, max_value=70, value=20)
    international = st.selectbox('International', options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

with col4:
    unemployment_rate = st.slider('Unemployment rate', min_value=0.0, max_value=30.0, value=10.0, step=0.1)
    inflation_rate = st.slider('Inflation rate', min_value=-5.0, max_value=20.0, value=2.0, step=0.1)
    gdp = st.slider('GDP', min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# First Semester Performance
st.subheader('First Semester Performance')
col5, col6 = st.columns(2)

with col5:
    curricular_units_1st_sem_credited = st.slider('Curricular units 1st sem (credited)', min_value=0, max_value=20, value=0)
    curricular_units_1st_sem_enrolled = st.slider('Curricular units 1st sem (enrolled)', min_value=0, max_value=20, value=6)
    curricular_units_1st_sem_evaluations = st.slider('Curricular units 1st sem (evaluations)', min_value=0, max_value=20, value=6)
    curricular_units_1st_sem_approved = st.slider('Curricular units 1st sem (approved)', min_value=0, max_value=20, value=6)

with col6:
    curricular_units_1st_sem_grade = st.slider('Curricular units 1st sem (grade)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    curricular_units_1st_sem_without_evaluations = st.slider('Curricular units 1st sem (without evaluations)', min_value=0, max_value=20, value=0)

# Second Semester Performance
st.subheader('Second Semester Performance')
col7, col8 = st.columns(2)

with col7:
    curricular_units_2nd_sem_credited = st.slider('Curricular units 2nd sem (credited)', min_value=0, max_value=20, value=0)
    curricular_units_2nd_sem_enrolled = st.slider('Curricular units 2nd sem (enrolled)', min_value=0, max_value=20, value=6)
    curricular_units_2nd_sem_evaluations = st.slider('Curricular units 2nd sem (evaluations)', min_value=0, max_value=20, value=6)
    curricular_units_2nd_sem_approved = st.slider('Curricular units 2nd sem (approved)', min_value=0, max_value=20, value=6)

with col8:
    curricular_units_2nd_sem_grade = st.slider('Curricular units 2nd sem (grade)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
    curricular_units_2nd_sem_without_evaluations = st.slider('Curricular units 2nd sem (without evaluations)', min_value=0, max_value=20, value=0)

# Prediction button
if st.button('🔮 Predict Academic Success', type='primary', use_container_width=True):
    # Convert features to DataFrame
    input_df = pd.DataFrame([{
        'Marital status': marital_status,
        'Application mode': application_mode,
        'Application order': application_order,
        'Course': course,
        'Daytime/evening attendance': daytime_evening,
        'Previous qualification': previous_qualification,
        'Previous qualification (grade)': previous_qualification_grade,
        'Nacionality': nationality,
        "Mother's qualification": mothers_qualification,
        "Father's qualification": fathers_qualification,
        "Mother's occupation": mothers_occupation,
        "Father's occupation": fathers_occupation,
        'Admission grade': admission_grade,
        'Displaced': displaced,
        'Educational special needs': educational_special_needs,
        'Debtor': debtor,
        'Tuition fees up to date': tuition_fees_up_to_date,
        'Gender': gender,
        'Scholarship holder': scholarship_holder,
        'Age at enrollment': age_at_enrollment,
        'International': international,
        'Curricular units 1st sem (credited)': curricular_units_1st_sem_credited,
        'Curricular units 1st sem (enrolled)': curricular_units_1st_sem_enrolled,
        'Curricular units 1st sem (evaluations)': curricular_units_1st_sem_evaluations,
        'Curricular units 1st sem (approved)': curricular_units_1st_sem_approved,
        'Curricular units 1st sem (grade)': curricular_units_1st_sem_grade,
        'Curricular units 1st sem (without evaluations)': curricular_units_1st_sem_without_evaluations,
        'Curricular units 2nd sem (credited)': curricular_units_2nd_sem_credited,
        'Curricular units 2nd sem (enrolled)': curricular_units_2nd_sem_enrolled,
        'Curricular units 2nd sem (evaluations)': curricular_units_2nd_sem_evaluations,
        'Curricular units 2nd sem (approved)': curricular_units_2nd_sem_approved,
        'Curricular units 2nd sem (grade)': curricular_units_2nd_sem_grade,
        'Curricular units 2nd sem (without evaluations)': curricular_units_2nd_sem_without_evaluations,
        'Unemployment rate': unemployment_rate,
        'Inflation rate': inflation_rate,
        'GDP': gdp
    }])
    
    # Make prediction
    prediction = pipeline.predict(input_df)[0]
    
    # Display result
    st.success(f'Predicted Academic Success: {prediction}')