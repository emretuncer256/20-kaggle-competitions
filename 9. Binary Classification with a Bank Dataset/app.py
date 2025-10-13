import streamlit as st
import pandas as pd
import joblib

# Load the pipeline
pipeline = joblib.load('bank_classification_pipeline.pkl')

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
st.title('🏦 Binary Classification with a Bank Dataset')
st.write('Predict the binary classification of the bank dataset. Link: [Binary Classification with a Bank Dataset](https://www.kaggle.com/competitions/playground-series-s5e8)')
st.image('https://www.kaggle.com/competitions/91719/images/header', width='stretch')

# Input features
st.subheader('📊 Input Features')

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    balance = st.slider('Balance', min_value=-10000.0, max_value=100000.0, value=1200.0, step=1.0, help="Customer's account balance")
    duration = st.slider('Duration', min_value=1, max_value=5000, value=250, step=1, help="Duration of the call in seconds")
    job = st.selectbox('Job', categorical_choices.get('job', []), help="Customer's job type")
    marital = st.selectbox('Marital Status', categorical_choices.get('marital', []), help="Customer's marital status")
    education = st.selectbox('Education', categorical_choices.get('education', []), help="Customer's education level")

with col2:
    contact = st.selectbox('Contact', categorical_choices.get('contact', []), help="Communication type")
    month = st.selectbox('Month', categorical_choices.get('month', []), help="Last contact month")
    poutcome = st.selectbox('Previous Outcome', categorical_choices.get('poutcome', []), help="Outcome of previous campaign")
    housing = st.selectbox('Housing', ['no', 'yes'], help="Does the customer have a housing loan?")

if st.button('Predict', type='primary', use_container_width=True):
    try:
        input_df = pd.DataFrame({
            'balance': [balance],
            'duration': [duration],
            'housing': [housing],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'contact': [contact],
            'month': [month],
            'poutcome': [poutcome]
        })
        prediction = pipeline.predict(input_df)
        st.metric('Predicted Binary Classification', f"{prediction[0]}", border=True)
    except Exception as e:
        st.error('Error while predicting the binary classification')
        st.error(e)