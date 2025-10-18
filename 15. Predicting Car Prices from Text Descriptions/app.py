import streamlit as st
import numpy as np
import joblib
import nltk
import neattext as ntx

# Load the pipeline
pipeline = joblib.load('car_price_predictor.pkl')

stemmer = nltk.SnowballStemmer('english')

def process_text(text):
    text = text.lower()
    text = ntx.remove_stopwords(text)
    text = ntx.remove_puncts(text)
    text = ntx.remove_special_characters(text)
    text = ntx.remove_multiple_spaces(text)
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Streamlit app
st.title('🚗 Predicting Car Prices from Text Descriptions')
st.write('Predict the price of a car based on the text description of the car. You can find the competition and the dataset [here](https://www.kaggle.com/competitions/predicting-car-prices-from-text-descriptions-copy).')

# Input feature
text_description = st.text_area('Text Description', help='Enter the text description of the car')

# Predict button
if st.button('🔍 Predict Price', type='primary', use_container_width=True):
    text_description = process_text(text_description)
    prediction = pipeline.predict([text_description])
    prediction = np.expm1(prediction)
    st.metric(label='💰 Predicted Price', value=f'${prediction[0]:.2f}', border=True)