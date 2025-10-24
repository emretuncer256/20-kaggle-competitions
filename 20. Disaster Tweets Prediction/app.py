import streamlit as st
import neattext as ntx
import nltk
import re
import joblib

# Load the pipeline
pipeline = joblib.load('disaster_tweets_pipeline.pkl')

stemmer = nltk.SnowballStemmer('english')
def process_text(text):
    text = text.lower()
    text = ntx.remove_emails(text)
    text = ntx.remove_dates(text)
    text = ntx.remove_hashtags(text)
    text = ntx.remove_phone_numbers(text)
    text = ntx.remove_urls(text)
    text = ntx.remove_userhandles(text)
    text = ntx.remove_custom_pattern(text, r'&amp;')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = ntx.remove_numbers(text)
    text = ntx.remove_stopwords(text)
    text = ntx.remove_multiple_spaces(text)
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Streamlit app
st.title('🔍 Disaster Tweets Prediction')
st.write('Predict the disaster tweets based on the given dataset. Link: [Disaster Tweets Prediction](https://www.kaggle.com/competitions/nlp-getting-started)')

# Input features
st.subheader('📊 Input Features')

text = st.text_area('Enter the text to predict the disaster tweets')
hashtags = st.multiselect(
    'Hashtags',
    ['#earthquake', '#flood', '#wildfire', '#tornado', '#volcano', '#tsunami'],
    default=None,
    placeholder='Select hashtags or enter a new hashtag',
    accept_new_options=True,
    max_selections=3
)

if st.button('Predict Disaster Tweets', type='primary', use_container_width=True):
    text = process_text(text)
    hashtags = ', '.join(hashtags) if hashtags else ''
    input_text = text + ' ' + hashtags
    prediction = pipeline.predict([input_text])[0]
    prediction_proba = pipeline.predict_proba([input_text])[0]

    if prediction == 1:
        st.error(f'🔴 Predicted as disaster tweets (Probability: {prediction_proba[1]:.2%})')
    else:
        st.success(f'✅ Predicted as non-disaster tweets (Probability: {prediction_proba[0]:.2%})')