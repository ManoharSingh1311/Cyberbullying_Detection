
#install streamlit: pip install streamlit
# run: streamlit run app.py

import streamlit as st
import pickle
import time
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Define stopwords manually
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

# Load the trained model
model = pickle.load(open('twitter_validation.pkl', 'rb'))

# Load training data from CSV
df_train = pd.read_csv('twitter_validation.csv', header=None, names=['sentiment', 'text'])

# Fit the TfidfVectorizer with training data
vectorizer = TfidfVectorizer(stop_words=stopwords)
vectorizer.fit(df_train['text'])

st.title('CYBERBULLYING PREDICTION WITH RANDOM FOREST CLASSIFIER')
tweet = st.text_input('Enter your tweet')
submit = st.button('Predict')

if submit:
    start = time.time()
    # Preprocess the input text
    processed_tweet = tweet.lower()  # Convert to lowercase
    processed_tweet = re.sub(r'http\S+', '', processed_tweet)  # Remove URLs
    processed_tweet = re.sub(r'<.*?>', '', processed_tweet)  # Remove HTML tags
    processed_tweet = re.sub(r'[^a-zA-Z\s]', '', processed_tweet)  # Remove special characters and punctuations
    processed_tweet = re.sub(r'^RT[\s]+', '', processed_tweet)  # Remove retweet tags

    # Vectorize the preprocessed input text using the same vectorizer as during training
    input_text_vect = vectorizer.transform([processed_tweet])

    # Make prediction
    prediction = model.predict(input_text_vect)
    end = time.time()
    st.write('Prediction time taken:', round(end-start, 2), 'seconds')

    st.write(prediction[0])
    if prediction[0] == "Negative":
        st.write('Please do not write bad comments! It comes under CyberBullying')
