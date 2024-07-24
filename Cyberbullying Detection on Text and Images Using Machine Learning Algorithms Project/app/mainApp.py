import streamlit as st
import pickle
import pytesseract
from PIL import Image
import re

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

model = pickle.load(open('../models/best_model.pkl', 'rb'))

vectorizer = pickle.load(open('../models/tfidf_vectorizer.pkl', 'rb'))

def preprocess_text(text):
    if not text:
        return None 
    text = text.lower()  
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    text = re.sub(r'^RT[\s]+', '', text) 
    return text.strip() 

st.title('CYBERBULLYING/HATE SPEECH PREDICTION')

tweet_input = st.text_input('Enter your tweet')

image = st.file_uploader('Upload an image', type=['jpg', 'png'])

submit = st.button('Predict')

if submit:
    if tweet_input:
        preprocessed_text = preprocess_text(tweet_input)
        vectorized_text = vectorizer.transform([preprocessed_text])
        
        prediction = model.predict(vectorized_text)
        st.write('Prediction for text input:', prediction[0])
        if prediction[0] == 'Negative':
            st.write('Your text contains cyberbullying keywords!')
        elif prediction[0] == 'Positive':
            st.write('Your text is positive and free from cyberbullying content.')
        elif prediction[0] == 'Neutral':
            st.write('Your text is neutral and free from cyberbullying content.')
        else:
            st.write('Your text is irrelevant regarding cyberbullying detection.')

    if image:
        
        st.image(image, caption='Uploaded Image', use_column_width=True)

        extracted_text = pytesseract.image_to_string(Image.open(image))
        
        st.write('Extracted text from the image:', extracted_text)

        preprocessed_text = preprocess_text(extracted_text)
        
        if preprocessed_text:
            vectorized_text = vectorizer.transform([preprocessed_text])
            
            image_prediction = model.predict(vectorized_text)
            st.write('Prediction for image text:', image_prediction[0])
            if image_prediction[0] == 'Negative':
                st.write('The text extracted from the image contains cyberbullying keywords!')
            elif image_prediction[0] == 'Positive':
                st.write('The text extracted from the image is positive and free from cyberbullying content.')
            elif image_prediction[0] == 'Neutral':
                st.write('The text extracted from the image is neutral and free from cyberbullying content.')
            else:
                st.write('The text extracted from the image is irrelevant regarding cyberbullying detection.')
        else:
            st.write('No text extracted from the image. Please try again.')
