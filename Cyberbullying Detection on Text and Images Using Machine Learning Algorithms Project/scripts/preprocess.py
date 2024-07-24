import os
import re
import pickle
import pandas as pd
import nltk
from PIL import Image
import pytesseract
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

stopwords = list(nltk_stopwords.words('english'))

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 

def preprocess_text(text):

    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'^RT[\s]+', '', text)
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stopwords])
    return text

def preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((256, 256))
        extracted_text = pytesseract.image_to_string(img)
        preprocessed_text = preprocess_text(extracted_text)
        return preprocessed_text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

def preprocess_csv_data(input_file, output_file):
    df = pd.read_csv(input_file, usecols=[2, 3], header=None, names=['sentiment', 'text'])
    df.dropna(inplace=True)
    df = df[df['text'].apply(len) > 1]
    df['text'] = df['text'].apply(preprocess_text)
    df.to_csv(output_file, index=False)
    return df

def save_vectorizer(vectorizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(vectorizer, f)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def main():
    try:
        input_file = os.path.join('..', 'data', 'twitter_training.csv')
        output_file = os.path.join('..', 'data', 'preprocessed_twitter_training.csv')
        
        df = preprocess_csv_data(input_file, output_file)
        
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
        
        vectorizer = TfidfVectorizer(stop_words=stopwords)
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)
        
        vectorizer_filename = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')
        save_vectorizer(vectorizer, vectorizer_filename)
        
        train_data_filename = os.path.join('..', 'data', 'train_data.pkl')
        test_data_filename = os.path.join('..', 'data', 'test_data.pkl')
        save_data((X_train_vect, y_train), train_data_filename)
        save_data((X_test_vect, y_test), test_data_filename)
        
        print("Preprocessing completed and data saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
