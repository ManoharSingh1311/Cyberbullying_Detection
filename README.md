# Cyberbullying and Hate Speech Detection

Overview:
This project aims to detect cyberbullying and hate speech in text and images using advanced machine learning techniques. It involves preprocessing data, training models, and deploying a web application for real-time analysis.

Features:
1. **Text and Image Processing**: Utilizes natural language processing (NLP) techniques and Optical Character Recognition (OCR) for extracting and analyzing text from images.
2. **Model Training**: Trains various machine learning models, including Random Forest, SVM, and Gradient Boosting, to classify text as offensive or non-offensive.
3. **Web Application**: Provides an interactive interface for users to input text or upload images and receive real-time predictions on potential cyberbullying or hate speech content.

Technologies Used:
- **Python**: Core programming language for data processing, model training, and application development.
- **Streamlit**: Framework for building the web application interface.
- **Scikit-learn**: Library for machine learning algorithms and model evaluation.
- **Pandas**: Used for data manipulation and analysis.
- **Tesseract OCR**: Tool for extracting text from images.
- **NLTK**: Library for text preprocessing tasks like tokenization, stopword removal, and lemmatization.
- **Matplotlib & Seaborn**: Libraries for data visualization.

Project Structure:
Cyberbullying_Hate_Speech_Detection/
│
├── data/
│ ├── twitter_training.csv
│ ├── preprocessed_training.csv
│ ├── train_data.pkl
│ └── test_data.pkl
│
├── images/
│ ├── img1.jpg
│ └── img2.jpg
│
├── models/
│ ├── best_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── notebooks/
│ └── exploratory_data_analysis.ipynb
│
├── app/
│ └── mainApp.py
│
├── scripts/
│ ├── preprocess.py
│ └── train_models.py
│
├── README.md
└── requirements.txt


Usage:
1. **Clone the repository**: `git clone [repository_url]`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the application**: Navigate to the `app/` directory and run `streamlit run mainApp.py`
4. **Interact with the app**: Input text or upload images to get predictions on whether the content contains cyberbullying or hate speech.

Credits:
- **NLTK**: For text processing tools and resources.
- **Streamlit**: For providing an easy-to-use framework for creating web applications.
- **Tesseract OCR**: For OCR capabilities.


This project aims to develop a machine learning model capable of predicting cyberbullying in social media posts, particularly on Twitter. By analyzing the text content of tweets, the model classifies them as either positive or negative sentiment, with a focus on identifying negative sentiment indicative of cyberbullying. The project includes data preprocessing, model training using various algorithms, exploratory data analysis, model evaluation, and the development of a user-friendly web application for real-time prediction. 

Contact: burathimannu@gmail.com
