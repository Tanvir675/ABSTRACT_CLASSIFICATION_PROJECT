import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

with open('.github/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('.github/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Abstract Classification System")
st.write("Paste a single research abstract below to predict its category (Science, Business, or Social Science).")

user_input = st.text_area("Paste abstract here:")

if st.button("Predict Abstract"):
    if user_input.strip() != "":
        cleaned_input = preprocess_text(user_input)
        vectorized_input = tfidf.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        
        st.success(f"Predicted Category: {prediction.upper()}")
    else:
        st.warning("Please enter some text to predict.")
