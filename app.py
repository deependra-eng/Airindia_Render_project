import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained Logistic Regression model and TF-IDF vectorizer
loaded_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # Load vectorizer

st.title("Text Classification with Logistic Regression")

user_input = st.text_area("Enter Text for Classification:", "")

if st.button("Classify"):
    if user_input:
        # Preprocess text using the same methods as in your training
        input_tfidf = vectorizer.transform([user_input])  # Vectorize input text

        # Use your model to make predictions
        prediction = loaded_model.predict(input_tfidf)
        
        # Display the prediction with some styling
        if prediction[0] == 1:
            st.success("The text is classified as Positive!")
        else:
            st.error("The text is classified as Negative!")
    else:
        st.warning("Please enter some text.")
