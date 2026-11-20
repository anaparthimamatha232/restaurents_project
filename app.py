# app.py - Streamlit frontend for Restaurant Review Sentiment Prediction

import streamlit as st
import tensorflow as tf
import numpy as np
import re
import spacy

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('restaurant_sentiment_model.keras')
    return model

@st.cache_resource
def load_nlp():
    return spacy.load('en_core_web_sm')

model = load_model()
nlp = load_nlp()

import nltk
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

def clean_text(txt):
    if not isinstance(txt, str):
        txt = str(txt)
    txt = txt.lower()
    txt = re.sub(r'\\[nt]*', ' ', txt)
    txt = re.sub(r'[^A-Za-z\s]', ' ', txt)
    txt = re.sub(r'[\s+]', ' ', txt)
    txt = " ".join([x for x in txt.split(' ') if x not in stop_words])
    return txt

def lemmatize_review(txt):
    words = nlp(txt)
    txt = ' '.join(word.lemma_ for word in words)
    return txt

def preprocess_text_for_pred(txt):
    txt = clean_text(txt)
    txt = lemmatize_review(txt)
    return txt

def classify_review(review):
    cleaned_review = preprocess_text_for_pred(review)
    review_array = np.array([cleaned_review], dtype=object)
    prediction = model.predict(review_array, verbose=0)
    
    prob_negative = float(tf.sigmoid(prediction[0][0]))  # sigmoid output = probability of Negative (class 1)
    predicted_class = 1 if prob_negative > 0.5 else 0
    
    if predicted_class == 0:
        confidence = 1 - prob_negative  # confidence of being Positive
    else:
        confidence = prob_negative       # confidence of being Negative
    
    return predicted_class, confidence

st.set_page_config(page_title="Restaurant Review Sentiment Predictor", page_icon="🍽️")
st.title("🍽️ Restaurant Review Sentiment Predictor")
st.write("Enter a customer review below to predict whether it's Positive or Negative.")

review_input = st.text_area("Enter your review:", height=120)

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            predicted_class, confidence = classify_review(review_input)

        if predicted_class == 0:
            st.success("👍 Positive Review")
            st.write(f"Confidence: {confidence*100:.1f}%")
        else:
            st.error("👎 Negative Review")
            st.write(f"Confidence: {(1-confidence)*100:.1f}%")

st.markdown("---")
st.caption("Model: Bidirectional LSTM | Trained on restaurant review dataset")