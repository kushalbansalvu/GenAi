import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np


word_index=imdb.get_word_index()
reverse_word_index= {value: key for key, value in word_index.items()}


#Load the pretrained model with ReLu activation
model = load_model('simple_rnn_imdb.h5')

#Step 2 : Helper functions

#Function to decode reviews
def decode_review(encode_review):
    return ''.join([reverse_word_index.get(i-3, '?') for i in encode_review])

#Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

###Step 3 Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]


###Streamlit app
import streamlit as st

st.title('IMDB Movie review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative')

#User Input

user_input = st.text_area('Moview Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    ####prediction 
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    #Display the result
    st.write(f'SENTIMENT:{sentiment}')
    st.write(f'Prediction:{prediction}')

else:
    st.write('Please enter a moview review.')






