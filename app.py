import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# load model
model = load_model("hamlet_model.keras")

# load tokenizer
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# load padding
with open("padding.pickle", "rb") as handle:
    padding = pickle.load(handle)


# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[
            -(max_sequence_len - 1) :
        ]  # Ensure the sequence length matches max_sequence_len-1

    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")

    # Make predictions
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# Streamlit app

st.title("Next Word Prediction App")

# User input
input_text = st.text_input("Enter text:", "")

# Predict the next word
if input_text:
    max_sequence_len = (
        model.input_shape[1] + 1
    )  # max_sequence_len=model.input_shape[1]+1
    predicted_word = predict_next_word(
        model, tokenizer, input_text, max_sequence_len=max_sequence_len
    )

    # make button
    if st.button("Predict"):
        st.write(f"Predicted next word: {predicted_word}")
