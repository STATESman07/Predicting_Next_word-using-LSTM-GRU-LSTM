# Predicting_Next_word-using-LSTM-GRU-LSTM
This project demonstrates the application of deep learning in natural language processing by building a Next Word Prediction model using Long Short-Term Memory (LSTM) networks. The model is trained on the text of Shakespeare’s Hamlet, which provides a linguistically rich and complex dataset, ideal for language modeling tasks.
# 🧠 Next Word Prediction Using LSTM

## 📚 Overview

This project is a demonstration of a deep learning model built to predict the **next word** in a given sequence using an **LSTM (Long Short-Term Memory)** neural network. It is trained on the text of **Shakespeare's "Hamlet"**, which provides a complex and rich dataset ideal for sequence prediction tasks.

A **Streamlit** web application is also included for real-time interaction with the model.

---

## 📌 Project Pipeline

### 1. Data Collection
- The dataset is derived from the full text of *Hamlet* by William Shakespeare.

### 2. Preprocessing
- Clean the text and remove punctuation.
- Tokenize the words and convert the text into input sequences.
- Pad the sequences to a fixed length.
- Split the data into training and validation sets.

### 3. Model Building
- An **Embedding layer** to represent words in a dense vector space.
- Two stacked **LSTM layers** to learn temporal patterns in sequences.
- A final **Dense layer with Softmax** to predict the probability distribution of the next word.

### 4. Model Training
- Uses categorical crossentropy loss and the Adam optimizer.
- **EarlyStopping** is used to avoid overfitting by monitoring validation loss.

### 5. Model Evaluation
- Tested with input sequences to predict the most likely next word.

### 6. Deployment
- A **Streamlit app** is created for users to input any sequence and get the predicted next word.

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NLTK
- NumPy
- Streamlit
