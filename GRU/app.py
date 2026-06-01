import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
vocab_size = 10000
max_length = 200

# ----------------------------
# BUILD MODELS
# ----------------------------
def build_rnn():
    return Sequential([
        Embedding(vocab_size, 64),
        SimpleRNN(64),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

def build_lstm():
    return Sequential([
        Embedding(vocab_size, 64),
        LSTM(64),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

def build_gru():
    return Sequential([
        Embedding(vocab_size, 64),
        GRU(64),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

# ----------------------------
# INITIALIZE MODELS
# ----------------------------
rnn_model = build_rnn()
lstm_model = build_lstm()
gru_model = build_gru()

# 🔥 IMPORTANT: BUILD MODELS BEFORE LOADING WEIGHTS
rnn_model.build(input_shape=(None, max_length))
lstm_model.build(input_shape=(None, max_length))
gru_model.build(input_shape=(None, max_length))

# ----------------------------
# LOAD WEIGHTS
# ----------------------------
rnn_model.load_weights("rnn_model.weights.h5")
lstm_model.load_weights("lstm_model.weights.h5")
gru_model.load_weights("gru_model.weights.h5")

# ----------------------------
# WORD INDEX
# ----------------------------
word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()
    encoded = []
    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # unknown word
    return encoded

def predict(model, text):
    seq = encode_review(text)
    padded = pad_sequences([seq], maxlen=max_length)
    pred = model.predict(padded, verbose=0)[0][0]
    return pred

# ----------------------------
# UI
# ----------------------------
st.title("🎬 Movie Review Sentiment Analysis System")
st.subheader("Deep Learning Based Sentiment Classification")

review = st.text_area("Enter your movie review here...")

model_option = st.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM", "GRU"]
)

# ----------------------------
# SINGLE MODEL PREDICTION
# ----------------------------
if st.button("Analyze Review"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        if model_option == "SimpleRNN":
            model = rnn_model
        elif model_option == "LSTM":
            model = lstm_model
        else:
            model = gru_model

        prob = predict(model, review)

        sentiment = "Positive" if prob >= 0.5 else "Negative"
        confidence = prob if prob >= 0.5 else 1 - prob

        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence*100:.2f}%")

        # Chart
        pos_prob = prob
        neg_prob = 1 - prob

        st.write("### Probability Distribution")

        fig, ax = plt.subplots()
        ax.bar(["Positive", "Negative"], [pos_prob, neg_prob])
        ax.set_ylabel("Probability")
        ax.set_title("Confidence")

        st.pyplot(fig)

# ----------------------------
# COMPARE ALL MODELS
# ----------------------------
if st.button("Compare All Models"):
    if review.strip() == "":
        st.warning("Enter a review first.")
    else:
        rnn_p = predict(rnn_model, review)
        lstm_p = predict(lstm_model, review)
        gru_p = predict(gru_model, review)

        st.write("### Model Comparison")

        st.write(f"SimpleRNN: {'Positive' if rnn_p>=0.5 else 'Negative'} ({rnn_p:.4f})")
        st.write(f"LSTM: {'Positive' if lstm_p>=0.5 else 'Negative'} ({lstm_p:.4f})")
        st.write(f"GRU: {'Positive' if gru_p>=0.5 else 'Negative'} ({gru_p:.4f})")
