import streamlit as st

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)
with st.sidebar:
    st.title("🤖 Chatbot")
    st.write("Machine Learning Chatbot")
    st.write("Model : Logistic Regression")
    st.write("Made by Anant Yadav")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.rerun()
    
import pandas as pd
import pickle
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("bot.csv")

# Load saved model and vectorizer
with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def get_response(user_input):
    clean = clean_text(user_input)
    vector = vectorizer.transform([clean])
    intent = model.predict(vector)[0]
    response = df[df["intent"] == intent]["response"].values[0]
    return response

st.title("🤖 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
user_input = st.chat_input("Type your message")

if user_input:
    st.session_state.messages.append(("user", user_input))

    reply = get_response(user_input)
    st.session_state.messages.append(("bot", reply))

for sender, msg in st.session_state.messages:
    if sender == "user":
        with st.chat_message("user"):
            st.write(msg)
    else:
        with st.chat_message("assistant"):
            st.write(msg)
