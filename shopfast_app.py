import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def clean_text(text):
    return text.strip().lower()
@st.cache_data
def load_intents(csv_path):
    df = pd.read_csv(csv_path)
    if df.columns[0] == 'User_Query,Intent':
        df = df[df.index != 0]  
        df[['User_Query', 'Intent']] = df['User_Query,Intent'].str.split(',', n=1, expand=True)
    elif len(df.columns) == 1:
        df[['User_Query', 'Intent']] = df.iloc[:, 0].str.split(',', n=1, expand=True)
    df['User_Query'] = df['User_Query'].apply(clean_text)
    return df

def predict_intent(user_input, df, vectorizer):
    user_input = clean_text(user_input)
    queries = df['User_Query'].tolist()
    X = vectorizer.fit_transform(queries + [user_input])
    similarities = cosine_similarity(X[-1], X[:-1])
    max_index = similarities.argmax()
    if similarities[0, max_index] > 0.2:
        return df.iloc[max_index]['Intent']
    else:
        return "Sorry, I couldn't understand your request."
st.set_page_config(page_title="ShopFast Mini AI Assistant", layout="centered")
st.title("🛍️ ShopFast Mini AI Assistant")

user_query = st.text_input("Ask a question:")

if user_query:
    try:
        df = load_intents("intent_data.csv")
        vectorizer = TfidfVectorizer()
        intent = predict_intent(user_query, df, vectorizer)
        st.success(f"**Predicted Intent:** {intent}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
