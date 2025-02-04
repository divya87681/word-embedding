import streamlit as st
from gensim.models import Word2Vec

# Load trained Word2Vec model
model = Word2Vec.load("word2vec.model")

st.title("🔎 Word Embeddings Explorer")

word = st.text_input("Enter a word to find similar words:")

if word:
    try:
        similar_words = model.wv.most_similar(word, topn=5)
        st.write("Top 5 similar words:")
        for w, score in similar_words:
            st.write(f"{w}: {score:.2f}")
    except KeyError:
        st.write("⚠️ Word not found in vocabulary. Try another one.")
