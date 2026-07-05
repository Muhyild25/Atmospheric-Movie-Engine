"""
Atmospheric Movie Recommendation Engine
Developed by Muhammed Yıldırım using Machine Learning & Data Engineering technologies.
This app uses a Content-Based Filtering approach (TF-IDF & Cosine Similarity) 
to recommend movies based on director style, cast, and atmospheric keywords.
"""

import streamlit as st
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_model():
    """
    Extracts data from the SQLite database, performs feature engineering,
    and initializes the NLP vectorizer and similarity matrix.
    Cached by Streamlit to optimize performance and prevent reloading.
    """
    conn = sqlite3.connect('movies.db')
    movies = pd.read_sql_query("SELECT * FROM cleaned_movies", conn)
    conn.close()
    
    # Feature Engineering: Weighting director and cast for better atmospheric match
    movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + \
                     movies['keywords'] + " " + (movies['director'] + " ") * 3 + \
                     (movies['cast'] + " ") * 2
                     
    # Vectorize text features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = vectorizer.fit_transform(movies['tags']).toarray()
    
    # Calculate Cosine Similarity distances
    similarity = cosine_similarity(vectors)
    
    return movies, similarity

# Initialize the data and NLP model components
movies, similarity = load_model()

def recommend_movie(movie_title):
    """
    Calculates the top 5 most similar movies based on the cosine similarity matrix.
    """
    try:
        # Locate the index of the queried movie (case-insensitive)
        movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        distances = similarity[movie_index]
        
        # Retrieve the top 5 closest matches (excluding the movie itself)
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recommendations = []
        for i in movie_list:
            recommendations.append(movies.iloc[i[0]].title)
        return recommendations
    except IndexError:
        return []

# --- WEB UI CONFIGURATION ---
st.set_page_config(page_title="Yıldırım Film Motoru", page_icon="⚡")

st.title("⚡ Yıldırım Film Motoru ⚡")
st.markdown("Sevdiğiniz bir filmi seçin, yapay zeka size **yönetmen tarzı** ve **atmosferi** en çok benzeyen yapımları önersin.")

st.divider()

# User Input Section
movie_titles = movies['title'].values
selected_movie = st.selectbox("Bir film seçin veya adını yazın:", movie_titles)

# Recommendation Execution
if st.button("Bana Film Öner"):
    if selected_movie:
        st.subheader(f"'{selected_movie}' Sevenler İçin Öneriler:")
        
        results = recommend_movie(selected_movie)
        
        if results:
            for film in results:
                st.write(f"🎬 {film}")
        else:
            st.error("Bir hata oluştu veya film bulunamadı.")

st.divider()
st.caption("⚡ Muhammed Yıldırım tarafından Yapay Zeka & Veri Mühendisliği teknolojileriyle geliştirilmiştir.")