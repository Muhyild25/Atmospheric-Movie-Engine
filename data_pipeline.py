"""
Data Pipeline for Atmospheric Movie Engine.
Extracts data from raw TMDB CSV files, transforms JSON columns,
and loads the cleaned feature set into an SQLite database.
"""
import pandas as pd
import ast
import sqlite3

def extract_names(json_str):
    """Extracts 'name' values from a stringified JSON list."""
    names = []
    try:
        for item in ast.literal_eval(json_str):
            names.append(item['name'])
    except:
        pass
    return names

def extract_director(crew_json_str):
    """Extracts the director's name from the stringified crew list."""
    try:
        for item in ast.literal_eval(crew_json_str):
            if item['job'] == 'Director':
                return [item['name']]
    except:
        pass
    return []

def run_pipeline():
    """Executes the ETL (Extract, Transform, Load) process."""
    print("1. Extracting data from raw CSV files...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    print("2. Merging datasets...")
    movies = movies.merge(credits, on='title')

    print("3. Transforming JSON columns and extracting top 3 cast members...")
    movies['genres'] = movies['genres'].apply(extract_names)
    movies['keywords'] = movies['keywords'].apply(extract_names)
    movies['director'] = movies['crew'].apply(extract_director)
    
    # Extract only the top 3 actors for the cast feature
    movies['cast'] = movies['cast'].apply(lambda x: extract_names(x)[:3])

    # Selecting required columns for the NLP model
    selected_columns = ['movie_id', 'title', 'overview', 'genres', 'keywords', 'director', 'cast']
    df = movies[selected_columns].copy()
    
    # Handle missing values
    df['overview'] = df['overview'].fillna('')

    print("4. Formatting data for SQL insertion...")
    df['genres'] = df['genres'].apply(lambda x: " ".join(x))
    df['keywords'] = df['keywords'].apply(lambda x: " ".join(x))
    df['director'] = df['director'].apply(lambda x: " ".join(x))
    df['cast'] = df['cast'].apply(lambda x: " ".join(x))

    print("5. Loading cleaned data into SQLite database...")
    conn = sqlite3.connect('movies.db')
    df.to_sql('cleaned_movies', conn, if_exists='replace', index=False)
    conn.close()
    
    print("Pipeline successfully updated! 'movies.db' is ready.")

if __name__ == "__main__":
    run_pipeline()