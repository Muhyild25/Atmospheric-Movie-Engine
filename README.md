# ⚡ Atmospheric Movie Recommendation Engine

An end-to-end Machine Learning web application that recommends movies based on director style, cast, and atmospheric keywords. 

## Features
* **Content-Based Filtering:** Uses NLP (TF-IDF) and Cosine Similarity to analyze movie overviews, genres, keywords, directors, and top cast.
* **ETL Pipeline:** Custom Python script to extract, transform, and load raw JSON-formatted string data from CSVs into a relational SQLite database.
* **Interactive UI:** Built a modern, caching-enabled web interface using Streamlit.

## Technologies Used
* **Python** (Pandas, Scikit-learn, AST)
* **Streamlit** (Web Interface)
* **SQLite** (Database)