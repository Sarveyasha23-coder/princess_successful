import streamlit as st
import pandas as pd

from src.student_segmentation.preprocess import load_data, clean_data, scale_data
from src.student_segmentation.model import kmeans_model
from src.anime_recommender.preprocess import load_data as load_anime
from src.anime_recommender.model import build_model
from src.anime_recommender.recommend import recommend

st.title("👑 Princess Successful ML App")

menu = st.sidebar.selectbox("Choose Project", ["Student Segmentation", "Anime Recommendation"])

# ---------------- STUDENT SEGMENTATION ----------------
if menu == "Student Segmentation":
    st.header("🎯 Student Clustering")

    df = load_data("data/student/03_Clustering_Marketing.csv")
    df = clean_data(df)
    X, df = scale_data(df)

    _, labels, score = kmeans_model(X)

    st.write("Silhouette Score:", score)
    st.write("Clustered Data Preview:")
    st.dataframe(df.head())

# ---------------- ANIME RECOMMENDER ----------------
else:
    st.header("🎬 Anime Recommender")

    df_anime, anime = load_anime("data/anime/anime.csv", "data/anime/rating.csv")
    similarity = build_model(anime)

    anime_name = st.text_input("Enter Anime Name", "Naruto")

    if st.button("Recommend"):
        result = recommend(anime, similarity, anime_name)
        st.write(result)