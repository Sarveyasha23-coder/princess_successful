import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Student Segmentation imports
from src.student_segmentation.preprocess import load_data, clean_data, scale_data
from src.student_segmentation.model import kmeans_model

# Anime Recommender imports
from src.anime_recommender.preprocess import load_data as load_anime
from src.anime_recommender.model import build_model
from src.anime_recommender.recommend import recommend


# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Princess Successful ML App", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("👑 Princess ML App")

menu = st.sidebar.selectbox(
    "Choose Project",
    ["Student Segmentation", "Anime Recommendation"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
This app includes:

🎯 Student Segmentation using Clustering  
🎬 Anime Recommendation using NLP  

Built using Machine Learning & Streamlit
""")

# ---------------- TITLE ----------------
st.title("👑 Princess Successful ML App")

# =========================================================
# 🎯 STUDENT SEGMENTATION
# =========================================================
if menu == "Student Segmentation":

    st.header("🎯 Student Clustering Analysis")

    # Load Data
    df = load_data("data/student/03_Clustering_Marketing.csv")
    df = clean_data(df)
    X, df = scale_data(df)

    # Model
    _, labels, score = kmeans_model(X)

    # Show Score
    st.success(f"Silhouette Score: {round(score, 3)}")

    # Add cluster column
    df['cluster'] = labels

    # ---------------- PCA Visualization ----------------
    st.subheader("📊 Cluster Visualization (PCA)")

    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(comp[:, 0], comp[:, 1], c=labels)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Clusters Visualization")
    st.pyplot(fig)

    # ---------------- Cluster Insights ----------------
    st.subheader("📈 Cluster Insights")

    cluster_summary = df.groupby('cluster').mean(numeric_only=True)
    st.dataframe(cluster_summary)

    # ---------------- Data Preview ----------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())


# =========================================================
# 🎬 ANIME RECOMMENDER
# =========================================================
else:

    st.header("🎬 Anime Recommendation System")

    # Load Data
    df_anime, anime = load_anime(
        "data/anime/anime.csv",
        "data/anime/rating.csv"
    )

    similarity = build_model(anime)

    # Input
    anime_name = st.text_input("Enter Anime Name", "Naruto")

    if st.button("Recommend"):

        result = recommend(anime, similarity, anime_name)

        if isinstance(result, str):
            st.error(result)
        else:
            st.subheader("🔥 Top Recommendations")

            for i, name in enumerate(result):
                st.write(f"{i+1}. {name}")
