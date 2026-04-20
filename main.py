from src.student_segmentation.preprocess import load_data, clean_data, scale_data
from src.student_segmentation.model import kmeans_model, hierarchical_model, dbscan_model
from src.student_segmentation.visualize import plot_clusters

from src.anime_recommender.preprocess import load_data as load_anime
from src.anime_recommender.model import build_model
from src.anime_recommender.recommend import recommend

print("\n🚀 Running Student Segmentation...\n")

df = load_data("data/student/03_Clustering_Marketing.csv")
df = clean_data(df)
X, df = scale_data(df)

_, k_labels, k_score = kmeans_model(X)
h_labels, h_score = hierarchical_model(X)
d_labels, d_score = dbscan_model(X)

print("KMeans Score:", k_score)
print("Hierarchical Score:", h_score)
print("DBSCAN Score:", d_score)

plot_clusters(X, k_labels)

print("\n🎬 Running Anime Recommendation...\n")

df_anime, anime = load_anime("data/anime/anime.csv", "data/anime/rating.csv")

similarity = build_model(anime)

result = recommend(anime, similarity, "Naruto")
print("\nTop Recommendations:\n", result)