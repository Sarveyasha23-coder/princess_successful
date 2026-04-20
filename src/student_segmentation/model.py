from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

def kmeans_model(X, k=3):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return model, labels, score

def hierarchical_model(X):
    model = AgglomerativeClustering(n_clusters=3)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return labels, score

def dbscan_model(X):
    model = DBSCAN(eps=0.5, min_samples=5)
    labels = model.fit_predict(X)

    mask = labels != -1
    if len(set(labels[mask])) > 1:
        score = silhouette_score(X[mask], labels[mask])
    else:
        score = -1

    return labels, score