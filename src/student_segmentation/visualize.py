import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    plt.scatter(comp[:, 0], comp[:, 1], c=labels)
    plt.title("Student Clusters (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()