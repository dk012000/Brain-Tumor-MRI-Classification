import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score
import matplotlib.pyplot as plt

# Load the cleaned dataset
def load_images(data_path, img_size):
    images = []
    labels = []
    class_labels = os.listdir(data_path)
    for idx, cls in enumerate(class_labels):
        class_path = os.path.join(data_path, cls)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img.flatten())
            labels.append(idx)
    return np.array(images), np.array(labels), class_labels

# Perform PCA for dimensionality reduction
def apply_pca(images, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_images = pca.fit_transform(images)
    return reduced_images, pca

# Perform K-Means Clustering
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans

# Perform EM Clustering (Gaussian Mixture Model)
def em_clustering(data, n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(data)
    return labels, gmm

# Visualization of clusters
def plot_clusters(data, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=15)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Paths to cleaned data
    train_data_path = "cleaned/Training"
    test_data_path = "cleaned/Testing"
    img_size = 64  # Resize images to 64x64 for faster computation

    # Load training and testing data
    train_images, train_labels, class_labels = load_images(train_data_path, img_size)
    test_images, test_labels, _ = load_images(test_data_path, img_size)

    # Apply PCA to reduce to 2D
    reduced_train, pca = apply_pca(train_images, n_components=2)

    # Perform K-Means
    kmeans_labels, kmeans = kmeans_clustering(reduced_train, n_clusters=len(class_labels))

    # Perform EM Clustering
    em_labels, em = em_clustering(reduced_train, n_components=len(class_labels))

    # Evaluate Clustering using Adjusted Rand Index (compare cluster labels to ground truth)
    kmeans_ari = adjusted_rand_score(train_labels, kmeans_labels)
    em_ari = adjusted_rand_score(train_labels, em_labels)

    # Print evaluation metrics
    print(f"K-Means Adjusted Rand Index: {kmeans_ari:.4f}")
    print(f"EM Adjusted Rand Index: {em_ari:.4f}")

    # Visualize Clustering Results
    plot_clusters(reduced_train, kmeans_labels, title="K-Means Clustering")
    plot_clusters(reduced_train, em_labels, title="EM Clustering (Gaussian Mixture)")

    # Project test data using PCA
    reduced_test = pca.transform(test_images)

    # Predict cluster for test data
    kmeans_test_labels = kmeans.predict(reduced_test)
    em_test_labels = em.predict(reduced_test)

    # Print test labels (unsupervised, we interpret clusters)
    print("K-Means Test Clusters:", kmeans_test_labels[:10])
    print("EM Test Clusters:", em_test_labels[:10])

    # Optional: Use test labels (if known) to compute metrics like accuracy
    print("Note: ARI compares clustering with true labels, but clusters might not align directly with classes.")

