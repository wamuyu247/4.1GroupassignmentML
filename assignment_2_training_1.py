# Assignment 2 Training 1 - Unsupervised Clustering & Dimensionality Reduction
# This script contains the full analysis code

# Cell 1: Imports
!pip install kagglehub pandas numpy matplotlib seaborn scikit-learn

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Cell 2: Download dataset
path = kagglehub.dataset_download("abdurraziq01/cloud-computing-performance-metrics")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
file_path = os.path.join(path, csv_file)
df = pd.read_csv(file_path)

# Cell 3: Preprocess data
numeric_df = df.select_dtypes(include=[np.number])
numeric_df_clean = numeric_df.dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df_clean)

# Cell 4: Elbow method (optimized)
sample_size = 15000
np.random.seed(42)
sample_indices = np.random.choice(len(scaled_data), sample_size, replace=False)
scaled_sample = scaled_data[sample_indices]

inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=100)
    kmeans.fit(scaled_sample)
    inertias.append(kmeans.inertia_)
    if len(set(kmeans.labels_)) > 1:
        score = silhouette_score(scaled_sample, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

best_k_silhouette = K_range[np.argmax(silhouette_scores)]

# Cell 5: Clustering with optimal K
optimal_k = best_k_silhouette
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(scaled_sample)

print(f"✅ Clustering completed with K={optimal_k}")
