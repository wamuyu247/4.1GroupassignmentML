# Assignment 2 Training 1 - Complete Analysis
# Generated from Colab notebook

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

# Cell 2: Download and load dataset
path = kagglehub.dataset_download("abdurraziq01/cloud-computing-performance-metrics")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
file_path = os.path.join(path, csv_file)
df = pd.read_csv(file_path)
print(f"Dataset shape: {df.shape}")

# Cell 3: Preprocess data
numeric_df = df.select_dtypes(include=[np.number])
numeric_df_clean = numeric_df.dropna()
print(f"Cleaned shape: {numeric_df_clean.shape}")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df_clean)
print(f"Scaled data shape: {scaled_data.shape}")

# Cell 4: Elbow method
sample_size = 15000
np.random.seed(42)
sample_indices = np.random.choice(len(scaled_data), sample_size, replace=False)
scaled_sample = scaled_data[sample_indices]

inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    print(f"Testing K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=100)
    kmeans.fit(scaled_sample)
    inertias.append(kmeans.inertia_)
    if len(set(kmeans.labels_)) > 1:
        score = silhouette_score(scaled_sample, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

best_k = K_range[np.argmax(silhouette_scores)]
print(f"Optimal K: {best_k}")

# Cell 5: Final clustering
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(scaled_sample)

print(f"✅ Analysis complete! Found {best_k} clusters")
print(f"Silhouette Score: {max(silhouette_scores):.3f}")
