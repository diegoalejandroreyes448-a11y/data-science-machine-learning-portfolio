
# ============================================================
# PROJECT: SPOTIFY MUSIC CLUSTERING AND DIMENSIONALITY REDUCTION
# Techniques: StandardScaler, PCA, K-Means, DBSCAN
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score

warnings.filterwarnings("ignore")
np.random.seed(42)


# ============================================================
# 2. CONFIGURE DATA SOURCE
# ============================================================

DATA_PATH = "spotify_tracks.csv"

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]


# ============================================================
# 3. GENERATE SYNTHETIC MUSIC DATA
# ============================================================

def generate_synthetic_data(seed=42):
    """
    Generate synthetic song features representing five
    example music genres. These records are not real Spotify data.
    """

    rng = np.random.default_rng(seed)

    def create_group(n, means, stds, genre):
        data = {}

        for feature in means:
            values = rng.normal(means[feature], stds[feature], n)

            if feature == "loudness":
                values = np.clip(values, -60, 0)
            elif feature in [
                "danceability",
                "energy",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
            ]:
                values = np.clip(values, 0, 1)
            else:
                values = np.clip(values, 0, None)

            data[feature] = values

        data["genre"] = genre
        return pd.DataFrame(data)

    electronic = create_group(
        220,
        {
            "danceability": 0.82, "energy": 0.90, "loudness": -4.0,
            "speechiness": 0.07, "acousticness": 0.02,
            "instrumentalness": 0.35, "liveness": 0.18,
            "valence": 0.60, "tempo": 126, "duration_ms": 210000,
        },
        {
            "danceability": 0.04, "energy": 0.03, "loudness": 1.0,
            "speechiness": 0.02, "acousticness": 0.02,
            "instrumentalness": 0.12, "liveness": 0.05,
            "valence": 0.10, "tempo": 4, "duration_ms": 15000,
        },
        "electronic",
    )

    acoustic = create_group(
        200,
        {
            "danceability": 0.40, "energy": 0.22, "loudness": -13.0,
            "speechiness": 0.035, "acousticness": 0.88,
            "instrumentalness": 0.02, "liveness": 0.12,
            "valence": 0.35, "tempo": 92, "duration_ms": 230000,
        },
        {
            "danceability": 0.05, "energy": 0.05, "loudness": 1.5,
            "speechiness": 0.015, "acousticness": 0.06,
            "instrumentalness": 0.03, "liveness": 0.04,
            "valence": 0.10, "tempo": 10, "duration_ms": 25000,
        },
        "acoustic",
    )

    rap = create_group(
        210,
        {
            "danceability": 0.80, "energy": 0.62, "loudness": -6.5,
            "speechiness": 0.32, "acousticness": 0.06,
            "instrumentalness": 0.0, "liveness": 0.16,
            "valence": 0.50, "tempo": 98, "duration_ms": 195000,
        },
        {
            "danceability": 0.05, "energy": 0.06, "loudness": 1.2,
            "speechiness": 0.06, "acousticness": 0.03,
            "instrumentalness": 0.01, "liveness": 0.05,
            "valence": 0.12, "tempo": 8, "duration_ms": 18000,
        },
        "rap",
    )

    classical = create_group(
        150,
        {
            "danceability": 0.20, "energy": 0.12, "loudness": -20.0,
            "speechiness": 0.035, "acousticness": 0.97,
            "instrumentalness": 0.90, "liveness": 0.14,
            "valence": 0.25, "tempo": 86, "duration_ms": 280000,
        },
        {
            "danceability": 0.04, "energy": 0.04, "loudness": 2.0,
            "speechiness": 0.015, "acousticness": 0.02,
            "instrumentalness": 0.06, "liveness": 0.05,
            "valence": 0.10, "tempo": 12, "duration_ms": 40000,
        },
        "classical",
    )

    rock = create_group(
        200,
        {
            "danceability": 0.50, "energy": 0.85, "loudness": -5.0,
            "speechiness": 0.045, "acousticness": 0.05,
            "instrumentalness": 0.10, "liveness": 0.24,
            "valence": 0.55, "tempo": 120, "duration_ms": 240000,
        },
        {
            "danceability": 0.05, "energy": 0.04, "loudness": 1.0,
            "speechiness": 0.015, "acousticness": 0.03,
            "instrumentalness": 0.06, "liveness": 0.06,
            "valence": 0.12, "tempo": 10, "duration_ms": 20000,
        },
        "rock",
    )

    df = pd.concat(
        [electronic, acoustic, rap, classical, rock],
        ignore_index=True,
    )

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "song_id", [f"song_{i}" for i in range(len(df))])

    return df


# Load real data if the CSV exists; otherwise use synthetic data.
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    print("Real CSV loaded:", df.shape)
else:
    df = generate_synthetic_data()
    print("CSV not found. Synthetic data generated:", df.shape)

print("\nDataset preview:")
print(df.head())


# ============================================================
# 4. VALIDATE AND PREPARE THE DATA
# ============================================================

missing_features = [feature for feature in FEATURES if feature not in df.columns]

if missing_features:
    raise ValueError(
        f"The dataset is missing these required columns: {missing_features}"
    )

# Remove duplicate song IDs when the column is available.
if "song_id" in df.columns:
    df = df.drop_duplicates(subset=["song_id"])

# Convert feature columns to numeric values.
for feature in FEATURES:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

# Remove rows with missing audio features.
df = df.dropna(subset=FEATURES).reset_index(drop=True)

if len(df) < 3:
    raise ValueError("At least three valid songs are required for clustering.")

print("\nDataset dimensions:", df.shape)
print("\nMissing values:")
print(df[FEATURES].isnull().sum())

print("\nFeature statistics:")
print(df[FEATURES].describe())


# ============================================================
# 5. VISUALIZE FEATURE DISTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(18, 6))
axes = axes.flatten()

for i, feature in enumerate(FEATURES):
    axes[i].hist(df[feature], bins=25)
    axes[i].set_title(feature)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

plt.suptitle("Distribution of Music Features")
plt.tight_layout()
plt.show()


# ============================================================
# 6. STANDARDIZE THE FEATURES
# ============================================================

X = df[FEATURES].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features standardized successfully.")


# ============================================================
# 7. APPLY PCA AND SELECT COMPONENTS
# ============================================================

pca_full = PCA(random_state=42)
pca_full.fit(X_scaled)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(7, 4))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o",
)
plt.axhline(0.90, linestyle="--")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

n_components = int(np.argmax(cumulative_variance >= 0.90) + 1)

print(
    f"Number of components required to explain at least 90% "
    f"of the variance: {n_components}"
)

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)


# Create a separate three-component representation for visualization.
pca_visual = PCA(n_components=3, random_state=42)
X_visual = pca_visual.fit_transform(X_scaled)

df["pc1"] = X_visual[:, 0]
df["pc2"] = X_visual[:, 1]
df["pc3"] = X_visual[:, 2]


# ============================================================
# 8. SELECT THE NUMBER OF K-MEANS CLUSTERS
# ============================================================

possible_k = range(2, min(10, len(df)))
inertias = []
silhouette_scores = []

for k in possible_k:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)

    inertias.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(possible_k), inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(possible_k), silhouette_scores, marker="o")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.show()

for k, score in zip(possible_k, silhouette_scores):
    print(f"k = {k}: silhouette score = {score:.3f}")


# ============================================================
# 9. TRAIN THE FINAL K-MEANS MODEL
# ============================================================

# Five clusters are selected to test the five synthetic genres.
# For real data, choose k based on the elbow plot, silhouette score,
# and the practical purpose of the analysis.

final_k = min(5, len(df) - 1)

kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=10,
)

df["kmeans_cluster"] = kmeans.fit_predict(X_pca)

kmeans_score = silhouette_score(X_pca, df["kmeans_cluster"])

print("\nK-Means silhouette score:", round(kmeans_score, 3))
print("\nSongs per K-Means cluster:")
print(df["kmeans_cluster"].value_counts().sort_index())


# ============================================================
# 10. APPLY DBSCAN
# ============================================================

neighbors_count = min(8, len(df) - 1)

neighbors = NearestNeighbors(n_neighbors=neighbors_count)
neighbors.fit(X_pca)

distances, _ = neighbors.kneighbors(X_pca)
sorted_distances = np.sort(distances[:, -1])

plt.figure(figsize=(7, 4))
plt.plot(sorted_distances)
plt.xlabel("Songs Sorted by Neighbor Distance")
plt.ylabel("Distance to Neighbors")
plt.title("DBSCAN Epsilon Selection")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# This is an example parameter. Tune eps using the distance plot
# and the characteristics of the actual dataset.
dbscan = DBSCAN(eps=1.0, min_samples=8)
df["dbscan_cluster"] = dbscan.fit_predict(X_pca)

dbscan_labels = df["dbscan_cluster"]
number_of_dbscan_clusters = len(set(dbscan_labels)) - (
    1 if -1 in set(dbscan_labels) else 0
)
noise_count = int((dbscan_labels == -1).sum())

print("\nDBSCAN clusters found:", number_of_dbscan_clusters)
print("Songs classified as noise:", noise_count)
print("\nDBSCAN cluster sizes:")
print(dbscan_labels.value_counts().sort_index())


# ============================================================
# 11. EVALUATE CLUSTERING AGAINST KNOWN GENRES
# ============================================================

# The genre column is only used for evaluation when it exists.
# It is not included among the features used by either algorithm.

if "genre" in df.columns:
    kmeans_ari = adjusted_rand_score(
        df["genre"],
        df["kmeans_cluster"],
    )

    dbscan_ari = adjusted_rand_score(
        df["genre"],
        df["dbscan_cluster"],
    )

    print("\nAdjusted Rand Index:")
    print("K-Means:", round(kmeans_ari, 3))
    print("DBSCAN:", round(dbscan_ari, 3))

    print("\nK-Means comparison with known genres:")
    print(pd.crosstab(df["genre"], df["kmeans_cluster"]))

    print("\nDBSCAN comparison with known genres:")
    print(pd.crosstab(df["genre"], df["dbscan_cluster"]))


# ============================================================
# 12. VISUALIZE CLUSTERS IN TWO DIMENSIONS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    df["pc1"],
    df["pc2"],
    c=df["kmeans_cluster"],
    cmap="tab10",
    s=18,
)
axes[0].set_title("K-Means Clusters")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(
    df["pc1"],
    df["pc2"],
    c=df["dbscan_cluster"],
    cmap="tab10",
    s=18,
)
axes[1].set_title("DBSCAN Clusters (-1 = Noise)")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()


# ============================================================
# 13. VISUALIZE K-MEANS IN THREE DIMENSIONS
# ============================================================

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    df["pc1"],
    df["pc2"],
    df["pc3"],
    c=df["kmeans_cluster"],
    cmap="tab10",
    s=18,
)

ax.set_title("K-Means Clusters in 3D")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")

plt.tight_layout()
plt.show()


# ============================================================
# 14. ANALYZE THE ORIGINAL FEATURES BY CLUSTER
# ============================================================

cluster_profiles = df.groupby("kmeans_cluster")[FEATURES].mean()

print("\nAverage feature values by K-Means cluster:")
print(cluster_profiles.round(3))

# Normalize the profiles for easier visual comparison.
feature_min = cluster_profiles.min()
feature_range = cluster_profiles.max() - feature_min

normalized_profiles = (
    (cluster_profiles - feature_min)
    .div(feature_range.replace(0, 1), axis=1)
)

print("\nNormalized cluster profiles:")
print(normalized_profiles.round(3))


# ============================================================
# 15. CREATE A RADAR CHART
# ============================================================

categories = FEATURES
angles = np.linspace(
    0,
    2 * np.pi,
    len(categories),
    endpoint=False,
).tolist()

angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(8, 8),
    subplot_kw={"polar": True},
)

colors = cm.tab10(np.linspace(0, 1, final_k))

for cluster_id in normalized_profiles.index:
    values = normalized_profiles.loc[cluster_id].tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        label=f"Cluster {cluster_id}",
    )
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8)
ax.set_yticklabels([])

plt.title("Normalized Music Profiles by Cluster")
plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
plt.tight_layout()
plt.show()


# ============================================================
# 16. COMPARE FEATURES USING BAR CHARTS
# ============================================================

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for i, feature in enumerate(FEATURES):
    axes[i].bar(
        cluster_profiles.index.astype(str),
        cluster_profiles[feature],
    )
    axes[i].set_title(feature)
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel("Average Value")

plt.suptitle("Average Music Features by K-Means Cluster")
plt.tight_layout()
plt.show()


# ============================================================
# 17. EXPORT RESULTS
# ============================================================

df.to_csv(
    "spotify_clustering_results.csv",
    index=False,
    encoding="utf-8-sig",
)

cluster_profiles.to_csv(
    "spotify_cluster_profiles.csv",
    encoding="utf-8-sig",
)

print("\nFiles exported successfully:")
print("- spotify_clustering_results.csv")
print("- spotify_cluster_profiles.csv")


# ============================================================
# END OF PROJECT
# ============================================================
