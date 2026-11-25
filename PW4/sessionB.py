import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

wine = load_wine(as_frame=True)
X = wine.data
cultivar = wine.target

print(wine.DESCR)
print(X.describe())

pca_raw = PCA()
pca_raw.fit(X)
print("PC1 explained variance ratio (raw):", pca_raw.explained_variance_ratio_[0])
pc1_loadings = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("PC1 loadings (raw):\n", pc1_loadings)

scaler = StandardScaler()
Z = scaler.fit_transform(X)

pca = PCA()
pca.fit(Z)

evr = pca.explained_variance_ratio_
cumvar = np.cumsum(evr)
print("Explained variance ratios:\n", evr)
print("Cumulative:\n", cumvar)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(evr)+1), evr, label="individual")
ax.plot(range(1, len(evr)+1), cumvar, marker="o", color="red", label="cumulative")
ax.axhline(0.9, color="gray", linestyle="--")
ax.set_xlabel("component")
ax.set_ylabel("explained variance ratio")
ax.legend()
plt.tight_layout()
plt.savefig("figures/B3_scree.png")
plt.close()

scores = pca.transform(Z)

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("figures/B3_pca_plain.png")
plt.close()

inertias_seed = []
for rs in range(8):
    km = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=rs)
    km.fit(Z)
    inertias_seed.append(km.inertia_)
print("Inertias with n_init=1:", inertias_seed)

inertias_k = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km.fit(Z)
    inertias_k.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, 11), inertias_k, marker="o")
ax.set_xlabel("k")
ax.set_ylabel("inertia")
plt.tight_layout()
plt.savefig("figures/B4_elbow.png")
plt.close()

BEST_K = 3
km_best = KMeans(n_clusters=BEST_K, init="k-means++", n_init=10, max_iter=300, random_state=0)
km_best.fit(Z)

print("Cluster sizes:", np.bincount(km_best.labels_))
centers_real = scaler.inverse_transform(km_best.cluster_centers_)
centers_df = pd.DataFrame(centers_real, columns=X.columns)
print(centers_df)

fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=km_best.labels_, cmap="tab10", alpha=0.6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.colorbar(scatter, ax=ax)
plt.tight_layout()
plt.savefig("figures/B5_pca_clusters.png")
plt.close()

def agreement(labels, cultivar):
    table = pd.crosstab(cultivar, labels)
    return table.max(axis=0).sum() / len(cultivar)

km_scaled = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(Z)
ag_scaled = agreement(km_scaled.labels_, cultivar)

km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(X.values)
ag_raw = agreement(km_raw.labels_, cultivar)

Z2 = scores[:, :2]
km_pca2 = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=0).fit(Z2)
ag_pca2 = agreement(km_pca2.labels_, cultivar)

print("Crosstab (scaled):\n", pd.crosstab(cultivar, km_scaled.labels_))
print("Crosstab (raw):\n", pd.crosstab(cultivar, km_raw.labels_))
print("Crosstab (PC1-PC2):\n", pd.crosstab(cultivar, km_pca2.labels_))

agree_df = pd.DataFrame({
    "version": ["scaled", "raw", "pca2"],
    "agreement": [ag_scaled, ag_raw, ag_pca2]
})
print(agree_df)
agree_df.to_csv("results/B_agreement.csv", index=False)

Xtr, Xte, ctr, cte = train_test_split(
    X, cultivar, test_size=0.25, random_state=42, stratify=cultivar)

pipe1 = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
pipe1.fit(Xtr, ctr)
print("KNN full acc:", pipe1.score(Xte, cte))

pipe2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5))
pipe2.fit(Xtr, ctr)
print("KNN PCA2 acc:", pipe2.score(Xte, cte))
