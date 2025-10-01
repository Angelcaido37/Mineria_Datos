
# -*- coding: utf-8 -*-
"""
Script 02: Clustering con K-Means y DBSCAN, método del codo y silueta,
reducción de dimensionalidad con PCA y t-SNE, y visualización 2D/3D.

Ejecuta: python 02_clustering_kmeans_dbscan.py
Requiere: loan_recovery_dataset.csv en el mismo directorio.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Cargamos el dataset desde el CSV
from pathlib import Path
csv_path = Path("Unidad III") / "Ejercicio" / "loan_recovery_dataset.csv"
df = pd.read_csv(csv_path)

# Seleccionamos variables relevantes para segmentación (evitamos la etiqueta)
features_num = [
    "age", "monthly_income", "loan_amount", "interest_rate_annual", "term_months",
    "employment_length_years", "credit_score", "dti_percent", "num_late_payments",
    "days_past_due", "app_logins_30d", "sms_open_rate", "email_open_rate",
    "whatsapp_opt_in", "previous_restructuring", "macro_unemployment", "macro_inflation"
]

features_cat = ["employment_status", "region", "last_contact_channel", "has_collateral"]

X = df[features_num + features_cat].copy()

# One-hot para categóricas y escalado para numéricas
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ]
)

X_proc = preprocess.fit_transform(X)

# Método del codo para K
Ks = list(range(2, 11))
inertias = []
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_proc)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(Ks, inertias, marker="o")
plt.xlabel("Número de clústeres (K)")
plt.ylabel("Inercia (Suma de distancias cuadradas)")
plt.title("Método del codo - KMeans")
plt.tight_layout()
plt.show()

# Silueta para algunos K
sil_scores = []
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_proc)
    sil = silhouette_score(X_proc, labels)
    sil_scores.append(sil)

plt.figure()
plt.plot(Ks, sil_scores, marker="o")
plt.xlabel("Número de clústeres (K)")
plt.ylabel("Score de silueta")
plt.title("Silueta por K - KMeans")
plt.tight_layout()
plt.show()

# Elegimos un K razonable (por ejemplo, el de mejor silueta)
best_k = Ks[int(np.argmax(sil_scores))]
kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_proc)
df["cluster_kmeans"] = labels_km

print(f"Elegido K={best_k} por silueta. Distribución de clústeres KMeans:\n", df["cluster_kmeans"].value_counts())

# DBSCAN (parámetros heurísticos)
dbscan = DBSCAN(eps=2.0, min_samples=20)
labels_db = dbscan.fit_predict(X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc)
df["cluster_dbscan"] = labels_db

print("Distribución DBSCAN (label -1 es ruido):\n", df["cluster_dbscan"].value_counts())

# PCA para visualización 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc)

plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_km, s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans en espacio PCA (2D)")
plt.tight_layout()
plt.show()

# t-SNE para visualización 2D (costoso pero útil)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca", learning_rate="auto")
X_tsne = tsne.fit_transform(X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc)

plt.figure()
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=labels_km, s=10)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("KMeans en t-SNE (2D)")
plt.tight_layout()
plt.show()

# Guardamos un resumen por clúster para análisis de negocio
summary = df.groupby("cluster_kmeans")[
    ["loan_amount","interest_rate_annual","dti_percent","days_past_due","default_60d","paid_within_30d"]
].mean().round(2)
print("Resumen por clúster (medias):\n", summary)
summary.to_csv("cluster_summary.csv", index=True)
print("Archivo 'cluster_summary.csv' exportado.")
