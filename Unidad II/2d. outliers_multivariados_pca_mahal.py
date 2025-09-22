# -*- coding: utf-8 -*-
# ==============================================================
# 2d. Outliers multivariados con PCA + Distancia de Mahalanobis
# Dataset: House Prices (Kaggle)
# ==============================================================
# Requisitos: pandas, numpy, scikit-learn, matplotlib, scipy (opcional)
# Ejecuta:
#    python "2d. outliers_multivariados_pca_mahal.py"
# Salidas:
#    - housing_mahal_pca.csv
#    - fig_pca_mahalanobis.png
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------- Configuración ----------------
DATA_PATH = r"Datasets/housing_train.csv"
OUTPUT_CSV = "housing_mahal_pca.csv"

ALPHA = 0.003  # umbral chi-cuadrado (1-ALPHA) ~ 99.7% para marcar outliers

# ---------------- Carga e higiene ----------------
df = pd.read_csv(DATA_PATH)

# Numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_num = df[num_cols].copy()

# Imputación y escalado
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X_num)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# ---------------- Distancia de Mahalanobis ----------------
# Cálculo manual, estable numéricamente con pseudo-inversa
X = X_scaled
mu = X.mean(axis=0)
Xc = X - mu
# Matriz de covarianzas
cov = np.cov(Xc, rowvar=False)
# Inversa (pseudo-inversa por estabilidad)
cov_inv = np.linalg.pinv(cov)

# d_M(x) = sqrt( (x-mu)^T * cov_inv * (x-mu) )
left = Xc @ cov_inv
d2 = np.einsum("ij,ij->i", left, Xc)  # distancia al cuadrado
d = np.sqrt(d2)

# Umbral por chi-cuadrado con df = n_features
dfree = X.shape[1]
threshold = chi2.ppf(1 - ALPHA, dfree)

is_outlier_mahal = (d2 > threshold).astype(int)

# ---------------- PCA para visualización ----------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Plot: un solo scatter con color por outlier
plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], s=12, alpha=0.8, c=is_outlier_mahal)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA (2D) con outliers por Mahalanobis")
plt.tight_layout()
plt.savefig("fig_pca_mahalanobis.png")
plt.close()

# ---------------- Guardado ----------------
out = df.copy()
out["mahalanobis_d"] = d
out["mahalanobis_d2"] = d2
out["outlier_mahal"] = is_outlier_mahal

out["PCA1"] = X_pca[:,0]
out["PCA2"] = X_pca[:,1]

out.to_csv(OUTPUT_CSV, index=False)

print("=== Mahalanobis + PCA ===")
print(f"Filas totales: {len(out)}")
print(f"Outliers (Mahalanobis): {int(out['outlier_mahal'].sum())}")
print(f"Umbral Chi^2 (df={dfree}, 1-ALPHA={1-ALPHA:.3f}): {threshold:.2f}")
print(f"Archivo guardado: {OUTPUT_CSV}")
print("Gráfico: fig_pca_mahalanobis.png")
