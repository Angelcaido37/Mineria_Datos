# -*- coding: utf-8 -*-
# ==============================================================
# 2c. Detección de outliers con algoritmos de ML
#    - Isolation Forest
#    - One-Class SVM
# Dataset: House Prices (Kaggle)
# ==============================================================
# Requisitos: pandas, numpy, scikit-learn, matplotlib
# Ejecuta:
#    python "2c. outliers_ml_isoforest_ocsvm.py"
# Salidas:
#    - housing_outliers_ml.csv
#    - fig_isoforest_scores.png
#    - fig_ocsvm_scores.png
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt

# ---------------- Configuración ----------------
DATA_PATH = r"Datasets/housing_train.csv"
OUTPUT_CSV = "housing_outliers_ml.csv"

CONTAMINATION = 0.05  # fracción esperada de outliers (ajústalo si quieres)
NU = 0.05             # parámetro de OCSVM (aprox. fracción de outliers)

# ---------------- Carga e higiene ----------------
df = pd.read_csv(DATA_PATH)

# Seleccionamos solo columnas numéricas para estos modelos
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X_num = df[num_cols].copy()

# Imputación de faltantes por mediana
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_num)

# Escalado estándar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# ---------------- Isolation Forest ----------------
iso = IsolationForest(
    contamination=CONTAMINATION,
    random_state=42,
    n_estimators=300,
    bootstrap=True
)
iso.fit(X_scaled)
iso_scores = iso.score_samples(X_scaled)    # mayor = más normal
iso_pred = iso.predict(X_scaled)            # -1 outlier, 1 inlier
df["outlier_iso"] = (iso_pred == -1).astype(int)

# Graficar distribución de "anormalidad" (usar -score para ver colas derechas)
plt.figure(figsize=(8,5))
plt.hist(-iso_scores, bins=50)
plt.xlabel("Anormalidad (−score_samples)")
plt.ylabel("Frecuencia")
plt.title("Isolation Forest - Distribución de anormalidad")
plt.tight_layout()
plt.savefig("fig_isoforest_scores.png")
plt.close()

# ---------------- One-Class SVM ----------------
oc = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=NU
)
oc.fit(X_scaled)
oc_pred = oc.predict(X_scaled)              # -1 outlier, 1 inlier
# Para tener una "puntuación", usamos decision_function (más negativo = más anómalo)
oc_scores = oc.decision_function(X_scaled)
df["outlier_ocsvm"] = (oc_pred == -1).astype(int)

plt.figure(figsize=(8,5))
plt.hist(-oc_scores, bins=50)
plt.xlabel("Anormalidad (−decision_function)")
plt.ylabel("Frecuencia")
plt.title("One-Class SVM - Distribución de anormalidad")
plt.tight_layout()
plt.savefig("fig_ocsvm_scores.png")
plt.close()

# ---------------- Resumen y guardado ----------------
n_iso = int(df["outlier_iso"].sum())
n_oc = int(df["outlier_ocsvm"].sum())
n_total = len(df)

print("=== Resumen de outliers (ML) ===")
print(f"Total filas: {n_total}")
print(f"Isolation Forest outliers: {n_iso} ({n_iso/n_total:.2%})")
print(f"One-Class SVM outliers:   {n_oc} ({n_oc/n_total:.2%})")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nArchivo guardado: {OUTPUT_CSV}")
print("Gráficos: fig_isoforest_scores.png, fig_ocsvm_scores.png")
