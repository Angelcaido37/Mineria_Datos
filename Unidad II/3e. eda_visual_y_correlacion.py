# -*- coding: utf-8 -*-
# ==============================================================
# 3e. EDA visual y correlación (Univariado/Bivariado)
# Dataset: House Prices (Kaggle) -> Datasets/housing_train.csv
# Salidas: 
#   - outputs/figs/hist_*.png, box_*.png, pairplot_basico.png, heatmap_corr.png
#   - outputs/csv/top_correlaciones_SalePrice.csv
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Configuración ----------------
DATA_PATH = r"Datasets/housing_train.csv"
OUT_FIGS = r"outputs/figs"
OUT_CSV  = r"outputs/csv"

# Crear carpetas si no existen
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

# ---------------- Carga ----------------
df = pd.read_csv(DATA_PATH)

# ---------------- Tipos ----------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"Columnas numéricas: {len(num_cols)} | categóricas: {len(cat_cols)}")

# ---------------- Univariado: numéricas ----------------
# Histogramas + KDE y boxplots por cada variable numérica (cuidado con la cantidad).
# Para evitar demasiados archivos, limitamos a primeras 12 variables (puedes ajustar).
max_vars = min(12, len(num_cols))
for col in num_cols[:max_vars]:
    # Histograma
    plt.figure(figsize=(7,4))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"Histograma - {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, f"hist_{col}.png"))
    plt.close()

    # Boxplot
    plt.figure(figsize=(5,4))
    plt.boxplot(df[col].dropna(), vert=True)
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, f"box_{col}.png"))
    plt.close()

# ---------------- Bivariado ----------------
# 1) Matriz de correlación (numéricas)
if len(num_cols) > 1:
    corr = df[num_cols].corr(numeric_only=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", square=False)
    plt.title("Matriz de correlación (numéricas)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIGS, "heatmap_corr.png"))
    plt.close()

# 2) Pairplot básico sobre subconjunto de numéricas (para no saturar)
sub_cols = [c for c in num_cols if c.lower() in {"saleprice", "grlivarea", "lotarea", "yearbuilt", "overallqual"}]
if len(sub_cols) >= 2:
    try:
        sns.pairplot(df[sub_cols].dropna())
        plt.savefig(os.path.join(OUT_FIGS, "pairplot_basico.png"))
        plt.close()
    except Exception as e:
        print("Pairplot no generado (posible falta de columnas esperadas). Detalle:", e)

# ---------------- Top correlaciones con SalePrice ----------------
target = "SalePrice"
if target in df.columns:
    corrs = df[num_cols].corr(numeric_only=True)[target].drop(labels=[target]).sort_values(ascending=False)
    top = corrs.head(10).reset_index()
    top.columns = ["variable", "correlacion_con_SalePrice"]
    top.to_csv(os.path.join(OUT_CSV, "top_correlaciones_SalePrice.csv"), index=False)
    print("Top correlaciones guardado en outputs/csv/top_correlaciones_SalePrice.csv")
else:
    print("SalePrice no está en el dataset; se omite ranking de correlaciones.")

print("EDA visual completado. Figuras en outputs/figs.")
