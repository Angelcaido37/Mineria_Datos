# ==============================================================
# TEMA 3: TRANSFORMACIÓN DE VARIABLES (con recomendaciones automáticas)
# Dataset: House Prices (Kaggle)
# ==============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv(r"Datasets/housing_train.csv")

# 1. Separar numéricas y categóricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print("=== VARIABLES NUMÉRICAS ===")
print(numeric_cols.tolist())
print("\n=== VARIABLES CATEGÓRICAS ===")
print(categorical_cols.tolist())

# 2. Medir asimetría (skewness) en variables numéricas
skewness = df[numeric_cols].skew().sort_values(ascending=False)
print("\n=== Skewness (asimetría) ===")
print(skewness)

# 3. Reglas de decisión para sugerir transformaciones
print("\n=== Recomendaciones de Transformación ===")
for col in numeric_cols:
    skew_val = df[col].skew()
    if abs(skew_val) > 1:  # muy sesgada
        print(f"{col}: sesgada (skew={skew_val:.2f}) → aplicar log-transform o Box-Cox")
    elif abs(skew_val) > 0.5:  # moderadamente sesgada
        print(f"{col}: moderadamente sesgada (skew={skew_val:.2f}) → considerar log-transform")
    else:
        print(f"{col}: distribución ~normal (skew={skew_val:.2f}) → no necesita transformación")

print("\n=== Variables categóricas ===")
print("Usar OneHotEncoder o LabelEncoder según el modelo")

# 4. Visualización opcional de una variable ejemplo
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.histplot(df["SalePrice"], kde=True, bins=30, color="blue")
plt.title("SalePrice original")

plt.subplot(1,2,2)
sns.histplot(np.log1p(df["SalePrice"]), kde=True, bins=30, color="orange")
plt.title("SalePrice con log-transform")
plt.show()
