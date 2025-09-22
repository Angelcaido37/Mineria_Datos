# ==============================================================
# TEMA 3: TRANSFORMACIÓN DE VARIABLES
# Dataset real: House Prices (Kaggle)
# ==============================================================

# 1. Importamos librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn para escalado y codificación
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 2. Cargamos el dataset
df = pd.read_csv(r"Datasets/housing_train.csv")

# 3. Exploramos variables numéricas y categóricas
print("=== Variables numéricas ===")
print(df.select_dtypes(include=[np.number]).columns.tolist())

print("\n=== Variables categóricas ===")
print(df.select_dtypes(include=["object"]).columns.tolist())

# ==============================================================
# TRANSFORMACIONES EN VARIABLES NUMÉRICAS
# ==============================================================

# 4. Visualizamos la distribución original de SalePrice
plt.figure(figsize=(10,5))
sns.histplot(df["SalePrice"], kde=True, bins=30)
plt.title("Distribución original de SalePrice")
plt.show()

# 5. Transformación logarítmica (reduce sesgo en variables sesgadas)
df["SalePrice_log"] = np.log1p(df["SalePrice"])  # log(1+x) para evitar problemas con ceros

plt.figure(figsize=(10,5))
sns.histplot(df["SalePrice_log"], kde=True, bins=30, color="orange")
plt.title("Distribución de SalePrice después de log-transform")
plt.show()

# 6. Escalado Min-Max (lleva valores a rango [0,1])
scaler_minmax = MinMaxScaler()
df["SalePrice_minmax"] = scaler_minmax.fit_transform(df[["SalePrice"]])

print("\nEjemplo de MinMaxScaler (SalePrice):")
print(df[["SalePrice", "SalePrice_minmax"]].head())

# 7. Escalado Estandarizado (media=0, desviación estándar=1)
scaler_std = StandardScaler()
df["SalePrice_std"] = scaler_std.fit_transform(df[["SalePrice"]])

print("\nEjemplo de StandardScaler (SalePrice):")
print(df[["SalePrice", "SalePrice_std"]].head())

# ==============================================================
# TRANSFORMACIONES EN VARIABLES CATEGÓRICAS
# ==============================================================

# 8. Codificación LabelEncoder (convierte categorías a números)
le = LabelEncoder()
df["MSZoning_label"] = le.fit_transform(df["MSZoning"])

print("\nEjemplo de LabelEncoder (MSZoning):")
print(df[["MSZoning", "MSZoning_label"]].head())

# 9. Codificación One-Hot (variables dummies → crea columnas 0/1)
df_onehot = pd.get_dummies(df["MSZoning"], prefix="MSZoning")

print("\nEjemplo de One-Hot Encoding (MSZoning):")
print(df_onehot.head())

# ==============================================================
# COMPARACIÓN FINAL
# ==============================================================

print("\nColumnas nuevas creadas:")
print([col for col in df.columns if "SalePrice" in col or "MSZoning" in col])
