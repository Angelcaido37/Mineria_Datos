# ==============================================================
# TEMA 2: DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# Dataset: USA Housing dataset (Kaggle)
# ==============================================================

# 1. Importamos librerías necesarias
import pandas as pd          # Manejo de datos en DataFrames
import numpy as np           # Operaciones numéricas
import matplotlib.pyplot as plt  # Visualizaciones básicas
import seaborn as sns        # Visualizaciones estadísticas
from scipy import stats      # Para Z-score

# 2. Cargamos el dataset de entrenamiento
# Este archivo se descarga desde Kaggle: "House Prices - Advanced Regression Techniques"
df = pd.read_csv(r"Datasets/housing_train.csv")

# 3. Revisamos estadísticas básicas de dos variables clave
print("=== Información del dataset ===")
print(df[["SalePrice", "LotArea"]].describe())  
# SalePrice = precio de venta (nuestro objetivo principal)
# LotArea = área del terreno de la casa

# 4. Visualizamos la variable SalePrice con un boxplot y un histograma
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)   # Primer gráfico
sns.boxplot(x=df["SalePrice"])  # Boxplot → muestra cuartiles y outliers
plt.title("Boxplot: SalePrice")

plt.subplot(1,2,2)   # Segundo gráfico
sns.histplot(df["SalePrice"], kde=True, bins=30)  # Histograma con curva de densidad
plt.title("Distribución: SalePrice")

plt.show()

# ==============================================================
# MÉTODO 1: Detección de Outliers con Rango Intercuartílico (IQR)
# ==============================================================

# 5. Calculamos el primer y tercer cuartil
Q1 = df["SalePrice"].quantile(0.25)  # Percentil 25%
Q3 = df["SalePrice"].quantile(0.75)  # Percentil 75%
IQR = Q3 - Q1                        # Rango intercuartílico

# 6. Definimos límites inferior y superior
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\nLímites IQR: {lower:.2f} – {upper:.2f}")

# 7. Filtramos los outliers usando los límites
outliers_iqr = df[(df["SalePrice"] < lower) | (df["SalePrice"] > upper)]
print(f"Outliers detectados (IQR): {len(outliers_iqr)}")

# ==============================================================
# MÉTODO 2: Detección de Outliers con Z-score
# ==============================================================

# 8. Calculamos Z-score (cuántas desviaciones estándar se aleja cada valor de la media)
z_scores = np.abs(stats.zscore(df["SalePrice"]))  

# 9. Marcamos como outliers los valores con Z-score > 3
outliers_z = df[z_scores > 3]
print(f"Outliers detectados (Z-score > 3): {len(outliers_z)}")

# ==============================================================
# VISUALIZACIÓN DE OUTLIERS
# ==============================================================

# 10. Hacemos un scatterplot entre LotArea y SalePrice
# Esto ayuda a ver outliers en dos dimensiones
plt.figure(figsize=(10,6))
sns.scatterplot(x=df["LotArea"], y=df["SalePrice"])
plt.title("Scatterplot: LotArea vs SalePrice (con outliers visibles)")
plt.show()

# ==============================================================
# TRATAMIENTO DE OUTLIERS
# ==============================================================

# 11. Creamos un nuevo DataFrame eliminando los outliers según IQR
df_no_outliers = df[(df["SalePrice"] >= lower) & (df["SalePrice"] <= upper)]

print(f"\nFilas originales: {len(df)}")
print(f"Filas después de eliminar outliers (IQR): {len(df_no_outliers)}")

# 12. Comparamos con boxplots antes y después de eliminar outliers
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(x=df["SalePrice"])
plt.title("Antes (con outliers)")

plt.subplot(1,2,2)
sns.boxplot(x=df_no_outliers["SalePrice"])
plt.title("Después (sin outliers)")

plt.show()
