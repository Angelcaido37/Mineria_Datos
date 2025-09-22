# ==============================================================
# TEMA 1: TRATAMIENTO DE VALORES FALTANTES
# Dataset real: Titanic (Kaggle)
# ==============================================================

# 1. Importamos librerías necesarias
import pandas as pd  # Para manipulación de datos
import numpy as np   # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización básica
import seaborn as sns  # Para visualizaciones estadísticas

from sklearn.impute import SimpleImputer  # Imputación simple
from sklearn.impute import KNNImputer     # Imputación KNN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  # Imputación MICE (multivariada)

# 2. Cargamos el dataset Titanic (de Kaggle)
# Asegúrate de que tengas el archivo "tested.csv" en tu carpeta de trabajo
df = pd.read_csv(r"Datasets\tested.csv")

print("=== INFORMACIÓN DEL DATASET ===")
print(df.info())  # Muestra cuántos valores nulos hay por columna
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# 3. Análisis inicial de valores faltantes
missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_df = pd.DataFrame({
    "Valores_Faltantes": missing_count,
    "Porcentaje (%)": missing_percent
}).sort_values(by="Valores_Faltantes", ascending=False)

print("\n=== RESUMEN DE VALORES FALTANTES ===")
print(missing_df)

# 4. Visualización de valores faltantes
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Mapa de Valores Faltantes en Titanic")
plt.show()

# 5. Estrategia 1: Eliminación
print("\n--- MÉTODO 1: ELIMINACIÓN ---")
df_dropna = df.dropna()  # Elimina todas las filas con al menos un NaN
print(f"Filas originales: {len(df)}")
print(f"Filas después de eliminación: {len(df_dropna)}")

# 6. Estrategia 2: Imputación Simple
print("\n--- MÉTODO 2: IMPUTACIÓN SIMPLE ---")

df_simple = df.copy()

# Para variables numéricas: rellenamos con la MEDIA
imputer_mean = SimpleImputer(strategy="mean") #median, most_frequent, constant
df_simple["Age"] = imputer_mean.fit_transform(df[["Age"]])

# Para variables categóricas: rellenamos con la MODA (valor más frecuente)
imputer_mode = SimpleImputer(strategy="most_frequent")
df_simple["Embarked"] = imputer_mode.fit_transform(df[["Embarked"]])[:, 0]

print("Valores faltantes después de imputación simple:")
print(df_simple.isnull().sum())

# 7. Estrategia 3: Imputación con KNN
print("\n--- MÉTODO 3: IMPUTACIÓN KNN ---")

# Seleccionamos solo variables numéricas para KNN
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_knn = df[numeric_cols].copy()
#KNNImputer no puede manejar directamente variables categóricas como Sex o Embarked
knn_imputer = KNNImputer(n_neighbors=5)  
df_knn_imputed = pd.DataFrame( #Convertimos el array resultante otra vez en DataFrame de pandas.
    knn_imputer.fit_transform(df_knn), #fit: analiza el dataset (df_knn), calcula distancias entre registros (similaridad).
    columns=numeric_cols  #transform: usa esos cálculos para rellenar los valores faltantes en cada columna.
)

print("Ejemplo de imputación KNN en la variable 'Age':")
print(df_knn_imputed["Age"].head())

# 8. Estrategia 4: Imputación Iterativa (MICE)
print("\n--- MÉTODO 4: IMPUTACIÓN ITERATIVA (MICE) ---")

mice_imputer = IterativeImputer(random_state=42, max_iter=10)
df_mice_imputed = pd.DataFrame(
    mice_imputer.fit_transform(df_knn),
    columns=numeric_cols
)

print("Ejemplo de imputación MICE en la variable 'Age':")
print(df_mice_imputed["Age"].head())

# 9. Comparación de distribuciones
plt.figure(figsize=(15,5))

# Original sin NaN
sns.kdeplot(df["Age"].dropna(), label="Original", fill=True)

# Imputación Media
sns.kdeplot(df_simple["Age"], label="Imputación Media", fill=True)

# Imputación KNN
sns.kdeplot(df_knn_imputed["Age"], label="Imputación KNN", fill=True)

# Imputación MICE
sns.kdeplot(df_mice_imputed["Age"], label="Imputación MICE", fill=True)

plt.title("Comparación de Métodos de Imputación en la variable 'Age'")
plt.legend()
plt.show()
