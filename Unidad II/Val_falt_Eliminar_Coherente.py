import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# 1. Cargar dataset
# ============================
df = pd.read_csv(r"J:\Mi unidad\Mineria de Datos\Algoritmos\Datasets\synthetic_dataset.csv")

# ============================
# 2. Análisis de valores faltantes 
# ============================
missing_count = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

# ============================
# 3. Mostrar resultados 
# ============================
print("Conteo de valores faltantes:\n", missing_count)
print("\nPorcentaje de valores faltantes (%):\n", missing_percent)
print("\nPrimeras 10 filas del dataset:\n")
print(df.head(10))

# ============================
# 4. Visualización de datos faltantes 
# ============================
plt.figure(figsize=(12, 8))

# Subplot 1: Mapa de calor de valores faltantes
plt.subplot(2, 2, 1)
sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title('Mapa de Valores Faltantes')
plt.xlabel("Columnas")
plt.ylabel("Filas")

# Subplot 2: Conteo de valores faltantes
plt.subplot(2, 2, 2)
missing_count.plot(kind='bar')
plt.title('Conteo de Valores Faltantes por Variable')
plt.ylabel("Número de valores faltantes")
plt.xticks(rotation=45, ha="right")

# Subplot 3: Porcentaje de valores faltantes
plt.subplot(2, 2, 3)
missing_percent.plot(kind='bar', color='orange')
plt.title('Porcentaje de Valores Faltantes por Variable')
plt.ylabel("Porcentaje (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# ============================
# 5. Limpieza de datos
# ============================
# Primero columnas con menos del 20% de NaN
threshold = 0.2
df_clean = df.loc[:, df.isnull().mean() < threshold]

# Luego eliminar filas con NaN en esas columnas
df_clean = df_clean.dropna()

print(f"\nDataset limpio final: {len(df_clean)} filas × {len(df_clean.columns)} columnas")
print("\nPrimeras 20 filas del dataset limpio:\n")
print(df_clean.head(20))

# ============================
# 6. Visualización después de la limpieza 
# ============================
# Nota: si usaste dropna(), es esperable que no queden NaN.

# Recalcular valores faltantes en el dataset limpio (por columna)
missing_count_clean = df_clean.isnull().sum().sort_values(ascending=False)
missing_percent_clean = (df_clean.isnull().mean() * 100).sort_values(ascending=False)

# Recalcular valores faltantes por fila (distribución)
missing_count_rows_clean = df_clean.isnull().sum(axis=1)
num_rows_with_nan_clean = (missing_count_rows_clean > 0).sum()

print(f"\nFilas con al menos un NaN después de la limpieza: {num_rows_with_nan_clean}")
print(f"Porcentaje de filas afectadas: {num_rows_with_nan_clean / len(df_clean) * 100:.2f}%")

plt.figure(figsize=(12, 8))

# Subplot 1: Mapa de calor después de la limpieza
plt.subplot(2, 2, 1)
sns.heatmap(df_clean.isnull(), cbar=True, cmap='viridis')
plt.title('Mapa de Valores Faltantes (Después de la Limpieza)')
plt.xlabel("Columnas")
plt.ylabel("Filas")

# Subplot 2: Conteo de valores faltantes después de la limpieza (columnas)
plt.subplot(2, 2, 2)
missing_count_clean.plot(kind='bar')
plt.title('Conteo de Valores Faltantes por Columna (Limpio)')
plt.ylabel("Número de valores faltantes")
plt.xticks(rotation=45, ha="right")

# Subplot 3: Distribución de NaN por fila (debería concentrarse en 0)
plt.subplot(2, 2, 3)
missing_count_rows_clean.value_counts().sort_index().plot(kind='bar', color='teal')
plt.title('Distribución de Valores Faltantes por Fila (Limpio)')
plt.xlabel("Cantidad de NaN en la fila")
plt.ylabel("Número de filas")

plt.tight_layout()
plt.show()

# Imprimir el dataframe limpio completo (cuidado si es muy grande)
# print(df_clean.to_string())

# Para revisión rápida:
print("\nResumen del dataset limpio:\n")
print(df_clean.info())
