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
# 3. Mostrar resultados en consola
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

# Eliminación de filas con al menos un valor faltante
df_dropna_rows = df.dropna()
print(f"\nDataset original: {len(df)} filas")
print(f"Después de eliminar filas con NaN: {len(df_dropna_rows)} filas")

# Eliminación de columnas con más del 20% de valores faltantes
threshold = 0.2  # 20% máximo permitido de NaN
df_drop_cols = df.loc[:, df.isnull().mean() < threshold]
print(f"\nColumnas originales: {len(df.columns)}")
print(f"Columnas después de eliminar las que superan {threshold*100}% de NaN: {len(df_drop_cols.columns)}")
print(f"Columnas conservadas: {list(df_drop_cols.columns)}")


# ============================
# 6. Visualización después de la limpieza
# ============================

# Recalcular valores faltantes en el dataset limpio
missing_count_clean = df_drop_cols.isnull().sum().sort_values(ascending=False)
missing_percent_clean = (df_drop_cols.isnull().mean() * 100).sort_values(ascending=False)

plt.figure(figsize=(12, 8))

# Subplot 1: Mapa de calor después de la limpieza
plt.subplot(2, 2, 1)
sns.heatmap(df_drop_cols.isnull(), cbar=True, cmap='viridis')
plt.title('Mapa de Valores Faltantes (Después de la Limpieza)')
plt.xlabel("Columnas")
plt.ylabel("Filas")

# Subplot 2: Conteo de valores faltantes después de la limpieza
plt.subplot(2, 2, 2)
missing_count_clean.plot(kind='bar')
plt.title('Conteo de Valores Faltantes (Limpio)')
plt.ylabel("Número de valores faltantes")
plt.xticks(rotation=45, ha="right")

# Subplot 3: Porcentaje de valores faltantes después de la limpieza
plt.subplot(2, 2, 3)
missing_percent_clean.plot(kind='bar', color='orange')
plt.title('Porcentaje de Valores Faltantes (Limpio)')
plt.ylabel("Porcentaje (%)")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()