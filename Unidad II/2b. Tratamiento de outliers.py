# ==============================================================
# TEMA 2: DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# Dataset: USA Housing (Kaggle)
# Detección y eliminación automática de outliers con IQR
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar dataset
df = pd.read_csv(r"Datasets/housing_train.csv")

# 2. Seleccionar solo columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 3. Función para eliminar outliers con IQR
def eliminar_outliers_iqr(data, cols):
    df_clean = data.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Filtrar filas dentro del rango permitido
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        
        print(f"{col}: outliers eliminados, nuevo tamaño = {len(df_clean)} filas")
    return df_clean

# 4. Dataset sin outliers
df_no_outliers = eliminar_outliers_iqr(df, numeric_cols)

print("\n=== Comparación de tamaños ===")
print(f"Original: {len(df)} filas")
print(f"Sin outliers: {len(df_no_outliers)} filas")

# 5. Visualización antes y después para la variable Price
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.boxplot(x=df["SalePrice"], color="orange")
plt.title("Antes (con outliers)")

plt.subplot(1,2,2)
sns.boxplot(x=df_no_outliers["SalePrice"], color="green")
plt.title("Después (sin outliers)")

plt.show()

#guardar un archivo csv
df_no_outliers.to_csv("Datasets/housing_train_clean.csv", index=False)
print("Archivo guardado correctamente.")