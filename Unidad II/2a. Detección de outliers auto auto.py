# ==============================================================
# TEMA 2: DETECCIÓN Y TRATAMIENTO DE OUTLIERS
# Dataset: USA Housing (Kaggle)
# Detección automática en todas las variables numéricas
# ==============================================================

# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Cargar dataset
df = pd.read_csv(r"Datasets/housing_train.csv")

# 3. Seleccionar solo columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns

# 4. Función para detectar outliers con IQR
def detectar_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower) | (data[col] > upper)]
    return len(outliers), lower, upper

# 5. Resumen de outliers por variable
print("=== Resumen de Outliers por Variable (IQR) ===")
outlier_summary = {}
for col in numeric_cols:
    count, lower, upper = detectar_outliers_iqr(df, col)
    outlier_summary[col] = count
    print(f"{col}: {count} outliers (límites: {lower:.2f} – {upper:.2f})")

# 6. Visualización automática con boxplots
num_vars = len(numeric_cols)
rows = (num_vars // 2) + (num_vars % 2)  # número de filas de subplots

plt.figure(figsize=(14, rows*4))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(rows, 2, i)
    sns.boxplot(x=df[col], color="skyblue")
    plt.title(f"Boxplot: {col}")
plt.tight_layout()
plt.show()
