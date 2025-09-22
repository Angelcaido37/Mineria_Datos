import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset desde un archivo CSV
df = pd.read_csv("J:\Mi unidad\Mineria de Datos\Algoritmos\Datasets\synthetic_dataset.csv") 

# Análisis de valores faltantes
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

# Mostrar resultados
print("Conteo de valores faltantes:\n", missing_count)
print("\nPorcentaje de valores faltantes:\n", missing_percent)

# Mostrar primeras filas para inspección
print("\nPrimeras 10 filas del dataset:\n")
print(df.head(10))

#Visualización de datos faltantes
plt.figure(figsize=(12, 8))

# Subplot 1: Mapa de calor de valores faltantes
plt.subplot(2, 2, 1)
sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title('Mapa de Valores Faltantes')

# Subplot 2: Conteo de valores faltantes
plt.subplot(2, 2, 2)
missing_count.plot(kind='bar')
plt.title('Conteo de Valores Faltantes por Variable')

# Subplot 3: Porcentaje de valores faltantes
plt.subplot(2, 2, 3)
missing_percent.plot(kind='bar', color='orange')
plt.title('Porcentaje de Valores Faltantes por Variable')

plt.tight_layout() #ajusta automáticamente los márgenes y espaciado para que los títulos y ejes no se encimen.
plt.show()


