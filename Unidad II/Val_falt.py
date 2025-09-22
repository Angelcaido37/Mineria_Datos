import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Crear un dataset de ejemplo con valores faltantes
np.random.seed(42)

data = {
    'edad': np.random.normal(35, 10, 1000),
    'ingresos': np.random.normal(50000, 15000, 1000),
    'experiencia': np.random.normal(8, 5, 1000),
    'educacion': np.random.choice(['Bachillerato', 'Universitario', 'Posgrado'], 1000)
}

df = pd.DataFrame(data)

# Introducir valores faltantes de forma artificial
missing_indices_edad = np.random.choice(df.index, size=100, replace=False)
df.loc[missing_indices_edad, 'edad'] = np.nan

# Redondear y convertir a entero
df['edad'] = df['edad'].round().astype('Int64')


# Análisis de valores faltantes
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

# Mostrar resultados
print("Conteo de valores faltantes:\n", missing_count)
print("\nPorcentaje de valores faltantes:\n", missing_percent)
print(df)

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

plt.tight_layout()
plt.show()



