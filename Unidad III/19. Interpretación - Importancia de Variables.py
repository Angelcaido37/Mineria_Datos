# -*- coding: utf-8 -*-
"""
Tema: 10.1 Interpretación de Modelos - Importancia de Variables
Dataset: Wine (Vino)
Explicación: Una vez que tenemos un modelo que predice bien, a menudo
queremos saber POR QUÉ predice de esa manera. Entender qué características
(variables) son las más importantes para las decisiones del modelo es un
paso crucial para la interpretabilidad. Esto nos ayuda a confiar en el modelo,
explicarlo a otros y, a veces, a obtener nuevos conocimientos sobre el problema.

Mostraremos dos métodos comunes con modelos basados en árboles (como RandomForest):
1.  **Importancia de Característica (Impurity-based):** Rápido y fácil. Se
    basa en qué tan frecuentemente una característica se usa para dividir un
    nodo en los árboles y cuánto reduce la impureza (Gini/entropía) en promedio.
    Puede estar sesgado hacia características con alta cardinalidad (muchos valores únicos).
2.  **Importancia por Permutación:** Más robusto y fiable. Después de entrenar
    un modelo, se toma una característica y se barajan sus valores de forma
    aleatoria. Luego se mide cuánto cae el rendimiento del modelo. Una caída
    grande significa que la característica era muy importante.
"""

# 1. Importar librerías
# ---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# 2. Cargar y preparar datos
# --------------------------
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Entrenar el modelo
# ---------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- MÉTODO 1: IMPORTANCIA BASADA EN IMPUREZA (GINI) ---
print("--- 1. Importancia de Característica (Basada en Impureza/Gini) ---")
# El modelo entrenado tiene un atributo `feature_importances_`
importances = model.feature_importances_
feature_names = X.columns

# Crear un DataFrame para una fácil visualización
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print("Top 5 características más importantes:")
print(feature_importance_df.head())

# Graficar la importancia
plt.figure(figsize=(18, 7))
plt.subplot(1, 2, 1)
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel("Importancia (Reducción de Impureza Gini)")
plt.ylabel("Característica")
plt.title("Importancia de Características (Método Gini)")
plt.gca().invert_yaxis() # Mostrar la más importante arriba


# --- MÉTODO 2: IMPORTANCIA POR PERMUTACIÓN ---
print("\n--- 2. Importancia por Permutación ---")
# Este método se calcula sobre un conjunto de datos que el modelo no ha visto
# (generalmente el de prueba o validación) para evitar sesgos.
# n_repeats: número de veces que se baraja cada característica para obtener una
# estimación más estable.
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Crear un DataFrame con los resultados
perm_importance_df = pd.DataFrame({'feature': feature_names, 'importance_mean': result.importances_mean})
perm_importance_df = perm_importance_df.sort_values('importance_mean', ascending=False)

print("Top 5 características más importantes (por permutación):")
print(perm_importance_df.head())

# Graficar la importancia por permutación
plt.subplot(1, 2, 2)
plt.barh(perm_importance_df['feature'], perm_importance_df['importance_mean'])
plt.xlabel("Caída de Rendimiento (Accuracy)")
plt.ylabel("Característica")
plt.title("Importancia de Características (por Permutación)")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

print("\n--- Comparación y Conclusión ---")
print("Ambos métodos suelen identificar características similares como las más importantes,")
print("pero la importancia por permutación se considera generalmente más fiable porque")
print("está directamente ligada a la métrica de rendimiento del modelo y funciona con")
print("cualquier tipo de modelo, no solo los basados en árboles.")
