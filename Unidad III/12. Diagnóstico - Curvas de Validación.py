# -*- coding: utf-8 -*-
"""
Tema: 6. Diagnóstico de Overfitting y Underfitting con Curvas de Validación
Dataset: Digits
Explicación: Las curvas de validación son una herramienta de diagnóstico
esencial. Nos muestran cómo cambia el rendimiento de un modelo (tanto en el
conjunto de entrenamiento como en el de validación) a medida que variamos
un solo hiperparámetro. Esto nos ayuda a identificar:
- **Underfitting (Subajuste):** Si tanto el score de entrenamiento como el de
  validación son bajos. El modelo es demasiado simple.
- **Overfitting (Sobreajuste):** Si el score de entrenamiento es muy alto,
  pero el de validación es significativamente más bajo. El modelo memorizó
  los datos de entrenamiento pero no generaliza bien.
- **Rango Óptimo:** La zona donde el score de validación es más alto y la
  diferencia (gap) con el score de entrenamiento es pequeña.
"""
# 1. Importar librerías
# ---------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier

# 2. Cargar datos
# ---------------
digits = load_digits()
X, y = digits.data, digits.target

# 3. Definir el modelo y el hiperparámetro a analizar
# ---------------------------------------------------
# Analizaremos cómo la `max_depth` (profundidad máxima) de un Árbol de
# Decisión afecta su rendimiento. Este es un parámetro clave para controlar
# la complejidad del modelo.
model = DecisionTreeClassifier(random_state=42)
param_name = "max_depth"
param_range = np.arange(1, 21) # Probaremos profundidades de 1 a 20

# 4. Calcular la curva de validación
# ----------------------------------
# `validation_curve` automatiza el proceso. Para cada valor en `param_range`:
# 1. Entrena y evalúa el modelo usando validación cruzada (cv=5).
# 2. Almacena los scores tanto del conjunto de entrenamiento como del de validación.
# Nos devuelve dos matrices: una con los scores de entrenamiento y otra con los de validación.
train_scores, test_scores = validation_curve(
    model, X, y,
    param_name=param_name,
    param_range=param_range,
    cv=5, # Validación cruzada de 5 pliegues
    scoring="accuracy",
    n_jobs=-1
)

# 5. Calcular la media y desviación estándar de los scores
# ---------------------------------------------------------
# Cada fila en train_scores y test_scores corresponde a un valor del `param_range`.
# Cada columna corresponde a un pliegue de la validación cruzada.
# Calculamos la media y la desviación estándar a lo largo de los pliegues (axis=1).
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 6. Graficar la curva de validación
# ----------------------------------
plt.figure(figsize=(10, 6))
plt.title(f"Curva de Validación para Decision Tree ({param_name})")
plt.xlabel(param_name)
plt.ylabel("Accuracy Score")
plt.ylim(0.0, 1.1)
plt.grid(True)

# Graficar la línea del score de entrenamiento promedio
plt.plot(param_range, train_scores_mean, label="Score de Entrenamiento", color="darkorange", marker='o')
# Graficar la banda de desviación estándar del entrenamiento
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="darkorange")

# Graficar la línea del score de validación promedio
plt.plot(param_range, test_scores_mean, label="Score de Validación Cruzada", color="navy", marker='o')
# Graficar la banda de desviación estándar de la validación
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="navy")

plt.legend(loc="best")
plt.show()

# 7. Interpretación del gráfico
# -----------------------------
print("--- ANÁLISIS DE LA CURVA DE VALIDACIÓN ---")
print("1. Zona de Underfitting (max_depth baja, ej. 1-4):")
print("   - Ambos scores (entrenamiento y validación) son bajos.")
print("   - El modelo es demasiado simple para capturar la complejidad de los datos.")
print("   - La brecha (gap) entre las dos curvas es pequeña.")

print("\n2. Zona de Overfitting (max_depth alta, ej. 12+):")
print("   - El score de entrenamiento se acerca a 1.0 (el modelo memoriza los datos).")
print("   - El score de validación se estanca o incluso empieza a bajar.")
print("   - La brecha entre las dos curvas es grande y creciente. Esto es una clara señal de sobreajuste.")

# Encontrar el mejor valor del hiperparámetro
best_depth_idx = np.argmax(test_scores_mean)
best_depth = param_range[best_depth_idx]
best_score = test_scores_mean[best_depth_idx]

print(f"\n3. Zona Óptima (alrededor de max_depth = {best_depth}):")
print(f"   - El score de validación alcanza su punto máximo ({best_score:.4f}).")
print("   - Se logra un buen equilibrio: el modelo es lo suficientemente complejo para aprender,")
print("     pero no tanto como para sobreajustarse.")
print("   - La brecha entre las curvas es razonablemente pequeña.")
