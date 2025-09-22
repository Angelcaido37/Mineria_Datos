# -*- coding: utf-8 -*-
"""
Tema: 2.1 Árboles de Decisión - Ajuste de Hiperparámetros con GridSearchCV
Dataset: Wine (Vino)
Explicación: Un árbol de decisión sin límites puede crecer demasiado y
memorizar los datos de entrenamiento (overfitting), perdiendo su capacidad
de generalizar a nuevos datos. Este script utiliza 'GridSearchCV' para
encontrar la mejor combinación de hiperparámetros que resulta en un modelo
más robusto y generalizable.
"""

# 1. Importar las librerías necesarias
# ------------------------------------
# sklearn.model_selection.GridSearchCV: realiza la búsqueda exhaustiva de hiperparámetros.
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# 2. Cargar el conjunto de datos
# ------------------------------------
# El dataset 'Wine' contiene el análisis químico de vinos de 3 cultivares diferentes.
wine = load_wine()
X = wine.data
y = wine.target

# Convertir a DataFrame para una mejor exploración
df = pd.DataFrame(data=X, columns=wine.feature_names)
df['cultivar'] = y
print("Primeras 5 filas del dataset Wine:")
print(df.head())

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# 'stratify=y' es útil en clasificación para mantener la misma proporción de clases en
# los conjuntos de entrenamiento y prueba que en el dataset original.

# 3. Definir la "parrilla" de hiperparámetros a explorar
# ------------------------------------------------------
# Creamos un diccionario donde las claves son los nombres de los hiperparámetros
# del DecisionTreeClassifier y los valores son listas de los valores que queremos probar.
param_grid = {
    'criterion': ['gini', 'entropy'], # Métrica para medir la calidad de una división.
    'max_depth': [None, 5, 10, 15, 20], # Profundidad máxima del árbol. 'None' significa sin límite.
    'min_samples_split': [2, 5, 10, 20], # Número mínimo de muestras requeridas para dividir un nodo.
    'min_samples_leaf': [1, 2, 5, 10] # Número mínimo de muestras requeridas en un nodo hoja.
}

# 4. Configurar y ejecutar GridSearchCV
# -------------------------------------
# GridSearchCV probará TODAS las combinaciones posibles de la parrilla de parámetros.
# En este caso: 2 * 5 * 4 * 4 = 160 combinaciones.

# Instanciamos el modelo base
dt = DecisionTreeClassifier(random_state=42)

# cv=5: Usará validación cruzada de 5 pliegues (K-Fold). Cada combinación se entrena y
# evalúa 5 veces, lo que hace la evaluación de rendimiento más robusta.
# scoring='accuracy': La métrica para decidir cuál combinación es la mejor.
# n_jobs=-1: Utiliza todos los procesadores disponibles para acelerar la búsqueda.
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

print("\nIniciando la búsqueda de los mejores hiperparámetros con GridSearchCV...")
start_time = time.time()

# Ejecutamos la búsqueda sobre los datos de entrenamiento
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"Búsqueda completada en {end_time - start_time:.2f} segundos.")

# 5. Mostrar los resultados de la búsqueda
# -----------------------------------------
print("\n--- Resultados de GridSearchCV ---")
# La mejor combinación de parámetros encontrada
print(f"Mejores parámetros encontrados: {grid_search.best_params_}")

# El score de validación cruzada promedio con los mejores parámetros
print(f"Mejor score de validación cruzada (Accuracy): {grid_search.best_score_:.4f}")

# El mejor estimador (modelo) ya está re-entrenado con todos los datos de
# entrenamiento usando los mejores parámetros encontrados.
best_clf = grid_search.best_estimator_

# 6. Evaluar el mejor modelo en el conjunto de prueba
# ----------------------------------------------------
# Ahora evaluamos este modelo optimizado en el conjunto de prueba, que
# el proceso de GridSearchCV nunca vio.
y_pred = best_clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)

print(f"\nAccuracy del modelo optimizado en el conjunto de prueba: {accuracy_test:.4f}")
print("\nReporte de Clasificación en el conjunto de prueba:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Podemos comparar con un modelo por defecto (sin optimizar)
print("\n--- Comparación con Modelo por Defecto ---")
default_clf = DecisionTreeClassifier(random_state=42)
default_clf.fit(X_train, y_train)
y_pred_default = default_clf.predict(X_test)
accuracy_default = accuracy_score(y_test, y_pred_default)
print(f"Accuracy del modelo por defecto en prueba: {accuracy_default:.4f}")

# La optimización de hiperparámetros ayuda a prevenir el sobreajuste y a crear
# un modelo que generaliza mejor a datos nuevos, aunque en algunos casos la
# diferencia de rendimiento no sea drástica, es una práctica fundamental.
