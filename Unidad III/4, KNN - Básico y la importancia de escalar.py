# -*- coding: utf-8 -*-
"""
Tema: 2.2 K-Nearest Neighbors (KNN) - Implementación Básica
Dataset: Breast Cancer (Cáncer de Mama)
Explicación: Este script introduce el algoritmo KNN. A diferencia de los
árboles de decisión, KNN es muy sensible a la escala de las características.
Demostraremos por qué el escalado de datos es un paso PREVIO y
CRUCIAL antes de entrenar un modelo KNN y cómo mejora drásticamente
el rendimiento.
"""

# 1. Importar librerías
# ---------------------
# StandardScaler: para estandarizar las características (media 0, desviación estándar 1).
# Pipeline: para encadenar pasos de preprocesamiento y modelado.
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# 2. Cargar el dataset
# --------------------
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Convertir a DataFrame para exploración
df = pd.DataFrame(X, columns=cancer.feature_names)
print("Descripción de las características (sin escalar):")
# Observa las grandes diferencias en las medias (mean) y desviaciones (std).
# 'mean area' está en cientos, mientras que 'mean smoothness' es < 0.1.
# Esto es un problema para KNN.
print(df.describe())

# Separar datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- CASO 1: KNN SIN ESCALAR LOS DATOS (¡MALA PRÁCTICA!) ---
print("\n" + "="*50)
print("CASO 1: Entrenando KNN SIN escalar los datos")
print("="*50)

# Crear y entrenar el modelo KNN
# n_neighbors=5: el modelo considerará los 5 vecinos más cercanos para clasificar.
knn_no_scale = KNeighborsClassifier(n_neighbors=5)
knn_no_scale.fit(X_train, y_train)

# Evaluar el modelo
y_pred_no_scale = knn_no_scale.predict(X_test)
accuracy_no_scale = accuracy_score(y_test, y_pred_no_scale)

print(f"\nAccuracy del KNN sin escalado: {accuracy_no_scale:.4f}")
print("\nReporte de Clasificación (sin escalado):")
print(classification_report(y_test, y_pred_no_scale, target_names=cancer.target_names))
print("\nConclusión del Caso 1: El rendimiento es bueno, pero puede mejorar. Las variables con")
print("escalas más grandes (como 'mean area') dominarán el cálculo de distancia, ignorando")
print("la información de variables con escalas más pequeñas.")


# --- CASO 2: KNN CON ESCALADO DE DATOS (¡BUENA PRÁCTICA!) ---
print("\n" + "="*50)
print("CASO 2: Entrenando KNN CON escalado de datos")
print("="*50)

# 3. Escalar los datos
# ---------------------
# Creamos un objeto StandardScaler
scaler = StandardScaler()

# Usamos `fit_transform` en los datos de entrenamiento.
# 'fit' calcula la media y desviación estándar de CADA característica.
# 'transform' aplica la estandarización: (valor - media) / desviación_std.
X_train_scaled = scaler.fit_transform(X_train)

# IMPORTANTE: Usamos `transform` (NO `fit_transform`) en los datos de prueba.
# Esto asegura que los datos de prueba se escalen usando la MISMA media y
# desviación estándar calculadas a partir de los datos de entrenamiento.
# Esto evita la "fuga de datos" (data leakage) del conjunto de prueba al de entrenamiento.
X_test_scaled = scaler.transform(X_test)

# Mostramos cómo se ven los datos después de escalar
df_scaled = pd.DataFrame(X_train_scaled, columns=cancer.feature_names)
print("\nDescripción de las características (escaladas):")
# Ahora todas las características tienen media cercana a 0 y desviación cercana a 1.
print(df_scaled.describe())

# 4. Crear y entrenar el modelo KNN con datos escalados
# ----------------------------------------------------
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

# 5. Evaluar el modelo con datos escalados
# ---------------------------------------
y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print(f"\nAccuracy del KNN con escalado: {accuracy_scaled:.4f}")
print("\nReporte de Clasificación (con escalado):")
print(classification_report(y_test, y_pred_scaled, target_names=cancer.target_names))


# --- RESUMEN Y MEJOR PRÁCTICA USANDO PIPELINE ---
print("\n" + "="*50)
print("RESUMEN Y MEJOR PRÁCTICA: USO DE PIPELINES")
print("="*50)
print(f"Mejora en Accuracy: {accuracy_scaled - accuracy_no_scale:.4f}")
print("El escalado de datos es fundamental para algoritmos basados en distancia como KNN.")

# Un Pipeline automatiza el flujo de trabajo (escalar -> entrenar).
# Es menos propenso a errores y más fácil de manejar.
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Paso 1: escalar los datos
    ('knn', KNeighborsClassifier(n_neighbors=5)) # Paso 2: entrenar el modelo
])

# Entrenamos el pipeline con los datos originales.
# El pipeline se encarga internamente de escalar los datos de entrenamiento.
pipeline.fit(X_train, y_train)

# Hacemos predicciones con los datos de prueba originales.
# El pipeline se encarga de escalar los datos de prueba antes de predecir.
y_pred_pipeline = pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"\nAccuracy del KNN usando un Pipeline: {accuracy_pipeline:.4f}")
print("El resultado es idéntico al proceso manual, pero el código es más limpio y seguro.")
