# -*- coding: utf-8 -*-
"""
Tema: 4. Validación Cruzada
Dataset: Digits (dígitos escritos a mano)
Explicación: Una simple división train/test depende mucho de la "suerte"
de cómo se dividieron los datos. La Validación Cruzada (Cross-Validation)
es una técnica mucho más robusta para estimar el rendimiento de un modelo.
Divide el dataset en 'K' partes (pliegues), entrena el modelo K veces
(usando K-1 pliegues para entrenar y 1 para probar), y promedia los resultados.
Esto da una estimación más fiable de cómo se comportará el modelo en datos nuevos.
"""

# 1. Importar librerías
# ---------------------
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# 2. Cargar datos
# ---------------
digits = load_digits()
X, y = digits.data, digits.target

# 3. Crear el modelo a evaluar
# ----------------------------
# Usaremos un RandomForest como ejemplo.
model = RandomForestClassifier(n_estimators=50, random_state=42)

# --- ESTRATEGIA 1: K-Fold Estándar ---
print("--- 1. Validación Cruzada con K-Fold Estándar ---")
# KFold divide los datos en K pliegues de manera secuencial o aleatoria.
# n_splits=5: dividiremos los datos en 5 pliegues.
# shuffle=True: baraja los datos antes de dividirlos, lo cual es casi siempre una buena idea.
# random_state=42: para reproducibilidad.
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# `cross_val_score` automatiza el proceso de validación cruzada.
# Le pasamos el modelo, los datos, las etiquetas y la estrategia de CV (kfold).
# Devuelve un array con el score (por defecto, accuracy) para cada uno de los 5 pliegues.
scores_kfold = cross_val_score(model, X, y, cv=kfold)

print(f"Scores para cada uno de los 5 pliegues: {scores_kfold}")
print(f"Accuracy Promedio: {scores_kfold.mean():.4f}")
print(f"Desviación Estándar de Accuracy: {scores_kfold.std():.4f}")
print("Interpretación: El modelo tiene un rendimiento promedio de {:.2f}%, y la desviación estándar nos da una idea de la variabilidad del rendimiento. Un rango de rendimiento probable es {:.2f}% ± {:.2f}%.".format(scores_kfold.mean()*100, scores_kfold.mean()*100, 2*scores_kfold.std()*100))


# --- ESTRATEGIA 2: Stratified K-Fold (Recomendado para Clasificación) ---
print("\n--- 2. Validación Cruzada con Stratified K-Fold ---")
# StratifiedKFold es una variación de KFold que preserva el porcentaje de
# muestras para cada clase en cada pliegue. Es especialmente importante
# cuando hay desequilibrio de clases, pero es una buena práctica en general
# para problemas de clasificación.
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_stratified = cross_val_score(model, X, y, cv=stratified_kfold)

print(f"Scores para cada uno de los 5 pliegues (Estratificado): {scores_stratified}")
print(f"Accuracy Promedio (Estratificado): {scores_stratified.mean():.4f}")
print(f"Desviación Estándar de Accuracy (Estratificado): {scores_stratified.std():.4f}")
print("Interpretación: Los resultados suelen ser más fiables con la estratificación, ya que cada pliegue de prueba es una mejor representación del dataset completo.")


# --- VALIDACIÓN CRUZADA CON DIFERENTES MÉTRICAS ---
print("\n--- 3. Validación Cruzada con Múltiples Métricas ---")
# Podemos cambiar la métrica de evaluación usando el parámetro `scoring`.
# Esto es útil si la exactitud no es la mejor medida para nuestro problema.

# Evaluamos usando F1-Score ponderado
scores_f1 = cross_val_score(model, X, y, cv=stratified_kfold, scoring='f1_weighted')
print(f"F1-Score Promedio (Ponderado): {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")

# Evaluamos usando Precision ponderada
scores_precision = cross_val_score(model, X, y, cv=stratified_kfold, scoring='precision_weighted')
print(f"Precision Promedio (Ponderada): {scores_precision.mean():.4f} ± {scores_precision.std():.4f}")

# Evaluamos usando Recall ponderado
scores_recall = cross_val_score(model, X, y, cv=stratified_kfold, scoring='recall_weighted')
print(f"Recall Promedio (Ponderado): {scores_recall.mean():.4f} ± {scores_recall.std():.4f}")

print("\nConclusión: La validación cruzada es la herramienta estándar para")
print("comparar modelos o configuraciones de hiperparámetros de manera robusta,")
print("ya que mitiga el efecto de la aleatoriedad en la división de los datos.")
