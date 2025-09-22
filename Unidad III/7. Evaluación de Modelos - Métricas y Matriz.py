# -*- coding: utf-8 -*-
"""
Tema: 3. Evaluación de Modelos
Dataset: Wine (Vino)
Explicación: Simplemente medir la exactitud (accuracy) no siempre es
suficiente. En problemas del mundo real, el costo de un Falso Positivo
puede ser muy diferente al de un Falso Negativo. Este script crea una
función reutilizable para evaluar un modelo de clasificación de forma
exhaustiva, mostrando la matriz de confusión y un reporte detallado
con métricas clave como Precisión, Recall y F1-Score.
"""

# 1. Importar librerías
# ---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# 2. Cargar y preparar los datos
# ------------------------------
wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Entrenar un modelo de ejemplo
# --------------------------------
# Usaremos un RandomForest, que es un modelo potente y de uso común.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Función de Evaluación Exhaustiva
# -----------------------------------
def evaluar_modelo_clasificacion(y_true, y_pred, labels, target_names):
    """
    Calcula y muestra un conjunto completo de métricas de evaluación
    para un problema de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas por el modelo.
        labels: Lista de las etiquetas de clase (ej. [0, 1, 2]).
        target_names: Nombres de las clases para los gráficos (ej. ['clase_A', 'clase_B']).
    """
    print("--- INFORME DE EVALUACIÓN DEL MODELO ---")
    
    # --- Métricas Generales ---
    # `average='weighted'` calcula la métrica para cada clase y encuentra su
    # promedio, ponderado por el número de instancias verdaderas para cada clase.
    # Esto tiene en cuenta el desequilibrio de clases.
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("\n--- Métricas Generales (Ponderadas) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # --- Reporte de Clasificación Detallado ---
    # Este reporte desglosa Precision, Recall y F1-Score para cada clase individualmente.
    # 'support' es el número de ocurrencias reales de la clase en los datos.
    print("\n--- Reporte de Clasificación por Clase ---")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # --- Matriz de Confusión ---
    # Muestra visualmente el rendimiento del clasificador.
    # Cada fila representa una clase real, mientras que cada columna representa una clase predicha.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.show()
    
    print("\n--- Interpretación de la Matriz de Confusión ---")
    print("La diagonal principal (de arriba-izquierda a abajo-derecha) muestra las predicciones correctas.")
    print("Los valores fuera de la diagonal son errores de clasificación.")
    print("Por ejemplo, el valor en la fila 'i' y columna 'j' es el número de veces que una instancia de la clase 'i' fue incorrectamente clasificada como clase 'j'.")
    print("------------------------------------------")

# 5. Usar la función para evaluar nuestro modelo
# ----------------------------------------------
evaluar_modelo_clasificacion(y_test, y_pred,
                             labels=np.unique(wine.target),
                             target_names=wine.target_names)
