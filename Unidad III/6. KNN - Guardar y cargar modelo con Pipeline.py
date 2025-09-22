# -*- coding: utf-8 -*-
"""
Tema: 2.2 K-Nearest Neighbors (KNN) - Persistencia con Pipeline
Dataset: Breast Cancer
Explicación: Para algoritmos como KNN que requieren un preprocesamiento
(como el escalado), es crucial guardar no solo el modelo, sino todo el flujo
de trabajo. Un `Pipeline` de scikit-learn es perfecto para esto, ya que
encapsula tanto el escalador como el clasificador en un solo objeto.
Este script muestra cómo entrenar y guardar un pipeline completo.
"""

# 1. Importar librerías
# ---------------------
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os

# --- PASO 1: ENTRENAMIENTO Y GUARDADO DEL PIPELINE ---

print("--- FASE 1: ENTRENAMIENTO Y GUARDADO DEL PIPELINE ---")

# 2. Cargar datos
# ---------------
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
# En esta fase no necesitamos separar en train/test porque estamos
# creando el modelo final para producción. Usamos todos los datos.

# 3. Crear y entrenar el Pipeline
# -------------------------------
# El pipeline define los pasos secuenciales de nuestro flujo de trabajo.
# 'scaler': El objeto StandardScaler.
# 'knn': El objeto KNeighborsClassifier. Usamos k=7 que encontramos como
# un buen valor en el script anterior.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=7))
])

print("\nEntrenando el Pipeline (Escalador + KNN)...")
# Al hacer `fit` en el pipeline, los datos (X) pasan primero por el
# `fit_transform` del scaler, y el resultado se pasa al `fit` del knn.
pipeline.fit(X, y)
print("Pipeline entrenado.")

# 4. Guardar el Pipeline completo
# -------------------------------
pipeline_filename = "pipeline_knn_cancer.joblib"

# Guardamos el objeto 'pipeline' completo. Este archivo ahora contiene
# tanto el escalador (con las medias y desviaciones estándar ya calculadas)
# como el modelo KNN entrenado.
joblib.dump(pipeline, pipeline_filename)

print(f"\nPipeline guardado exitosamente en: '{pipeline_filename}'")
if os.path.exists(pipeline_filename):
    print("El archivo del pipeline ha sido creado.")

# --- PASO 2: CARGA Y USO DEL PIPELINE ---
print("\n" + "="*50 + "\n")
print("--- FASE 2: CARGA Y USO DEL PIPELINE ---")

# 5. Cargar el Pipeline
# ---------------------
if os.path.exists(pipeline_filename):
    loaded_pipeline = joblib.load(pipeline_filename)
    print(f"Pipeline cargado desde '{pipeline_filename}'")
    
    # 6. Usar el Pipeline cargado para nuevas predicciones
    # ----------------------------------------------------
    # Simulamos la llegada de nuevos datos.
    # IMPORTANTE: Estos datos están en su escala original, no están escalados.
    
    # Muestra 1: Características de un tumor potencialmente maligno
    # Muestra 2: Características de un tumor potencialmente benigno
    nuevos_datos = np.array([
        [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
        [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
    ])

    print("\nRealizando predicciones sobre nuevos datos (en escala original):")
    
    # Al llamar a `predict` en el pipeline cargado:
    # 1. Los `nuevos_datos` pasan por el `transform` del escalador guardado.
    # 2. Los datos ya escalados se pasan al `predict` del modelo KNN guardado.
    # Todo esto ocurre de forma transparente para nosotros.
    predictions = loaded_pipeline.predict(nuevos_datos)
    probabilities = loaded_pipeline.predict_proba(nuevos_datos)

    # 7. Interpretar los resultados
    # ------------------------------
    target_names = cancer.target_names
    
    print("\nResultados de la predicción:")
    for i, pred in enumerate(predictions):
        predicted_class = target_names[pred]
        prob_dist = probabilities[i]
        print(f"  - Muestra {i+1}: Predicción = '{predicted_class}' (Clase {pred})")
        print(f"    Probabilidades: {target_names[0]}={prob_dist[0]:.2f}, {target_names[1]}={prob_dist[1]:.2f}")
else:
    print(f"Error: No se encontró el archivo '{pipeline_filename}'.")
