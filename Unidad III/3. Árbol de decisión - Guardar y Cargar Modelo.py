# -*- coding: utf-8 -*-
"""
Tema: 2.1 Árboles de Decisión - Guardar y Cargar (Persistencia de Modelos)
Dataset: Iris
Explicación: Entrenar un modelo puede llevar mucho tiempo. En lugar de
re-entrenarlo cada vez que lo necesitamos, podemos guardarlo en un archivo.
Este script muestra cómo entrenar un modelo, guardarlo, y luego cargarlo
para hacer nuevas predicciones, simulando un entorno de producción.
"""

# 1. Importar las librerías necesarias
# ------------------------------------
# joblib es una librería eficiente para guardar y cargar objetos de Python,
# especialmente útil para modelos de scikit-learn con grandes arrays de NumPy.
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import os

# --- PASO 1: ENTRENAMIENTO Y GUARDADO DEL MODELO (simula el entorno de desarrollo) ---

print("--- FASE 1: ENTRENAMIENTO Y GUARDADO DEL MODELO ---")

# 2. Cargar y entrenar el modelo (como en el script básico)
# ---------------------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target

# Usamos todos los datos para entrenar el modelo final.
# En un proyecto real, usaríamos los mejores hiperparámetros encontrados
# en la fase de validación.
model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=2, min_samples_split=2, random_state=42)
print("\nEntrenando el modelo con todos los datos de Iris...")
model.fit(X, y)
print("Modelo entrenado.")

# 3. Guardar el modelo entrenado en un archivo
# ---------------------------------------------
# Definimos el nombre del archivo. La extensión .joblib es una convención.
model_filename = "modelo_arbol_decision_iris.joblib"

# La función `dump` de joblib serializa el objeto del modelo y lo guarda.
joblib.dump(model, model_filename)

print(f"\nModelo guardado exitosamente en el archivo: '{model_filename}'")
# Verificamos que el archivo existe
if os.path.exists(model_filename):
    print("El archivo del modelo ha sido creado en el directorio actual.")
else:
    print("Error: El archivo del modelo no se pudo crear.")

# --- Simulación de un tiempo después, en otro script o en producción ---
print("\n" + "="*50 + "\n")
print("--- FASE 2: CARGA Y USO DEL MODELO (simula un entorno de producción) ---")

# 4. Cargar el modelo desde el archivo
# ------------------------------------
# Suponemos que estamos en un nuevo programa y necesitamos usar el modelo.
# Primero, verificamos si el archivo existe para evitar errores.
if os.path.exists(model_filename):
    # La función `load` de joblib lee el archivo y reconstruye el objeto del modelo.
    loaded_model = joblib.load(model_filename)
    print(f"Modelo cargado exitosamente desde '{model_filename}'")

    # 5. Usar el modelo cargado para hacer predicciones
    # -------------------------------------------------
    # Crearemos algunos datos nuevos (muestras que el modelo nunca ha visto).
    # Estas medidas podrían venir de un sensor, un formulario, etc.
    # Muestra 1: Parece una Setosa
    # Muestra 2: Parece una Versicolor
    # Muestra 3: Parece una Virginica
    nuevos_datos = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [6.0, 2.9, 4.5, 1.5],
        [6.9, 3.1, 5.4, 2.1]
    ])

    print("\nRealizando predicciones sobre nuevos datos:")
    print("Datos de entrada:")
    print(nuevos_datos)

    # Usamos el método `predict` del modelo cargado
    predictions = loaded_model.predict(nuevos_datos)
    
    # También podemos obtener las probabilidades de cada clase
    probabilities = loaded_model.predict_proba(nuevos_datos)

    # 6. Interpretar y mostrar los resultados
    # ---------------------------------------
    # Mapeamos los índices de las clases (0, 1, 2) a sus nombres reales.
    species_names = iris.target_names

    print("\nResultados de la predicción:")
    for i, prediction in enumerate(predictions):
        predicted_species = species_names[prediction]
        print(f"  - Muestra {i+1}: Predicción = {predicted_species}")
        # Mostramos las probabilidades para dar más contexto
        prob_dist = probabilities[i]
        print(f"    Probabilidades: {species_names[0]}={prob_dist[0]:.2f}, {species_names[1]}={prob_dist[1]:.2f}, {species_names[2]}={prob_dist[2]:.2f}")

else:
    print(f"Error: No se encontró el archivo del modelo '{model_filename}'.")
    print("Por favor, ejecuta la primera fase del script para crear el modelo.")
