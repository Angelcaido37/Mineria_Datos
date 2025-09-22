# -*- coding: utf-8 -*-
"""
Tema: 5.1 Ajuste de Hiperparámetros con GridSearchCV
Dataset: Breast Cancer
Explicación: GridSearchCV (Búsqueda en Rejilla con Validación Cruzada) es el
método más fundamental para la optimización de hiperparámetros. De forma
exhaustiva, construye y evalúa un modelo para cada combinación de los
hiperparámetros especificados en una "rejilla" o "parrilla". Utiliza
validación cruzada para evaluar cada combinación, lo que lo hace muy robusto.
Aunque puede ser computacionalmente costoso, garantiza que se exploren
todas las opciones definidas.
"""
# 1. Importar librerías
# ---------------------
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC # Support Vector Classifier, otro potente algoritmo de clasificación
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time

# 2. Cargar y preparar datos
# --------------------------
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Crear un Pipeline
# --------------------
# SVM es sensible a la escala de los datos, por lo que es una buena práctica
# incluir el escalado en un Pipeline junto con el modelo.
# Esto también nos permite ajustar hiperparámetros del modelo DENTRO del pipeline.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42)) # probability=True es útil para obtener probabilidades
])

# 4. Definir la parrilla de hiperparámetros
# -----------------------------------------
# Para acceder a los parámetros de un paso del pipeline, usamos la sintaxis:
# 'nombre_del_paso__nombre_del_parametro'
param_grid = {
    'svm__C': [0.1, 1, 10, 100],        # Parámetro de regularización. Controla el balance entre un margen de separación amplio y clasificar correctamente los puntos de entrenamiento.
    'svm__kernel': ['linear', 'rbf'], # El 'kernel' transforma los datos. 'linear' es para problemas linealmente separables, 'rbf' para problemas más complejos.
    'svm__gamma': ['scale', 'auto', 0.01, 0.001] # Parámetro del kernel RBF. Define cuánta influencia tiene un solo ejemplo de entrenamiento.
}
# Total de combinaciones a probar: 4 (C) * 2 (kernel) * 4 (gamma) = 32 combinaciones.
# Cada una se evaluará con 5-fold CV, así que serán 32 * 5 = 160 entrenamientos de modelo.

# 5. Configurar y Ejecutar GridSearchCV
# -------------------------------------
# cv=5: Validación cruzada de 5 pliegues.
# scoring='f1_weighted': Usaremos el F1-score ponderado como métrica de optimización,
# que es una buena opción general, especialmente si las clases están desbalanceadas.
# verbose=2: Muestra más información sobre el proceso.
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=2)

print("Iniciando GridSearchCV para el modelo SVM...")
start_time = time.time()
grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nGridSearchCV completado en {end_time - start_time:.2f} segundos.")

# 6. Analizar los resultados
# --------------------------
print("\n--- Resultados de la Búsqueda ---")
print(f"Mejor F1-score (ponderado) en validación cruzada: {grid_search.best_score_:.4f}")
print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# El `best_estimator_` es el pipeline completo (escalador + SVM) ya re-entrenado
# con los mejores parámetros usando todo el conjunto de entrenamiento.
best_model = grid_search.best_estimator_

# 7. Evaluar el modelo optimizado en el conjunto de prueba
# -------------------------------------------------------
print("\n--- Evaluación Final en el Conjunto de Prueba ---")
from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

# Opcional: Ver los resultados de todas las combinaciones
# -------------------------------------------------------
results_df = pd.DataFrame(grid_search.cv_results_)
# Seleccionamos y ordenamos las columnas más importantes para ver los resultados
cols_to_show = ['param_svm__C', 'param_svm__kernel', 'param_svm__gamma', 'mean_test_score', 'rank_test_score']
print("\nTop 10 combinaciones de hiperparámetros:")
print(results_df[cols_to_show].sort_values(by='rank_test_score').head(10))
