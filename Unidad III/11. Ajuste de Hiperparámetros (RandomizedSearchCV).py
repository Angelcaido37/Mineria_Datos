# -*- coding: utf-8 -*-
"""
Tema: 5.2 Ajuste de Hiperparámetros con RandomizedSearchCV
Dataset: Breast Cancer
Explicación: Cuando el espacio de búsqueda de hiperparámetros es muy grande,
GridSearchCV se vuelve computacionalmente inviable. RandomizedSearchCV es una
alternativa más eficiente. En lugar de probar todas las combinaciones
posibles, muestrea un número fijo de combinaciones (`n_iter`) de las
distribuciones de parámetros especificadas. A menudo, encuentra una muy
buena combinación de parámetros en mucho menos tiempo que GridSearchCV.
"""
# 1. Importar librerías
# ---------------------
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
import time

# 2. Cargar y preparar datos
# --------------------------
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Definir las distribuciones de hiperparámetros
# ------------------------------------------------
# En lugar de una lista de valores fijos, podemos especificar distribuciones
# de probabilidad de las cuales el algoritmo tomará muestras.
param_dist = {
    # Número de árboles en el bosque. `randint(100, 1000)` generará enteros aleatorios entre 100 y 999.
    'n_estimators': randint(100, 1000),
    
    # Profundidad máxima de cada árbol.
    'max_depth': [None] + list(randint(5, 50).rvs(10)), # 'None' o 10 valores aleatorios entre 5 y 49.
    
    # Número mínimo de muestras para dividir un nodo.
    'min_samples_split': randint(2, 20),
    
    # Número mínimo de muestras en una hoja.
    'min_samples_leaf': randint(1, 20),
    
    # Número de características a considerar en cada división.
    'max_features': ['sqrt', 'log2', None]
}

# 4. Configurar y Ejecutar RandomizedSearchCV
# -------------------------------------------
# Modelo base
rf = RandomForestClassifier(random_state=42)

# n_iter=100: Probará 100 combinaciones de parámetros aleatorias.
# Este es el principal control de balance entre tiempo de ejecución y calidad de la solución.
# cv=5: Validación cruzada de 5 pliegues.
# scoring='accuracy': Métrica a optimizar.
# random_state=42: Para que la selección aleatoria de parámetros sea reproducible.
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print(f"Iniciando RandomizedSearchCV con {random_search.n_iter} iteraciones...")
start_time = time.time()
random_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nRandomizedSearchCV completado en {end_time - start_time:.2f} segundos.")

# 5. Analizar los resultados
# --------------------------
print("\n--- Resultados de la Búsqueda Aleatoria ---")
print(f"Mejor Accuracy en validación cruzada: {random_search.best_score_:.4f}")
print("Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

best_model = random_search.best_estimator_

# 6. Evaluar el modelo optimizado en el conjunto de prueba
# -------------------------------------------------------
print("\n--- Evaluación Final en el Conjunto de Prueba ---")
from sklearn.metrics import classification_report
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

print("\nConclusión: RandomizedSearchCV es una excelente herramienta para una exploración")
print("amplia y eficiente del espacio de hiperparámetros, especialmente como un primer")
print("paso para acotar las áreas más prometedoras, que luego podrían ser exploradas")
print("más a fondo con GridSearchCV si fuera necesario.")
