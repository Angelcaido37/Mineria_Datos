# -*- coding: utf-8 -*-
"""
Tema: 4.2 Separación de Datos (Train/Validation/Test)
Dataset: Breast Cancer
Explicación: Una simple división train/test es buena, pero para un desarrollo
riguroso de modelos, especialmente cuando se ajustan hiperparámetros, se
necesita una tercera división: el conjunto de validación.

El flujo de trabajo es:
1.  **Conjunto de Entrenamiento (Train Set):** Se usa para entrenar el modelo.
2.  **Conjunto de Validación (Validation Set):** Se usa para ajustar los
    hiperparámetros del modelo (ej. encontrar el mejor 'K' en KNN o la
    mejor 'max_depth' en un árbol). El modelo se evalúa repetidamente en
    este conjunto para tomar decisiones de diseño.
3.  **Conjunto de Prueba (Test Set):** Se usa UNA SOLA VEZ, al final de todo
    el proceso, para obtener una estimación final e imparcial del rendimiento
    del modelo final. Este conjunto debe ser "tierra sagrada" y no tocarse
    durante el desarrollo.
"""

# 1. Importar librerías
# ---------------------
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 2. Cargar datos
# ---------------
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print(f"Tamaño total del dataset: {X.shape[0]} muestras")

# 3. Primera División: Separar el Conjunto de Prueba
# --------------------------------------------------
# El primer paso y más importante es aislar el conjunto de prueba.
# Lo guardaremos y no lo tocaremos hasta el final.
# Vamos a reservar un 20% para la prueba final.

# X_temp y y_temp contendrán el 80% restante de los datos para entrenamiento y validación.
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% para el test set
    random_state=42,
    stratify=y # Importante para mantener la proporción de clases
)

print("\n--- Primera División ---")
print(f"Tamaño del conjunto de Entrenamiento + Validación (temporal): {len(X_temp)} muestras ({len(X_temp)/len(X):.0%})")
print(f"Tamaño del conjunto de Prueba (final): {len(X_test)} muestras ({len(X_test)/len(X):.0%})")

# 4. Segunda División: Separar Entrenamiento de Validación
# -------------------------------------------------------
# Ahora, tomamos el 80% restante (X_temp, y_temp) y lo dividimos
# de nuevo para crear nuestros conjuntos de entrenamiento y validación.
# Queremos que el conjunto de validación sea el 25% de este 80% temporal.
# 25% de 80% es 20% del total original.
# Así, la división final será: 60% train, 20% validation, 20% test.

val_size_adjusted = 0.25 # 25% del conjunto temporal (X_temp)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=val_size_adjusted,
    random_state=42,
    stratify=y_temp
)


print("\n--- Segunda División (a partir del 80% temporal) ---")
print(f"Tamaño del conjunto de Entrenamiento (final): {len(X_train)} muestras ({len(X_train)/len(X):.0%})")
print(f"Tamaño del conjunto de Validación (final): {len(X_val)} muestras ({len(X_val)/len(X):.0%})")


print("\n--- Resumen de la División Final ---")
total_samples = len(X)
print(f"Total: {total_samples} muestras")
print(f"  - Entrenamiento: {len(y_train)} ({len(y_train)/total_samples:.2f}) -> Usado para `model.fit()`")
print(f"  - Validación:    {len(y_val)} ({len(y_val)/total_samples:.2f}) -> Usado para ajustar hiperparámetros")
print(f"  - Prueba:        {len(y_test)} ({len(y_test)/total_samples:.2f}) -> Usado para la evaluación final")


# 5. Ejemplo de uso (simulado)
# ----------------------------
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score
#
# # 1. Entrenar en el set de entrenamiento
# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train, y_train)
#
# # 2. Evaluar y ajustar en el set de validación
# # (Aquí haríamos un bucle para probar diferentes valores de K,
# # y elegiríamos el que mejor funcione en el validation set)
# y_pred_val = model.predict(X_val)
# accuracy_val = accuracy_score(y_val, y_pred_val)
# print(f"\nRendimiento en el conjunto de validación: {accuracy_val:.4f}")
#
# # 3. Una vez que hemos elegido el mejor modelo y sus hiperparámetros,
# # lo evaluamos por última vez en el test set.
# # (Opcionalmente, re-entrenamos el mejor modelo con train + validation)
# final_model = KNeighborsClassifier(n_neighbors=7) # Supongamos que k=7 fue el mejor
# final_model.fit(np.vstack((X_train, X_val)), np.concatenate((y_train, y_val)))
# y_pred_test = final_model.predict(X_test)
# accuracy_test = accuracy_score(y_test, y_pred_test)
# print(f"Rendimiento FINAL e IMPARCIAL en el conjunto de prueba: {accuracy_test:.4f}")
#
# Este resultado es el que reportaríamos como el rendimiento esperado del modelo.
