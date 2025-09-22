# -*- coding: utf-8 -*-
"""
Tema: 2.2 K-Nearest Neighbors (KNN) - Encontrar el valor óptimo de K
Dataset: Breast Cancer (Cáncer de Mama)
Explicación: La elección del hiperparámetro 'K' (el número de vecinos) es
crucial para el rendimiento de KNN.
- Un K muy pequeño (ej. K=1) hace al modelo muy sensible al ruido (alto sobreajuste).
- Un K muy grande (ej. K=N) hace al modelo muy simple y puede no capturar la
  estructura de los datos (alto subajuste).
Este script implementa el "método del codo" para encontrar un buen valor de K,
iterando sobre un rango de valores y graficando su rendimiento.
"""

# 1. Importar librerías
# ---------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 2. Cargar y preparar los datos
# ------------------------------
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos (esencial para KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Búsqueda del K óptimo
# --------------------------
# Definimos un rango de valores de K para probar.
# Empezamos en 1 y llegamos hasta 30.
k_range = range(1, 31)

# Lista para almacenar la exactitud (accuracy) para cada valor de K.
accuracy_scores = []
error_rates = []

print("Buscando el valor óptimo de K...")
# Iteramos sobre cada valor de K en el rango definido
for k in k_range:
    # Creamos una instancia de KNN con el valor actual de k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Entrenamos el modelo
    knn.fit(X_train_scaled, y_train)
    
    # Hacemos predicciones en el conjunto de prueba
    y_pred = knn.predict(X_test_scaled)
    
    # Calculamos la exactitud y la almacenamos
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
    
    # También es común analizar la tasa de error (1 - accuracy)
    error_rates.append(1 - accuracy)
    
    print(f"Para K = {k}, Accuracy = {accuracy:.4f}, Tasa de Error = {1-accuracy:.4f}")

# 4. Encontrar el mejor K y el score correspondiente
# -------------------------------------------------
# np.argmax(accuracy_scores) nos da el ÍNDICE del valor más alto en la lista.
# Sumamos 1 porque los índices empiezan en 0, pero nuestro rango de K empezó en 1.
best_k = np.argmax(accuracy_scores) + 1
best_accuracy = max(accuracy_scores)

print(f"\nEl valor óptimo de K es {best_k} con un Accuracy de {best_accuracy:.4f}")

# 5. Visualizar los resultados (Método del Codo)
# ----------------------------------------------
# Graficamos el rendimiento (Accuracy vs. K) para visualizar el "codo".
# El "codo" es el punto donde la mejora en el rendimiento comienza a aplanarse.
# A menudo, este punto representa un buen equilibrio entre sesgo y varianza.

plt.figure(figsize=(12, 5))

# Gráfico de Accuracy vs. K
plt.subplot(1, 2, 1)
plt.plot(k_range, accuracy_scores, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Accuracy vs. Valor de K')
plt.xlabel('K (Número de Vecinos)')
plt.ylabel('Accuracy')
# Marcamos el mejor punto
plt.axvline(x=best_k, color='green', linestyle='--', label=f'Mejor K = {best_k}')
plt.legend()
plt.grid(True)


# Gráfico de Tasa de Error vs. K
plt.subplot(1, 2, 2)
plt.plot(k_range, error_rates, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Tasa de Error vs. Valor de K')
plt.xlabel('K (Número de Vecinos)')
plt.ylabel('Tasa de Error')
# Buscamos el punto más bajo en la tasa de error
best_k_error = np.argmin(error_rates) + 1
plt.axvline(x=best_k_error, color='green', linestyle='--', label=f'K con menor error = {best_k_error}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nAnálisis del gráfico:")
print("Observa cómo el Accuracy sube rápidamente y luego se estabiliza o incluso baja.")
print("El valor de K donde se alcanza el máximo rendimiento (o el mínimo error) antes de que la curva se aplane es una buena elección.")
