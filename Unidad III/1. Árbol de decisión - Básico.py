# -*- coding: utf-8 -*-
"""
Tema: 2.1 Árboles de Decisión - Implementación Básica
Dataset: Iris (un clásico para empezar)
Explicación: Este script muestra los pasos fundamentales para entrenar y evaluar
un modelo de Árbol de Decisión. Es el punto de partida para entender cómo
funciona el algoritmo.
"""

# 1. Importar las librerías necesarias
# ------------------------------------
# sklearn.datasets: para cargar datasets de ejemplo
# sklearn.model_selection: para dividir los datos en entrenamiento y prueba
# sklearn.tree: contiene la implementación del Árbol de Decisión
# sklearn.metrics: para evaluar el rendimiento del modelo
# matplotlib.pyplot y sklearn.tree.plot_tree: para visualizar el árbol
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

# 2. Cargar el conjunto de datos
# ------------------------------------
# Usaremos el dataset 'Iris', que contiene mediciones de 3 especies de flores.
# Es ideal para problemas de clasificación multiclase.
iris = load_iris()
X = iris.data  # Características (largo y ancho de sépalo y pétalo)
y = iris.target  # Etiquetas (la especie de la flor: 0, 1, o 2)

# Para entender mejor los datos, los convertimos a un DataFrame de Pandas
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y
print("Primeras 5 filas del dataset Iris:")
print(df.head())
print("\nDescripción del dataset:")
print(df.describe())
print("\nDistribución de especies:")
print(df['species'].value_counts())


# 3. Separar los datos en conjuntos de entrenamiento y prueba
# -----------------------------------------------------------
# Es una práctica CRUCIAL en machine learning.
# Entrenamos el modelo con una parte de los datos (entrenamiento) y lo
# evaluamos con datos que nunca ha visto (prueba) para medir su
# capacidad de generalización.
# test_size=0.3: Usamos el 30% de los datos para el conjunto de prueba.
# random_state=42: Asegura que la división sea siempre la misma,
# lo que hace nuestros resultados reproducibles.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")


# 4. Crear y entrenar el modelo de Árbol de Decisión
# ----------------------------------------------------
# Instanciamos el clasificador. `DecisionTreeClassifier` tiene muchos
# parámetros (hiperparámetros) que podemos ajustar. Por ahora, usaremos
# uno simple.
# random_state=42: de nuevo, para reproducibilidad.
clf = DecisionTreeClassifier(random_state=42)

# El método `fit` es donde ocurre el "aprendizaje". El algoritmo analiza
# los datos de entrenamiento (X_train) y sus etiquetas (y_train) para
# construir el árbol de decisión.
print("\nEntrenando el modelo de Árbol de Decisión...")
clf.fit(X_train, y_train)
print("¡Modelo entrenado exitosamente!")


# 5. Realizar predicciones sobre el conjunto de prueba
# -----------------------------------------------------
# Usamos el modelo ya entrenado (`clf`) para predecir las especies de las
# flores en el conjunto de prueba, del cual solo le damos las características (X_test).
y_pred = clf.predict(X_test)


# 6. Evaluar el rendimiento del modelo
# -------------------------------------
# Comparamos las predicciones del modelo (y_pred) con las etiquetas reales (y_test).

# Accuracy (Exactitud): Es el porcentaje de predicciones correctas.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Exactitud) del modelo: {accuracy:.4f}")

# Matriz de Confusión: Nos dice qué clases el modelo confunde.
# Las filas son los valores reales, las columnas son las predicciones.
# La diagonal principal muestra las predicciones correctas.
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(cm)

# Reporte de Clasificación: Muestra métricas más detalladas por cada clase.
# - Precision: De todas las veces que el modelo predijo una clase, ¿cuántas acertó?
# - Recall (Sensibilidad): De todas las instancias reales de una clase, ¿cuántas identificó correctamente el modelo?
# - F1-score: Es la media armónica de precision y recall.
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# 7. Visualizar el Árbol de Decisión
# -----------------------------------
# Una de las grandes ventajas de los árboles es que son fáciles de interpretar.
# Podemos visualizar las reglas que el modelo ha aprendido.
plt.figure(figsize=(20, 10))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True, # Colorea los nodos según la clase mayoritaria
          rounded=True, # Nodos con bordes redondeados
          fontsize=10)
plt.title("Árbol de Decisión entrenado con el dataset Iris", fontsize=16)
plt.show()

# Visualizar la Matriz de Confusión con Seaborn para mayor claridad
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()
