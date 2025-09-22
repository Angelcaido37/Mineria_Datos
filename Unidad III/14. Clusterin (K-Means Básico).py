# -*- coding: utf-8 -*-
"""
Tema: 7.1 Algoritmos de Clustering - K-Means
Dataset: Sintético (creado con make_blobs)
Explicación: K-Means es el algoritmo de clustering más conocido. Es un
algoritmo de aprendizaje no supervisado, lo que significa que no usamos
etiquetas (y). Su objetivo es agrupar los datos en un número predefinido
de clústeres (K) de tal manera que los puntos dentro de un mismo clúster
sean muy similares entre sí y los puntos en clústeres diferentes sean muy
distintos.
"""

# 1. Importar librerías
# ---------------------
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Generar datos sintéticos
# ---------------------------
# `make_blobs` es perfecto para generar "nubes" de puntos, ideal para K-Means.
# n_samples: número total de puntos.
# centers: número de clústeres a generar.
# cluster_std: desviación estándar de los clústeres (qué tan dispersos están).
# random_state: para reproducibilidad.
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# Es una buena práctica escalar los datos antes de aplicar K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Visualizar los datos originales
# ----------------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=50, alpha=0.7)
plt.title("Datos Originales (sin etiquetas)")
plt.xlabel("Característica 1 (escalada)")
plt.ylabel("Característica 2 (escalada)")
plt.grid(True)


# 4. Aplicar el algoritmo K-Means
# -------------------------------
# K-Means necesita que le digamos cuántos clústeres buscar.
# En este caso, como generamos los datos, sabemos que K=4 es el número correcto.
# En problemas reales, tendríamos que usar técnicas como el método del codo o silueta.
n_clusters = 4

# `init='k-means++'`: Es un método de inicialización inteligente que acelera la convergencia.
# `n_init=10`: El algoritmo se ejecutará 10 veces con diferentes centroides iniciales,
# y el resultado final será el mejor de esas 10 ejecuciones. Esto lo hace más robusto.
# `random_state=42`: Para que la inicialización sea reproducible.
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)

# `fit_predict` entrena el modelo y devuelve la etiqueta del clúster para cada punto.
y_kmeans = kmeans.fit_predict(X_scaled)

# 5. Analizar los resultados
# --------------------------
# `kmeans.cluster_centers_`: Coordenadas de los centroides finales de cada clúster.
centroids = kmeans.cluster_centers_

# `kmeans.inertia_`: Suma de las distancias al cuadrado de cada muestra a su centroide
# más cercano. Es una medida de qué tan compactos son los clústeres. Un valor bajo es mejor.
inertia = kmeans.inertia_
print(f"El valor de inercia (WCSS) para K={n_clusters} es: {inertia:.2f}")

# 6. Visualizar los clústeres encontrados
# ---------------------------------------
plt.subplot(1, 2, 2)
# `c=y_kmeans` colorea cada punto según la etiqueta de clúster que le asignó K-Means.
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.7)

# Dibujar los centroides
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroides')

plt.title(f"Clustering con K-Means (K={n_clusters})")
plt.xlabel("Característica 1 (escalada)")
plt.ylabel("Característica 2 (escalada)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nConclusión: K-Means ha identificado correctamente los 4 grupos de datos.")
print("Es un algoritmo rápido y eficiente, pero funciona mejor cuando los clústeres son")
print("de tamaño similar, esféricos y están bien separados.")
