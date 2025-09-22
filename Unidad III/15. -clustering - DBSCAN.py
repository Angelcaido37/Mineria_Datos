# -*- coding: utf-8 -*-
"""
Tema: 7.2 Algoritmos de Clustering - DBSCAN
Dataset: Sintético (creado con make_moons)
Explicación: K-Means falla en encontrar clústeres que no son esféricos.
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) es un
algoritmo basado en densidad que puede encontrar clústeres de formas
arbitrarias. Sus ventajas principales son:
1. No necesita que se especifique el número de clústeres de antemano.
2. Puede identificar puntos como ruido (outliers).

Funciona con dos parámetros clave:
- `eps`: La distancia máxima entre dos muestras para que una se considere
         en la vecindad de la otra.
- `min_samples`: El número de muestras en una vecindad para que un punto
                 sea considerado como un punto central (core point).
"""
# 1. Importar librerías
# ---------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# 2. Generar datos no esféricos
# -----------------------------
# `make_moons` crea dos "medias lunas" entrelazadas. K-Means tendrá problemas con esto.
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# Escalar los datos es una buena práctica para DBSCAN también.
X_scaled = StandardScaler().fit_transform(X)

# 3. Aplicar K-Means (para comparación)
# -------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# 4. Aplicar DBSCAN
# -----------------
# La elección de `eps` y `min_samples` es crucial.
# Una regla general es empezar con min_samples = 2 * num_dimensiones. Aquí 2*2=4.
# `eps` puede encontrarse con técnicas como el "k-distance graph", pero por ahora
# lo elegiremos por experimentación.
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

# Analizar los resultados de DBSCAN
# `labels_` contiene la etiqueta del clúster para cada punto.
# La etiqueta -1 se asigna a los puntos considerados como ruido.
labels = dbscan.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f"Resultados de DBSCAN:")
print(f"Número de clústeres estimados: {n_clusters_}")
print(f"Número de puntos de ruido: {n_noise_}")

# 5. Visualizar los resultados
# ----------------------------
plt.figure(figsize=(18, 6))

# Gráfico 1: Datos Originales
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=50, alpha=0.7)
plt.title("Datos Originales ('make_moons')")
plt.grid(True)

# Gráfico 2: Resultado de K-Means
plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.7)
plt.title("Resultado de K-Means (Falla)")
plt.grid(True)

# Gráfico 3: Resultado de DBSCAN
plt.subplot(1, 3, 3)
# Colorear los puntos según el clúster, y marcar el ruido.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Ruido en color negro
        col = [0, 0, 0, 1]
    
    class_member_mask = (labels == k)
    xy = X_scaled[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=7, alpha=0.8)

plt.title(f'Resultado de DBSCAN (Éxito)\nClusters: {n_clusters_}, Ruido: {n_noise_}')
plt.grid(True)

plt.show()

print("\nConclusión: Mientras K-Means simplemente divide el espacio por la mitad,")
print("DBSCAN es capaz de seguir la densidad de los puntos y descubrir")
print("los clústeres con forma de media luna, demostrando su poder en")
print("conjuntos de datos más complejos.")
