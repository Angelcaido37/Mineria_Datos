# -*- coding: utf-8 -*-
"""
Tema: 8.2 Métodos para Determinar Número Óptimo de Clústeres - Método de la Silueta
Dataset: Sintético
Explicación: El Coeficiente de Silueta (Silhouette Score) es otra métrica
para evaluar la calidad de un clustering, y por tanto, para encontrar el
K óptimo. A diferencia del método del codo, no solo mide qué tan juntos
están los puntos en un clúster, sino también qué tan separados están de los
puntos de otros clústeres.

El score va de -1 a +1:
- **+1:** Indica que el punto está lejos de los clústeres vecinos. (Muy bueno)
- **0:** Indica que el punto está muy cerca o en el límite de un clúster vecino.
- **-1:** Indica que el punto podría haber sido asignado al clúster incorrecto.

Buscamos el valor de K que maximice el Coeficiente de Silueta promedio.
"""

# 1. Importar librerías
# ---------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# 2. Generar datos
# ----------------
X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.7, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 3. Calcular el score de silueta para un rango de K
# --------------------------------------------------
# El score de silueta solo se puede calcular para K >= 2
k_range = range(2, 11)
silhouette_scores = []

print("Calculando el score de silueta para diferentes K...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calcular el score de silueta promedio para todas las muestras
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Para K = {k}, el score de silueta promedio es: {silhouette_avg:.4f}")

# 4. Encontrar el K óptimo
# ------------------------
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nEl K óptimo según el método de la silueta es: K = {optimal_k}")

# 5. Graficar los scores de silueta
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.xlabel("Número de Clústeres (K)")
plt.ylabel("Score de Silueta Promedio")
plt.title("Método de la Silueta para Encontrar K Óptimo")
plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'K Óptimo = {optimal_k}')
plt.legend()
plt.grid(True)
plt.show()

# 6. Visualización detallada de la Silueta para el K óptimo
# ---------------------------------------------------------
# Esta visualización más avanzada nos muestra el score de silueta para cada
# punto, agrupado por clúster. Nos permite ver si algunos clústeres son
# mejores que otros.
print(f"\nGenerando visualización detallada para K = {optimal_k}...")
kmeans_best = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
best_labels = kmeans_best.fit_predict(X_scaled)
sample_silhouette_values = silhouette_samples(X_scaled, best_labels)

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(10, 7)
ax1.set_xlim([-0.2, 1])
ax1.set_ylim([0, len(X_scaled) + (optimal_k + 1) * 10])

y_lower = 10
for i in range(optimal_k):
    ith_cluster_silhouette_values = sample_silhouette_values[best_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / optimal_k)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax1.set_title(f"Gráfico de Silueta para K = {optimal_k}")
ax1.set_xlabel("Valores del coeficiente de silueta")
ax1.set_ylabel("Etiqueta del clúster")

# Línea vertical para el score de silueta promedio
silhouette_avg_best = silhouette_score(X_scaled, best_labels)
ax1.axvline(x=silhouette_avg_best, color="red", linestyle="--")
ax1.set_yticks([])
plt.show()

print("\nInterpretación: El método de la silueta a menudo es más robusto que el del codo,")
print("ya que considera tanto la cohesión (qué tan juntos están los puntos en un clúster)")
print("como la separación (qué tan lejos están los clústeres entre sí).")
