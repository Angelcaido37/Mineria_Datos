# -*- coding: utf-8 -*-
"""
Tema: 8.1 Métodos para Determinar Número Óptimo de Clústeres - Método del Codo
Dataset: Sintético
Explicación: El mayor desafío de K-Means es saber cuántos clústeres (K)
buscar. El Método del Codo (Elbow Method) es una heurística popular para
encontrar un buen valor de K.
La idea es ejecutar K-Means para un rango de valores de K (ej. de 1 a 10)
y para cada K, calcular la "Inercia" o WCSS (Within-Cluster Sum of Squares).
WCSS es la suma de las distancias al cuadrado entre cada punto y su centroide.
Luego, se grafica K vs. WCSS. La gráfica se verá como un brazo. El "codo" de
ese brazo es el punto donde la tasa de disminución de WCSS se reduce
drásticamente, y es un buen indicador del K óptimo.
"""
# 1. Importar librerías
# ---------------------
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator # Una librería útil para encontrar el codo automáticamente

# 2. Generar datos
# ----------------
X, y = make_blobs(n_samples=500, centers=5, cluster_std=1.0, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# 3. Calcular WCSS para un rango de valores de K
# ----------------------------------------------
wcss = []
k_range = range(1, 11) # Probaremos K de 1 a 10

print("Calculando WCSS para diferentes valores de K...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    # `inertia_` en scikit-learn es el valor de WCSS
    wcss.append(kmeans.inertia_)
    print(f"K = {k}, WCSS = {kmeans.inertia_:.2f}")

# 4. Graficar el Método del Codo
# ------------------------------
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Número de Clústeres (K)')
plt.ylabel('WCSS (Inercia)')
plt.title('Método del Codo para Encontrar K Óptimo')
plt.grid(True)

# 5. Encontrar el punto del codo automáticamente
# ----------------------------------------------
# La librería `kneed` nos ayuda a localizar el codo de forma programática.
# S: Sensibilidad de la detección.
# curve: 'convex' porque la curva es convexa.
# direction: 'decreasing' porque los valores de WCSS disminuyen.
kneedle = KneeLocator(k_range, wcss, S=1.0, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow

print(f"\nEl punto del codo detectado automáticamente es K = {optimal_k}")

# Marcar el codo en la gráfica
if optimal_k:
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Codo en K = {optimal_k}')
    plt.legend()

plt.show()

# 6. Interpretación
# -----------------
print("\n--- ANÁLISIS DEL GRÁFICO ---")
print("Observa que al pasar de K=1 a K=2, hay una caída masiva en WCSS. Lo mismo de K=2 a K=3.")
print("La disminución sigue siendo significativa hasta K=5.")
print(f"Después de K={optimal_k}, la curva se aplana. Agregar más clústeres ya no reduce WCSS")
print("de manera tan efectiva. Esto sugiere que estamos dividiendo clústeres que ya son cohesivos.")
print("Por lo tanto, elegimos el valor en el 'codo' como el número óptimo de clústeres.")
