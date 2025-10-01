# -*- coding: utf-8 -*-
"""
05_kmeans_auto_k.py (versión académica con comentarios)

Este script implementa K-Means (clustering no supervisado) con:
- Carga del dataset /Datasets/heart.csv (sin la columna target)
- Normalización de datos
- Selección automática de k mediante Silhouette y método del Codo
- Entrenamiento con mejor k
- Gráficas: Silhouette vs k, Método del Codo (WCSS) y Dispersión PCA 2D
"""

# ==== IMPORTACIONES BÁSICAS ====
import numpy as np                                     # Para cálculos numéricos
import pandas as pd                                    # Manejo de DataFrames
import matplotlib.pyplot as plt                        # Para graficar resultados
from pathlib import Path                               # Manejo de rutas

# ==== MODELOS Y MÉTRICAS ====
from sklearn.cluster import KMeans                     # Algoritmo K-Means
from sklearn.preprocessing import StandardScaler       # Escalado de variables
from sklearn.metrics import silhouette_score           # Métrica Silhouette
from sklearn.decomposition import PCA                  # Reducción de dimensionalidad (PCA)

# ==== PARÁMETROS DEL DATASET ====
CSV_PATH = Path("Datasets/heart.csv")                 # Ruta del dataset Heart
TARGET_COL = "target"                                  # Columna objetivo (no usada en clustering)


# ==== FUNCIÓN AUXILIAR: Método del Codo ====
def elbow_k(X_scaled, max_k: int = 12) -> int:
    """
    Calcula el mejor número de clusters k usando el método del Codo.
    - X_scaled: dataset ya escalado
    - max_k: máximo número de clusters a evaluar
    Retorna el valor de k en el "codo".
    """
    ks = range(1, max_k + 1)                           # Valores de k a probar
    wcss = []                                          # Lista de inercia (WCSS)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)                               # Entrenar K-Means
        wcss.append(km.inertia_)                       # Guardar inercia

    # Si no hay suficientes valores para calcular segunda derivada, devolver 2
    if len(wcss) < 3:
        return 2

    # Calcular segunda derivada discreta de WCSS para encontrar el "codo"
    second = []
    for i in range(1, len(wcss) - 1):
        second.append(wcss[i-1] - 2*wcss[i] + wcss[i+1])

    elbow_idx = int(np.argmax(second)) + 1             # Índice del máximo cambio
    return ks[elbow_idx]


# ==== PROGRAMA PRINCIPAL ====
if __name__ == "__main__":
    # 1. Cargar dataset
    assert CSV_PATH.exists(), f"No se encontró {CSV_PATH}"   # Validar existencia del archivo
    df = pd.read_csv(CSV_PATH)                               # Leer dataset
    assert TARGET_COL in df.columns, f"Falta columna {TARGET_COL}"  # Validar target
    X = df.drop(columns=[TARGET_COL])                        # Usamos todas las columnas excepto target

    # 2. Escalar datos
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)                             # Escalar todas las variables

    # 3. Calcular índice de Silhouette para distintos k
    sil_scores = {}
    for k in range(2, 13):                                   # Evaluar k=2..12
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        sil_scores[k] = silhouette_score(Xs, labels)         # Guardar puntaje de Silhouette

    # 4. Determinar mejor k por Silhouette y Codo
    k_sil = max(sil_scores, key=sil_scores.get)              # Mejor k según Silhouette
    k_elbow = elbow_k(Xs, max_k=12)                          # Mejor k según Codo
    candidates = sorted(set([k_sil, k_elbow]))               # Candidatos finales

    # 5. Entrenar con el mejor modelo entre candidatos
    best_k, best_inertia, best_model = None, np.inf, None
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        if km.inertia_ < best_inertia:                       # Escoger el de menor inercia
            best_k, best_inertia, best_model = k, km.inertia_, km

    # 6. Imprimir resultados finales
    print(f"k (silueta): {k_sil}  |  k (codo): {k_elbow}  |  k elegido: {best_k}")
    print(f"Inercia (WCSS) con k={best_k}: {best_inertia:.2f}")
    print(f"Iteraciones hasta convergencia: {best_model.n_iter_}")

    # 7. Gráfica: Silhouette vs k
    plt.figure(figsize=(6, 4))
    ks_sorted = sorted(sil_scores.keys())
    plt.plot(ks_sorted, [sil_scores[k] for k in ks_sorted], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Silhouette por k")
    plt.tight_layout()
    plt.show()

    # 8. Gráfica: Método del Codo (WCSS)
    plt.figure(figsize=(6, 4))
    wcss = []
    ks = range(1, 13)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        wcss.append(km.inertia_)
    plt.plot(list(ks), wcss, marker="o")
    plt.xlabel("k")
    plt.ylabel("WCSS (inercia)")
    plt.title("Método del Codo")
    plt.tight_layout()
    plt.show()

    # 9. Gráfica: Dispersión PCA 2D con clusters asignados
    labels = best_model.predict(Xs)                          # Etiquetas de cluster
    pca = PCA(n_components=2, random_state=42)               # Reducir a 2D
    X2 = pca.fit_transform(Xs)                               # Transformar datos
    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.title(f"K-Means (k={best_k}) - PCA 2D")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.tight_layout()
    plt.show()
