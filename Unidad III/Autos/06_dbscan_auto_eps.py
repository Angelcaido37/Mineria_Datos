# -*- coding: utf-8 -*-
"""
06_dbscan_auto_eps.py (versión académica y comentada)

Este script implementa DBSCAN (Density-Based Spatial Clustering of Applications with Noise) con:
- Carga del dataset /Datasets/heart.csv (sin columna target)
- Normalización de datos
- Estimación automática del parámetro eps mediante curva k-distance
- Entrenamiento de DBSCAN
- Reporte del número de clusters y porcentaje de ruido
- Gráficas: Curva k-distance y Dispersión PCA 2D
"""

# ==== IMPORTACIONES BÁSICAS ====
import numpy as np                                     # Librería numérica
import pandas as pd                                    # Para leer el dataset
import matplotlib.pyplot as plt                        # Para graficar
from pathlib import Path                               # Manejo seguro de rutas

# ==== MODELOS Y UTILIDADES ====
from sklearn.cluster import DBSCAN                     # Algoritmo DBSCAN
from sklearn.neighbors import NearestNeighbors         # Para estimar eps con k-distance
from sklearn.preprocessing import StandardScaler       # Escalado de variables
from sklearn.decomposition import PCA                  # Reducción de dimensionalidad (PCA)

# ==== PARÁMETROS DEL DATASET ====
CSV_PATH = Path("Datasets/heart.csv")                 # Ruta al dataset Heart
TARGET_COL = "target"                                  # Columna objetivo (no se usa en clustering)


# ==== FUNCIÓN AUXILIAR: sugerencia automática de eps ====
def suggest_eps(Xs, min_samples: int = 10):
    """
    Sugiere un valor para eps (radio de vecindad en DBSCAN) usando la curva k-distance.
    - Xs: dataset escalado
    - min_samples: número mínimo de vecinos
    Retorna: eps sugerido y la lista ordenada de distancias k-ésimas.
    """
    nn = NearestNeighbors(n_neighbors=min_samples)     # Inicializar buscador de vecinos
    nn.fit(Xs)                                         # Ajustar sobre los datos
    distances, _ = nn.kneighbors(Xs)                   # Calcular distancias a k vecinos
    d = np.sort(distances[:, -1])                      # Ordenar distancias al k-ésimo vecino

    # Si los datos son muy pequeños, usar percentil 90
    if len(d) < 3:
        return float(np.percentile(d, 90)), d

    # Calcular segunda derivada discreta para detectar codo
    second = np.diff(d, 2)
    idx = int(np.argmax(second)) + 2                   # Índice del punto de mayor curvatura
    eps = float(d[idx])                                # eps sugerido
    return eps, d


# ==== PROGRAMA PRINCIPAL ====
if __name__ == "__main__":
    # 1. Cargar dataset
    assert CSV_PATH.exists(), f"No se encontró {CSV_PATH}"   # Validar existencia
    df = pd.read_csv(CSV_PATH)                               # Leer dataset
    assert TARGET_COL in df.columns, f"Falta columna {TARGET_COL}"  # Validar columna objetivo
    X = df.drop(columns=[TARGET_COL])                        # Quitar target para clustering

    # 2. Escalar datos
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)                             # Normalizar variables

    # 3. Sugerir eps mediante curva k-distance
    min_samples = 10                                         # Valor típico de min_samples
    eps, d_sorted = suggest_eps(Xs, min_samples=min_samples)
    print(f"Sugerencia de eps: {eps:.3f} con min_samples={min_samples}")

    # 4. Gráfica: Curva k-distance
    plt.figure(figsize=(6, 4))
    plt.plot(d_sorted)                                       # Distancias ordenadas
    plt.ylabel("Distancia al k-ésimo vecino")
    plt.xlabel("Punto ordenado")
    plt.title("Curva k-distance (para sugerir eps)")
    plt.tight_layout()
    plt.show()

    # 5. Entrenar DBSCAN con eps sugerido
    db = DBSCAN(eps=eps, min_samples=min_samples)            # Definir modelo DBSCAN
    labels = db.fit_predict(Xs)                              # Ajustar y obtener etiquetas

    # 6. Calcular estadísticas de clustering
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Número de clusters sin ruido
    n_noise = int(np.sum(labels == -1))                        # Cantidad de puntos ruido
    noise_pct = 100.0 * n_noise / len(labels)                  # Porcentaje de ruido
    print(f"Número de clusters: {n_clusters}")
    print(f"Ruido: {n_noise} puntos ({noise_pct:.1f}%)")

    # 7. Gráfica: Dispersión PCA 2D con clusters asignados
    pca = PCA(n_components=2, random_state=42)               # Reducir a 2D
    X2 = pca.fit_transform(Xs)                               # Proyectar datos
    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="plasma", s=50, alpha=0.7)
    plt.title(f"DBSCAN (eps={eps:.3f}, min_samples={min_samples}) - PCA 2D")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.tight_layout()
    plt.show()
