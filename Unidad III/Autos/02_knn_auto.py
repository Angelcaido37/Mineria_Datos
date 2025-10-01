# -*- coding: utf-8 -*-
"""
02_knn_auto.py (versión académica, corregida y comentada)

Este script implementa el algoritmo K-Nearest Neighbors (KNN) con:
- Carga del dataset /Datasets/heart.csv
- División en entrenamiento y prueba
- Detección de desbalance y selección automática de métrica (f1_macro o f1_weighted)
- Pipeline con escalado de variables (StandardScaler) y KNN
- Búsqueda de hiperparámetros mediante GridSearchCV
- Evaluación final en el conjunto de prueba
- Gráficas: Matriz de Confusión y Curva ROC
"""

# ==== IMPORTACIONES BÁSICAS ====
import numpy as np                                      # Para cálculos numéricos
import pandas as pd                                     # Para manejo de DataFrames
from pathlib import Path                                # Para manejar rutas de archivos
import matplotlib.pyplot as plt                         # Para graficar

# ==== UTILIDADES DE SKLEARN ====
from collections import Counter                         # Para medir balance de clases
from sklearn.model_selection import train_test_split, GridSearchCV  # Para dividir datos y buscar hiperparámetros
from sklearn.pipeline import Pipeline                   # Para encadenar procesos (escalado + modelo)
from sklearn.preprocessing import StandardScaler, label_binarize    # Para normalización y binarización de etiquetas
from sklearn.neighbors import KNeighborsClassifier      # Algoritmo KNN
from sklearn.metrics import (                           # Métricas de evaluación
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc
)

# ==== PARÁMETROS DE ENTRADA ====
CSV_PATH = Path("Datasets/heart.csv")                  # Ruta del dataset Heart
TARGET_COL = "target"                                   # Nombre de la columna objetivo

# ==== PROGRAMA PRINCIPAL ====
if __name__ == "__main__":
    # 1. Cargar dataset
    assert CSV_PATH.exists(), f"No se encontró {CSV_PATH}"   # Validar existencia del archivo
    df = pd.read_csv(CSV_PATH)                               # Leer el CSV con pandas
    assert TARGET_COL in df.columns, f"Falta columna {TARGET_COL}"  # Validar columna objetivo
    
    X = df.drop(columns=[TARGET_COL])                        # Variables independientes
    y = df[TARGET_COL]                                       # Variable objetivo

    # 2. Separar en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42    # Estratificación conserva balance de clases
    )

    # 3. Medir desbalance y elegir métrica adecuada
    class_counts = Counter(y_train)                          # Contar instancias de cada clase
    total = sum(class_counts.values())                       # Número total de ejemplos
    minority_ratio = min(class_counts.values()) / total      # Proporción clase minoritaria
    chosen_scoring = "f1_macro" if minority_ratio < 0.30 else "f1_weighted"  # Selección automática
    print(f"Métrica elegida automáticamente para CV: {chosen_scoring} "
          f"(minority_ratio={minority_ratio:.3f}, counts={dict(class_counts)})")

    # 4. Definir pipeline (normalización + KNN)
    pipe = Pipeline([
        ("scaler", StandardScaler()),                        # Normalizar variables
        ("knn", KNeighborsClassifier())                      # Clasificador KNN
    ])

    # 5. Definir rejilla de hiperparámetros
    param_grid = {
        "knn__n_neighbors": list(range(1, 41, 2)),           # Número de vecinos (k)
        "knn__weights": ["uniform", "distance"],             # Uniforme o ponderado por distancia
        "knn__metric": ["euclidean", "manhattan", "minkowski"], # Tipos de distancia
        "knn__p": [1, 2]                                     # Parámetro p para Minkowski
    }

    # 6. Buscar mejor configuración con GridSearchCV
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=chosen_scoring,                              # Métrica seleccionada
        cv=5,                                                # Validación cruzada de 5 folds
        n_jobs=-1,                                           # Usar todos los núcleos disponibles
        verbose=1,                                           # Mostrar progreso
        refit=True                                           # Reentrenar con el mejor modelo
    )
    grid.fit(X_train, y_train)                               # Entrenar con búsqueda

    # 7. Resultados de la búsqueda
    print("Mejores hiperparámetros:", grid.best_params_)     # Mostrar la mejor combinación
    print("Mejor puntaje (CV):", round(grid.best_score_, 3)) # Mostrar puntaje promedio CV

    # 8. Evaluar en conjunto de prueba
    best_model = grid.best_estimator_                        # Extraer mejor modelo entrenado
    y_pred = best_model.predict(X_test)                      # Hacer predicciones en test

    # Calcular métricas principales
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Test -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

    # 9. Gráfica: Matriz de Confusión (corregida)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)  # Crear matriz
    disp.ax_.set_title("Matriz de Confusión (Test)")                # Agregar título
    plt.tight_layout()
    plt.show()

    # 10. Gráfica: Curva ROC
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)           # Si el modelo soporta probabilidades
    elif hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(X_test)       # Alternativa
    else:
        y_score = None                                       # No disponible

    if y_score is not None:
        classes = np.unique(y_test)                          # Extraer clases
        if len(classes) == 2:                                # Caso binario
            y_true = (y_test == classes.max()).astype(int)   # Codificar clase positiva
            prob_pos = y_score[:, 1] if y_score.ndim == 2 else y_score
            fpr, tpr, _ = roc_curve(y_true, prob_pos)        # Calcular curva ROC
            roc_auc = auc(fpr, tpr)                          # Área bajo la curva
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("Curva ROC (Test)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        else:                                                # Caso multiclase (One-vs-Rest)
            Yb = label_binarize(y_test, classes=classes)     # Binarizar etiquetas
            if y_score.ndim > 1:
                plt.figure(figsize=(6, 5))
                for i in range(Yb.shape[1]):
                    fpr, tpr, _ = roc_curve(Yb[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Clase {classes[i]} (AUC={roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("Curvas ROC One-vs-Rest (Test)")
                plt.legend()
                plt.tight_layout()
                plt.show()
