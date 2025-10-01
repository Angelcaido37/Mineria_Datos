# -*- coding: utf-8 -*-
"""
Árbol de Decisión (clasificación) con optimización automática sobre /Datasets/heart.csv
Incluye:
- Carga del dataset Heart
- Detección de desbalance de clases
- Selección automática de la métrica de evaluación
- Búsqueda de hiperparámetros (GridSearchCV)
- Evaluación del mejor modelo
- Gráficas (Árbol, Matriz de Confusión, Curva ROC)
"""

# ==== IMPORTACIONES ====
import numpy as np                                   # Librería para manejo numérico (arrays y operaciones matemáticas)
import pandas as pd                                  # Librería para manejo de DataFrames y lectura de CSV
from pathlib import Path                             # Para trabajar con rutas de archivos de manera segura
import matplotlib.pyplot as plt                      # Librería para realizar gráficas

from collections import Counter                      # Para contar elementos (útil en desbalance de clases)
from sklearn.model_selection import train_test_split # Para dividir el dataset en train/test
from sklearn.model_selection import GridSearchCV     # Para búsqueda exhaustiva de hiperparámetros
from sklearn.tree import DecisionTreeClassifier, plot_tree  # Árbol de decisión y función para graficarlo
from sklearn.pipeline import Pipeline                # Para armar pipelines y facilitar la validación
from sklearn.metrics import (                        # Métricas para evaluar el modelo
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize     # Para preparar datos multiclase en la curva ROC

# ==== PARÁMETROS DEL DATASET ====
CSV_PATH = Path("Datasets/heart.csv")               # Ruta al archivo CSV (ajustada a lo que pediste)
TARGET_COL = "target"                                # Columna que representa la variable objetivo (0/1)

# ==== FUNCIONES AUXILIARES ====
def cargar_dataset_heart(csv_path: Path, target_col: str):
    """Carga el CSV y separa variables predictoras (X) y objetivo (y)."""
    df = pd.read_csv(csv_path)                       # Leemos el CSV con pandas
    assert target_col in df.columns, f"Falta columna '{target_col}'" # Validamos que exista la columna objetivo
    y = df[target_col]                               # Extraemos la variable objetivo
    X = df.drop(columns=[target_col])                # Eliminamos la columna objetivo de las variables predictoras
    return X, y                                      # Retornamos X e y

def elegir_metricas_por_desbalance(y_train):
    """Detecta desbalance y selecciona métrica de validación adecuada."""
    class_counts = Counter(y_train)                  # Contamos la frecuencia de cada clase
    total = sum(class_counts.values())               # Número total de observaciones
    minority_ratio = min(class_counts.values()) / total # Calculamos proporción de la clase minoritaria
    # Regla: si <30% minoritaria => usar F1_macro, si no => F1_weighted
    chosen_scoring = "f1_macro" if minority_ratio < 0.30 else "f1_weighted"
    print(f"Métrica elegida: {chosen_scoring} (minority_ratio={minority_ratio:.3f})") # Informamos decisión
    return chosen_scoring

def graficar_arbol(best_pipeline, feature_names):
    """Grafica el árbol de decisión resultante."""
    try:
        plt.figure(figsize=(12, 9))                                  # Definimos tamaño de la figura
        plot_tree(best_pipeline.named_steps["clf"],                  # Obtenemos el clasificador dentro del pipeline
                  filled=True, feature_names=list(feature_names))    # Dibujamos el árbol con colores
        plt.title("Árbol de Decisión (mejor configuración)")          # Agregamos título
        plt.tight_layout()                                           # Ajustamos márgenes
        plt.show()                                                   # Mostramos la gráfica
    except Exception:
        pass                                                         # Si falla (ej. entorno sin display), ignoramos

def graficar_matriz_confusion(y_test, y_pred):
    """Grafica la matriz de confusión del conjunto de test."""
    try:
        fig = plt.figure(figsize=(6, 5))                             # Definimos tamaño
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)      # Creamos matriz de confusión
        plt.title("Matriz de Confusión (Test)")                      # Título
        plt.tight_layout()                                           # Ajustamos márgenes
        plt.show()                                                   # Mostramos
    except Exception:
        pass

def graficar_curva_roc(best_model, X_test, y_test):
    """Grafica curva ROC binaria o multiclase."""
    # Obtenemos puntajes (probabilidades o decision_function)
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)
    elif hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(X_test)
    else:
        y_score = None

    if y_score is None:                                              # Si el modelo no genera scores, salir
        return

    classes = np.unique(y_test)                                      # Identificamos clases únicas
    if len(classes) == 2:                                            # Caso binario
        y_true = (y_test == classes.max()).astype(int)               # Codificamos clase positiva como 1
        prob_pos = y_score[:, 1] if y_score.ndim == 2 else y_score   # Tomamos probabilidad de clase positiva
        fpr, tpr, _ = roc_curve(y_true, prob_pos)                    # Calculamos curva ROC
        roc_auc = auc(fpr, tpr)                                      # Calculamos AUC

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")             # Trazamos curva ROC
        plt.plot([0, 1], [0, 1], linestyle="--")                     # Línea de referencia
        plt.xlabel("FPR")                                            # Eje X
        plt.ylabel("TPR")                                            # Eje Y
        plt.title("Curva ROC (Test)")                                # Título
        plt.legend(loc="lower right")                                # Leyenda
        plt.tight_layout()
        plt.show()
    else:                                                            # Caso multiclase OVR
        Yb = label_binarize(y_test, classes=classes)                 # Binarizamos etiquetas
        if y_score.ndim > 1:                                         # Verificamos que haya varias columnas
            plt.figure(figsize=(6, 5))
            for i in range(Yb.shape[1]):                             # Iteramos sobre cada clase
                fpr, tpr, _ = roc_curve(Yb[:, i], y_score[:, i])     # Calculamos curva ROC de cada clase
                roc_auc = auc(fpr, tpr)                              # Calculamos AUC
                plt.plot(fpr, tpr, label=f"Clase {classes[i]} (AUC={roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("Curvas ROC One-vs-Rest (Test)")
            plt.legend()
            plt.tight_layout()
            plt.show()

# ==== PROGRAMA PRINCIPAL ====
if __name__ == "__main__":
    # 1. Cargar datos
    assert CSV_PATH.exists(), f"No se encontró {CSV_PATH}"           # Validamos existencia del archivo
    X, y = cargar_dataset_heart(CSV_PATH, TARGET_COL)                # Cargamos X e y

    # 2. Dividir train/test (estratificado para conservar proporciones de clase)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Elegir métrica automáticamente
    chosen_scoring = elegir_metricas_por_desbalance(y_train)

    # 4. Definir pipeline y rejilla de hiperparámetros
    pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=42))])  # Pipeline con clasificador
    param_grid = {                                                       # Valores a probar
        "clf__criterion": ["gini", "entropy", "log_loss"],
        "clf__max_depth": [None, 3, 5, 7, 10, 15],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__ccp_alpha": [0.0, 0.001, 0.01, 0.05]
    }

    # 5. GridSearchCV para optimizar modelo
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=chosen_scoring,
        cv=5,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    grid.fit(X_train, y_train)                                         # Entrenamos con validación cruzada

    # 6. Resultados de la búsqueda
    print("Mejores hiperparámetros:", grid.best_params_)               # Configuración óptima
    print("Mejor puntaje (CV):", round(grid.best_score_, 3))           # Puntaje promedio en CV

    # 7. Evaluación final en test
    best_model = grid.best_estimator_                                  # Extraemos mejor modelo entrenado
    y_pred = best_model.predict(X_test)                                # Predicciones en test
    acc = accuracy_score(y_test, y_pred)                               # Accuracy
    prec = precision_score(y_test, y_pred, average="weighted")         # Precisión ponderada
    rec = recall_score(y_test, y_pred, average="weighted")             # Recall ponderado
    f1 = f1_score(y_test, y_pred, average="weighted")                  # F1 ponderado
    print(f"Test -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

    # 8. Graficar resultados
    graficar_arbol(best_model, X_train.columns)
    graficar_matriz_confusion(y_test, y_pred)
    graficar_curva_roc(best_model, X_test, y_test)
