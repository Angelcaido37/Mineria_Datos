# -*- coding: utf-8 -*-
"""
04_random_forest_auto.py (versión académica y comentada)

Este script implementa el algoritmo Random Forest (Bosques Aleatorios) con:
- Carga del dataset /Datasets/heart.csv
- División en entrenamiento y prueba
- Selección automática de métrica (f1_macro o f1_weighted según desbalance)
- Búsqueda de hiperparámetros con RandomizedSearchCV
- Evaluación en conjunto de prueba
- Gráficas: Importancia de variables (Top-10), Matriz de Confusión y Curva ROC
"""

# ==== IMPORTACIONES BÁSICAS ====
import numpy as np                                    # Librería numérica
import pandas as pd                                   # Manejo de DataFrames
import matplotlib.pyplot as plt                       # Gráficas
from pathlib import Path                              # Manejo de rutas

# ==== MODELOS Y HERRAMIENTAS DE SKLEARN ====
from collections import Counter                       # Conteo de clases
from sklearn.ensemble import RandomForestClassifier   # Modelo Random Forest
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Split y optimización aleatoria
from sklearn.metrics import (                         # Métricas de evaluación
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize      # Para multiclase en curvas ROC
from scipy.stats import randint                       # Para distribuciones de hiperparámetros

# ==== PARÁMETROS DEL DATASET ====
CSV_PATH = Path("Datasets/heart.csv")                # Ruta del dataset Heart
TARGET_COL = "target"                                 # Columna objetivo (0/1)


# ==== PROGRAMA PRINCIPAL ====
if __name__ == "__main__":
    # 1. Cargar dataset
    assert CSV_PATH.exists(), f"No se encontró {CSV_PATH}"   # Validar existencia del archivo
    df = pd.read_csv(CSV_PATH)                               # Leer dataset
    assert TARGET_COL in df.columns, f"Falta columna {TARGET_COL}"  # Validar que exista target
    
    X = df.drop(columns=[TARGET_COL])                        # Variables independientes
    y = df[TARGET_COL]                                       # Variable objetivo

    # 2. Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Detectar desbalance de clases y elegir métrica automáticamente
    class_counts = Counter(y_train)                          # Conteo de clases
    total = sum(class_counts.values())                       # Total de ejemplos
    minority_ratio = min(class_counts.values()) / total      # Proporción clase minoritaria
    chosen_scoring = "f1_macro" if minority_ratio < 0.30 else "f1_weighted"
    print(f"Métrica elegida automáticamente para CV: {chosen_scoring} "
          f"(minority_ratio={minority_ratio:.3f}, counts={dict(class_counts)})")

    # 4. Definir modelo base Random Forest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)  # Modelo con todos los núcleos disponibles

    # 5. Definir distribuciones de hiperparámetros para RandomizedSearchCV
    param_dist = {
        "n_estimators": randint(100, 600),                   # Número de árboles
        "max_depth": [None] + list(range(3, 21)),            # Profundidad máxima
        "min_samples_split": randint(2, 20),                 # Mínimo de muestras para dividir un nodo
        "min_samples_leaf": randint(1, 10),                  # Mínimo de muestras en hoja
        "max_features": ["sqrt", "log2", None] + list(np.linspace(0.2, 1.0, 5)), # Subconjunto de variables
        "bootstrap": [True, False]                           # Uso de muestreo bootstrap
    }

    # 6. Optimización con RandomizedSearchCV (más eficiente que GridSearch en RF)
    rnd = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,                      # Espacio de búsqueda
        n_iter=60,                                           # Número de iteraciones (60 configuraciones aleatorias)
        scoring=chosen_scoring,                              # Métrica seleccionada automáticamente
        cv=5,                                                # Validación cruzada con 5 folds
        n_jobs=-1,                                           # Uso de todos los núcleos disponibles
        random_state=42,                                     # Reproducibilidad
        verbose=1,                                           # Mostrar progreso
        refit=True                                           # Reentrenar con la mejor config
    )
    rnd.fit(X_train, y_train)                                # Entrenamiento con búsqueda aleatoria

    # 7. Resultados de la búsqueda
    print("Mejores hiperparámetros:", rnd.best_params_)      # Configuración óptima
    print("Mejor puntaje (CV):", round(rnd.best_score_, 3))  # Puntaje medio en CV

    # 8. Evaluación en conjunto de prueba
    best_model = rnd.best_estimator_                         # Mejor modelo entrenado
    y_pred = best_model.predict(X_test)                      # Predicciones en test

    # Calcular métricas principales
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Test -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

    # 9. Gráfica: Importancia de variables (Top-10)
    importances = best_model.feature_importances_            # Importancia de cada variable
    feat_names = X_train.columns.tolist()                    # Nombres de variables
    order = np.argsort(importances)[::-1][:10]               # Seleccionar top-10
    plt.figure(figsize=(7, 5))
    plt.bar(range(len(order)), importances[order])           # Gráfico de barras
    plt.xticks(range(len(order)), [feat_names[i] for i in order], rotation=45, ha="right")
    plt.title("Importancia de variables (Top 10)")
    plt.tight_layout()
    plt.show()

    # 10. Gráfica: Matriz de confusión
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.ax_.set_title("Matriz de Confusión (Test)")
    plt.tight_layout()
    plt.show()

    # 11. Gráfica: Curva ROC
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)           # Si soporta predict_proba
    elif hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(X_test)       # Alternativa
    else:
        y_score = None

    if y_score is not None:
        classes = np.unique(y_test)                          # Extraer clases
        if len(classes) == 2:                                # Caso binario
            y_true = (y_test == classes.max()).astype(int)   # Codificar clase positiva
            prob_pos = y_score[:, 1] if y_score.ndim == 2 else y_score
            fpr, tpr, _ = roc_curve(y_true, prob_pos)        # Calcular ROC
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
            Yb = label_binarize(y_test, classes=classes)
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
