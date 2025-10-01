# -*- coding: utf-8 -*-
"""
03_svm_auto.py (versión académica con comentarios detallados)

Este script implementa el algoritmo SVM (Support Vector Machine) con:
- Carga del dataset /Datasets/heart.csv
- División en entrenamiento y prueba
- Selección automática de métrica (según desbalance de clases)
- Pipeline con estandarización + SVM
- Búsqueda de hiperparámetros (GridSearchCV)
- Evaluación en conjunto de prueba
- Gráficas: Matriz de Confusión y Curva ROC
"""

# ==== IMPORTACIONES BÁSICAS ====
import numpy as np                                    # Librería numérica
import pandas as pd                                   # Librería para manejo de datos
from pathlib import Path                              # Manejo seguro de rutas
import matplotlib.pyplot as plt                       # Para graficar resultados

# ==== MODELOS Y UTILIDADES DE SKLEARN ====
from collections import Counter                       # Para medir balance de clases
from sklearn.model_selection import train_test_split, GridSearchCV  # Split y optimización
from sklearn.pipeline import Pipeline                 # Para encadenar preprocesamiento y modelo
from sklearn.preprocessing import StandardScaler, label_binarize    # Escalado + binarización
from sklearn.svm import SVC                           # Clasificador SVM
from sklearn.metrics import (                         # Métricas de evaluación
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc
)

# ==== PARÁMETROS DEL DATASET ====
CSV_PATH = Path("Datasets/heart.csv")                # Ruta al dataset solicitada
TARGET_COL = "target"                                 # Nombre de la variable objetivo


# ==== PROGRAMA PRINCIPAL ====
if __name__ == "__main__":
    # 1. Cargar dataset
    assert CSV_PATH.exists(), f"No se encontró {CSV_PATH}"   # Validar existencia del archivo
    df = pd.read_csv(CSV_PATH)                               # Leer CSV en un DataFrame
    assert TARGET_COL in df.columns, f"Falta columna {TARGET_COL}"  # Verificar que esté la variable objetivo

    X = df.drop(columns=[TARGET_COL])                        # Variables independientes
    y = df[TARGET_COL]                                       # Variable objetivo

    # 2. Dividir en train/test con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42    # 20% test, manteniendo balance de clases
    )

    # 3. Detectar desbalance y elegir métrica automáticamente
    class_counts = Counter(y_train)                          # Conteo de clases
    total = sum(class_counts.values())                       # Total de observaciones
    minority_ratio = min(class_counts.values()) / total      # Proporción clase minoritaria
    chosen_scoring = "f1_macro" if minority_ratio < 0.30 else "f1_weighted"
    print(f"Métrica elegida automáticamente para CV: {chosen_scoring} "
          f"(minority_ratio={minority_ratio:.3f}, counts={dict(class_counts)})")

    # 4. Definir pipeline: normalización + clasificador SVM
    pipe = Pipeline([
        ("scaler", StandardScaler()),                        # Paso 1: escalar variables
        ("svm", SVC(probability=True, random_state=42))      # Paso 2: clasificador SVM
    ])

    # 5. Definir rejilla de hiperparámetros para GridSearchCV
    param_grid = {
        "svm__kernel": ["linear", "rbf", "poly"],            # Tipos de kernel
        "svm__C": [0.1, 1, 10, 100],                         # Parámetro de regularización
        "svm__gamma": ["scale", "auto"],                     # Parámetro gamma (para rbf/poly)
        "svm__degree": [2, 3, 4]                             # Grado del polinomio (si kernel=poly)
    }

    # 6. Optimización de hiperparámetros
    grid = GridSearchCV(
        estimator=pipe,                                      # Pipeline
        param_grid=param_grid,                               # Rejilla de hiperparámetros
        scoring=chosen_scoring,                              # Métrica elegida automáticamente
        cv=5,                                                # Validación cruzada con 5 folds
        n_jobs=-1,                                           # Usar todos los núcleos disponibles
        verbose=1,                                           # Mostrar progreso en consola
        refit=True                                           # Reentrenar con el mejor modelo
    )
    grid.fit(X_train, y_train)                               # Entrenar con GridSearchCV

    # 7. Mostrar mejores resultados
    print("Mejores hiperparámetros:", grid.best_params_)     # Configuración óptima
    print("Mejor puntaje (CV):", round(grid.best_score_, 3)) # Puntaje promedio en validación cruzada

    # 8. Evaluación en test
    best_model = grid.best_estimator_                        # Mejor modelo encontrado
    y_pred = best_model.predict(X_test)                      # Predicciones en test

    acc = accuracy_score(y_test, y_pred)                     # Exactitud
    prec = precision_score(y_test, y_pred, average="weighted") # Precisión
    rec = recall_score(y_test, y_pred, average="weighted")   # Exhaustividad
    f1 = f1_score(y_test, y_pred, average="weighted")        # F1-score
    print(f"Test -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

    # 9. Matriz de confusión
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)  # Crear matriz de confusión
    disp.ax_.set_title("Matriz de Confusión (Test)")                # Título
    plt.tight_layout()
    plt.show()

    # 10. Curva ROC (binaria o multiclase)
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)           # Si tiene método predict_proba
    elif hasattr(best_model, "decision_function"):
        y_score = best_model.decision_function(X_test)       # O decision_function
    else:
        y_score = None                                       # Si no hay, no se puede graficar

    if y_score is not None:
        classes = np.unique(y_test)                          # Clases presentes
        if len(classes) == 2:                                # Caso binario
            y_true = (y_test == classes.max()).astype(int)   # Codificar clase positiva
            prob_pos = y_score[:, 1] if y_score.ndim == 2 else y_score
            fpr, tpr, _ = roc_curve(y_true, prob_pos)        # Curva ROC
            roc_auc = auc(fpr, tpr)                          # AUC
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title("Curva ROC (Test)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        else:                                                # Caso multiclase OVR
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
