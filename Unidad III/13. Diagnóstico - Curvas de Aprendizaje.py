# -*- coding: utf-8 -*-
"""
Tema: 6.3 Diagnóstico con Curvas de Aprendizaje
Dataset: Breast Cancer
Explicación: Las curvas de aprendizaje nos ayudan a responder una pregunta
diferente: ¿Nos beneficiaremos de conseguir más datos?
Estas curvas muestran el rendimiento del modelo (en entrenamiento y validación)
a medida que aumenta el número de muestras de entrenamiento utilizadas.
Nos ayudan a diagnosticar si un modelo sufre de alto sesgo (underfitting) o
alta varianza (overfitting) y si agregar más datos podría ayudar.
"""
# 1. Importar librerías
# ---------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 2. Cargar datos
# ---------------
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 3. Función para graficar la curva de aprendizaje
# ------------------------------------------------
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Genera un gráfico de la curva de aprendizaje para un estimador dado.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Número de muestras de entrenamiento")
    plt.ylabel("Accuracy Score")
    plt.grid(True)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score de Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de Validación Cruzada")

    plt.legend(loc="best")
    return plt

# --- CASO 1: MODELO CON ALTA VARIANZA (OVERFITTING) ---
# Un RandomForest muy profundo (sin límite de profundidad) tiende a sobreajustar.
print("--- CASO 1: Diagnóstico de un modelo con Alta Varianza (Overfitting) ---")
estimator_overfitting = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

plot_learning_curve(estimator_overfitting, "Curva de Aprendizaje (Potencial Overfitting)", X, y, cv=5, n_jobs=-1)
plt.show()

print("Interpretación (Alta Varianza):")
print("1. El score de entrenamiento es muy alto (cercano a 1.0) y no baja mucho.")
print("2. El score de validación es considerablemente más bajo que el de entrenamiento.")
print("3. Hay una gran brecha (gap) entre la curva de entrenamiento y la de validación.")
print("4. La curva de validación sigue subiendo a medida que se agregan más datos. Esto es una buena señal.")
print("   => CONCLUSIÓN: El modelo se beneficiaría de MÁS DATOS DE ENTRENAMIENTO. También se puede")
print("      intentar reducir la complejidad del modelo (ej. limitar `max_depth`) o aumentar la regularización.")


# --- CASO 2: MODELO CON ALTO SESGO (UNDERFITTING) ---
# Un RandomForest muy simple (con max_depth=1) será demasiado simple.
print("\n--- CASO 2: Diagnóstico de un modelo con Alto Sesgo (Underfitting) ---")
estimator_underfitting = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42))
])

plot_learning_curve(estimator_underfitting, "Curva de Aprendizaje (Potencial Underfitting)", X, y, cv=5, n_jobs=-1)
plt.show()

print("Interpretación (Alto Sesgo):")
print("1. Tanto el score de entrenamiento como el de validación son bajos y convergen.")
print("2. Las dos curvas están muy juntas (la brecha es pequeña).")
print("3. Las curvas se aplanan rápidamente; agregar más datos no mejora significativamente el rendimiento.")
print("   => CONCLUSIÓN: Agregar más datos NO ayudará. El problema es el modelo en sí.")
print("      Se necesita un MODELO MÁS COMPLEJO (ej. aumentar `max_depth`, probar un algoritmo diferente,")
print("      o añadir más características informativas - feature engineering).")
