
# -*- coding: utf-8 -*-
"""
Script 01: Clasificación con Árboles de Decisión y KNN para predicción de incumplimiento.
Incluye: train-test split, escalado (para KNN), validación cruzada, GridSearchCV,
matrices de confusión, precision/recall/F1, AUC/ROC, KS, curvas de validación,
curvas de aprendizaje y calibración de probabilidades.


"""

# Importamos las librerías necesarias
import pandas as pd                        # Para manejo de datos en DataFrames
import numpy as np                         # Para operaciones numéricas
import matplotlib.pyplot as plt            # Para graficar
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, validation_curve, learning_curve  # Validación y tuning
from sklearn.preprocessing import StandardScaler                           # Escalado de variables
from sklearn.pipeline import Pipeline                                      # Pipeline para encadenar pasos
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve  # Métricas de evaluación
from sklearn.metrics import precision_recall_fscore_support                # Métricas adicionales
from sklearn.metrics import make_scorer                                    # Crear scorers personalizados
from sklearn.tree import DecisionTreeClassifier                            # Modelo de árbol de decisión
from sklearn.neighbors import KNeighborsClassifier                         # Modelo KNN
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # Calibración de probabilidades
from sklearn.metrics import roc_auc_score                                  # AUC
from scipy.stats import ks_2samp                                           # KS estadístico
from sklearn.compose import ColumnTransformer                              # Preprocesamiento por columnas
from sklearn.preprocessing import OneHotEncoder                            # Codificador para categóricas

# Cargamos el dataset desde el CSV
from pathlib import Path
csv_path = Path("Unidad III") / "Ejercicio" / "loan_recovery_dataset.csv"
df = pd.read_csv(csv_path)

# Definimos la variable objetivo: 'default_60d' (1 si incumple en 60 días)
y = df["default_60d"].values

# Seleccionamos variables predictoras (excluimos identificadores y variables leakage)
features_num = [
    "age", "monthly_income", "loan_amount", "interest_rate_annual", "term_months",
    "employment_length_years", "credit_score", "dti_percent", "num_late_payments",
    "days_past_due", "app_logins_30d", "sms_open_rate", "email_open_rate",
    "whatsapp_opt_in", "previous_restructuring", "macro_unemployment", "macro_inflation"
]

features_cat = [
    "employment_status", "region", "last_contact_channel", "last_contact_response", "has_collateral", "action_taken"
]

X = df[features_num + features_cat].copy()

# Definimos preprocesamiento: One-Hot para categóricas y passthrough para numéricas
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", features_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ]
)

# Definimos partición de entrenamiento/prueba estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Configuramos validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --------------------------
# 1) Árbol de Decisión
# --------------------------

# Creamos un pipeline: preprocesamiento + clasificador árbol
pipe_tree = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
])

# Definimos la grilla de hiperparámetros a explorar para el árbol
param_grid_tree = {
    "clf__max_depth": [3, 5, 7, 9],                # Profundidad máxima
    "clf__min_samples_split": [10, 50, 100, 200],          # Mínimo para dividir
    "clf__min_samples_leaf": [5, 20, 50, 100],             # Mínimo en hoja
    "clf__criterion": ["gini", "entropy"],               # Criterio de división
}

# Usamos AUC como métrica principal durante el tuning (apropiado para clases desbalanceadas)
grid_tree = GridSearchCV(
    pipe_tree, param_grid_tree, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
)

# Ajustamos el GridSearchCV con datos de entrenamiento
grid_tree.fit(X_train, y_train)

# Obtenemos el mejor estimador encontrado
best_tree = grid_tree.best_estimator_

# Predecimos probabilidades para AUC y clases para matriz de confusión
y_prob_tree = best_tree.predict_proba(X_test)[:, 1]

from sklearn.metrics import precision_recall_curve, fbeta_score

# Curva Precisión–Recall y búsqueda de umbral
prec, rec, thr = precision_recall_curve(y_test, y_prob_tree)

# Opción A: máximo F2 (recall-friendly)
best_idx = np.argmax([fbeta_score(y_test, (y_prob_tree>=t).astype(int), beta=2) for t in np.r_[thr, 1.0]])
best_thr = np.r_[thr, 1.0][best_idx]

# Opción B (alternativa): primer umbral con Recall >= 0.70 (ajusta a tu gusto)
# try:
#     best_thr = np.r_[thr, 1.0][np.where(rec[:-1] >= 0.70)[0][0]]
# except IndexError:
#     best_thr = 0.5  # fallback

print(f"Umbral óptimo (F2): {best_thr:.3f}")

# Predicción con umbral óptimo
y_pred_tree_opt = (y_prob_tree >= best_thr).astype(int)

# Métricas con umbral óptimo
cm_opt = confusion_matrix(y_test, y_pred_tree_opt)
p_opt, r_opt, f1_opt, _ = precision_recall_fscore_support(y_test, y_pred_tree_opt, average="binary", zero_division=0)
print("Matriz de confusión (umbral óptimo):\n", cm_opt)
print(f"Precision: {p_opt:.3f}  Recall: {r_opt:.3f}  F1: {f1_opt:.3f}")

y_pred_tree = (y_prob_tree >= 0.5).astype(int)

# Métricas de evaluación del árbol
auc_tree = roc_auc_score(y_test, y_prob_tree)
cm_tree = confusion_matrix(y_test, y_pred_tree)
precision_tree, recall_tree, f1_tree, _ = precision_recall_fscore_support(y_test, y_pred_tree, average="binary")

# Calculamos estadístico KS (separación entre distribuciones de score)
ks_tree = ks_2samp(y_prob_tree[y_test==1], y_prob_tree[y_test==0]).statistic

# Imprimimos resultados del árbol
print("=== Árbol de Decisión ===")
print("Mejores hiperparámetros:", grid_tree.best_params_)
#AUC = número que mide la capacidad discriminativa global del modelo (cuanto más cercano a 1, mejor).
print(f"AUC: {auc_tree:.4f}  |  KS: {ks_tree:.4f}") 
print("Matriz de confusión:\n", cm_tree)
print(f"Precision: {precision_tree:.4f}  Recall: {recall_tree:.4f}  F1: {f1_tree:.4f}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_tree))

# Curva ROC del árbol
# ROC = curva que muestra el desempeño del modelo para todos los umbrales de decisión.
fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree) 
plt.figure()
plt.plot(fpr_tree, tpr_tree, label=f"Árbol (AUC={auc_tree:.3f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR") # False Positive Rate (FPR) → % de clientes buenos clasificados como malos.
plt.ylabel("TPR") #True Positive Rate (TPR) → % de morosos detectados correctamente.
plt.title("ROC - Árbol de Decisión")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 2) KNN (con escalado)
# --------------------------

# Para KNN conviene escalar; hacemos pipeline separado para usar StandardScaler
pipe_knn = Pipeline(steps=[
    ("prep", preprocess),
    ("scaler", StandardScaler(with_mean=False)),  # with_mean=False para matrices dispersas
    ("clf", KNeighborsClassifier())
])

# Definimos grilla de hiperparámetros para KNN
param_grid_knn = {
    "clf__n_neighbors": [5, 15, 25, 35],
    "clf__weights": ["uniform", "distance"],
    "clf__p": [1, 2],  # Manhattan (1) o Euclidiana (2)
}

# GridSearchCV para KNN usando AUC
grid_knn = GridSearchCV(
    pipe_knn, param_grid_knn, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0
)

# Ajustamos el GridSearchCV
grid_knn.fit(X_train, y_train)

# Mejor estimador
best_knn = grid_knn.best_estimator_

# Probabilidades (usamos predict_proba; para KNN puede ser promedio de vecinos)
y_prob_knn = best_knn.predict_proba(X_test)[:, 1]
y_pred_knn = (y_prob_knn >= 0.5).astype(int)

# Métricas de evaluación
auc_knn = roc_auc_score(y_test, y_prob_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
precision_knn, recall_knn, f1_knn, _ = precision_recall_fscore_support(y_test, y_pred_knn, average="binary")
ks_knn = ks_2samp(y_prob_knn[y_test==1], y_prob_knn[y_test==0]).statistic

# Resultados KNN
print("=== KNN ===")
print("Mejores hiperparámetros:", grid_knn.best_params_)
print(f"AUC: {auc_knn:.4f}  |  KS: {ks_knn:.4f}")
print("Matriz de confusión:\n", cm_knn)
print(f"Precision: {precision_knn:.4f}  Recall: {recall_knn:.4f}  F1: {f1_knn:.4f}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred_knn))

# Curva ROC de KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
plt.figure()
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc_knn:.3f})")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC - KNN")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 3) Curvas de validación
# --------------------------

# Curva de validación para max_depth del Árbol
param_range = [3, 5, 7, 9, None]
train_scores, test_scores = validation_curve(
    pipe_tree, X_train, y_train, param_name="clf__max_depth",
    param_range=param_range, cv=cv, scoring="roc_auc", n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure()
plt.plot([str(p) for p in param_range], train_mean, marker="o", label="Train AUC")
plt.plot([str(p) for p in param_range], test_mean, marker="o", label="CV AUC")
plt.xlabel("max_depth")
plt.ylabel("AUC")
plt.title("Curva de Validación - Árbol (max_depth)")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 4) Curvas de aprendizaje
# --------------------------

train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    best_tree, X, y, cv=cv, scoring="roc_auc", n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=42
)
plt.figure()
plt.plot(train_sizes, train_scores_lc.mean(axis=1), marker="o", label="Train AUC")
plt.plot(train_sizes, test_scores_lc.mean(axis=1), marker="o", label="CV AUC")
plt.xlabel("Tamaño de entrenamiento")
plt.ylabel("AUC")
plt.title("Curva de Aprendizaje - Mejor Árbol")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# 5) Calibración de probabilidades
# --------------------------

# Calibramos el mejor árbol con isotonic (útil en scoring de crédito)
calibrated = CalibratedClassifierCV(best_tree, method="isotonic", cv=3)
calibrated.fit(X_train, y_train)
y_prob_cal = calibrated.predict_proba(X_test)[:, 1]

# Curva de calibración
prob_true, prob_pred = calibration_curve(y_test, y_prob_cal, n_bins=10, strategy="quantile")
plt.figure()
plt.plot(prob_pred, prob_true, marker="o", label="Calibrado (isotonic)")
plt.plot([0,1],[0,1],"--", label="Perfectamente calibrado")
plt.xlabel("Probabilidad predicha")
plt.ylabel("Frecuencia observada")
plt.title("Curva de Calibración")
plt.legend()
plt.tight_layout()
plt.show()

from sklearn.metrics import average_precision_score
ap = average_precision_score(y_test, y_prob_tree)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Curva Precision-Recall (AP={ap:.3f})")
plt.tight_layout()
plt.show()

