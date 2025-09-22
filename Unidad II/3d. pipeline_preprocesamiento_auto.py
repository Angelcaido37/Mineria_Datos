# -*- coding: utf-8 -*-
# ==============================================================
# TEMA 3: PIPELINE DE PREPROCESAMIENTO (automático y robusto)
# Dataset: House Prices (Kaggle)
# ==============================================================
# Este script construye un ColumnTransformer con pasos típicos de EDA/ML:
# - Numéricas: imputación por mediana, Yeo-Johnson (acepta <=0) y RobustScaler
# - Categóricas: imputación por "más frecuente" y OneHotEncoder (manejo auto)
# Guarda el pipeline ajustado y devuelve X_preprocesado listo para modelar.
# (Metodología inspirada en el PDF Unid. II: normalización/escalado y pipeline)
# ==============================================================

import pandas as pd  # manipulación de datos
import numpy as np   # utilidades numéricas
from sklearn.model_selection import train_test_split  # dividir train/test
from sklearn.compose import ColumnTransformer  # aplicar transformaciones por tipo
from sklearn.pipeline import Pipeline  # encadenar pasos
from sklearn.impute import SimpleImputer  # imputación simple
from sklearn.preprocessing import OneHotEncoder  # codificación categórica
from sklearn.preprocessing import PowerTransformer  # Yeo-Johnson
from sklearn.preprocessing import RobustScaler  # escalado robusto (menos sensible a outliers)
import joblib  # para guardar el pipeline a disco

# Cargar el dataset 
df = pd.read_csv(r"Datasets/housing_train.csv")  # lee datos

# Definir la variable objetivo (en Kaggle House Prices suele ser SalePrice)
target = "SalePrice"  # nombre de la y

# Separar X (features) e y (target)
X = df.drop(columns=[target], errors="ignore")  # todo menos la y
y = df[target] if target in df.columns else None  # y si existe

# Identificar tipos de columnas
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()  # numéricas
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()   # categóricas

# (Opcional) dividir en entrenamiento y prueba para ilustrar uso típico
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)  # 80/20 split reproducible

# Definir transformador para columnas numéricas
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),     # mediana es robusta a outliers
    ("power", PowerTransformer(method="yeo-johnson")), # corrige asimetría (acepta ceros/negativos)
    ("scaler", RobustScaler())                         # escalado robusto (usa mediana/IQR)
])  # ver PDF: normalización/escalamiento y transformaciones avanzadas

# Definir transformador para columnas categóricas
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),           # rellena con la moda
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # dummies seguros
])  # según PDF: One-Hot para nominales; 'ignore' evita errores con categorías nuevas

# Crear el ColumnTransformer que aplica por tipo
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),  # aplica pipeline numérico a num_cols
        ("cat", categorical_transformer, cat_cols)  # aplica pipeline categórico a cat_cols
    ],
    remainder="drop"  # descarta columnas no listadas (puedes cambiar a 'passthrough')
)  # ver PDF: Pipeline de Transformación Completo (ColumnTransformer)

# Ajustar el preprocesador a los datos de entrenamiento
preprocessor.fit(X_train)  # aprende estadísticas (medianas, lambdas YJ, categorías, etc.)

# Transformar train y test
X_train_prep = preprocessor.transform(X_train)  # aplica transformaciones a train
X_test_prep = preprocessor.transform(X_test)    # aplica transformaciones a test

# Obtener nombres de columnas resultantes (útil para revisar salidas)
# Para numéricas, los nombres quedan como estaban; para OneHot, usamos get_feature_names_out
num_feature_names = num_cols  # nombres originales numéricos
cat_feature_names = []
if len(cat_cols) > 0:
    cat_feature_names = preprocessor.named_transformers_["cat"] \
        .named_steps["onehot"] \
        .get_feature_names_out(cat_cols) \
        .tolist()  # genera nombres tipo "Columna_Categoria"

# Concatenar nombres finales en el mismo orden que ColumnTransformer
final_feature_names = num_feature_names + cat_feature_names  # orden: num luego cat

# Convertir matrices numpy a DataFrames con nombres para fácil inspección
X_train_prep_df = pd.DataFrame(X_train_prep, columns=final_feature_names, index=X_train.index)
X_test_prep_df = pd.DataFrame(X_test_prep, columns=final_feature_names, index=X_test.index)

# Mostrar un resumen rápido en consola
print("=== Preprocesamiento completado ===")
print(f"Columnas numéricas: {len(num_cols)} | Columnas categóricas: {len(cat_cols)}")
print(f"Shape X_train -> antes: {X_train.shape} | después: {X_train_prep_df.shape}")
print(f"Shape X_test  -> antes: {X_test.shape}  | después: {X_test_prep_df.shape}")
print("\nPrimeras columnas transformadas:")
print(X_train_prep_df.iloc[:5, :10])  # muestra un vistazo de las primeras 10 columnas

# Guardar el pipeline para reuso en modelado / inferencia
joblib.dump(preprocessor, "preprocessor_pipeline.joblib")  # persiste el objeto a disco
print("\nPipeline guardado como 'preprocessor_pipeline.joblib'")  # confirmación

# (Opcional) Guardar las matrices preprocesadas a CSV para auditoría o pruebas rápidas
X_train_prep_df.to_csv("X_train_preprocesado.csv", index=False)  # exporta train
X_test_prep_df.to_csv("X_test_preprocesado.csv", index=False)    # exporta test
print("Archivos 'X_train_preprocesado.csv' y 'X_test_preprocesado.csv' creados.")
