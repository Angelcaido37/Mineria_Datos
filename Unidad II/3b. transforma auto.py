# ==============================================================
# TEMA 3: TRANSFORMACIÓN DE VARIABLES (automática)
# Dataset: House Prices (Kaggle)
# ==============================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

# 1. Cargar dataset
df = pd.read_csv(r"Datasets/housing_train.csv")

# 2. Separar columnas numéricas y categóricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("=== Variables numéricas ===")
print(numeric_cols)
print("\n=== Variables categóricas ===")
print(categorical_cols)

# 3. Diccionario para guardar transformaciones
transformaciones = {}

# 4. Transformar variables numéricas según skewness
for col in numeric_cols:
    skew_val = df[col].skew()  # asimetría
    if abs(skew_val) > 1:  
        # Muy sesgada → aplicar log-transform
        df[col + "_log"] = np.log1p(df[col])
        transformaciones[col] = "log-transform"
    elif abs(skew_val) > 0.5:  
        # Moderadamente sesgada → escalar con MinMax
        scaler = MinMaxScaler()
        df[col + "_minmax"] = scaler.fit_transform(df[[col]])
        transformaciones[col] = "MinMaxScaler"
    else:
        # Aproximadamente normal → estandarizar
        scaler = StandardScaler()
        df[col + "_std"] = scaler.fit_transform(df[[col]])
        transformaciones[col] = "StandardScaler"

# 5. Transformar variables categóricas automáticamente
for col in categorical_cols:
    # Usamos LabelEncoder (rápido y simple)
    le = LabelEncoder()
    df[col + "_label"] = le.fit_transform(df[col].astype(str))
    transformaciones[col] = "LabelEncoder"

# 6. Mostrar resumen de transformaciones aplicadas
print("\n=== Transformaciones aplicadas ===")
for var, metodo in transformaciones.items():
    print(f"{var}: {metodo}")

# 7. Guardar dataset transformado
df.to_csv("train_transformed.csv", index=False)
print("\n Dataset transformado guardado como 'train_transformed.csv'")
