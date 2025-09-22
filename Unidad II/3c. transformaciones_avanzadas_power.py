# -*- coding: utf-8 -*-
# ==============================================================
# TEMA 3: TRANSFORMACIÓN DE VARIABLES (avanzadas y automáticas)
# Dataset: House Prices (Kaggle)
# (Power transforms, log/raíz y criterios de normalización).
# ==============================================================
# Referencias de clase: ver Unidad-II (Transformación de Variables) y Pipeline
# (Box-Cox / Yeo-Johnson / escalado) para el razonamiento metodológico.
# ==============================================================

# Importamos librerías necesarias
import pandas as pd  # para manejo de datos en tablas
import numpy as np   # para operaciones numéricas
from scipy import stats  # para boxcox y medidas estadísticas
from sklearn.preprocessing import PowerTransformer  # para Yeo-Johnson
import warnings  # para silenciar algunos warnings opcionales

# Opcional: silenciar warnings de SciPy cuando hay valores no válidos
warnings.filterwarnings("ignore")

# Cargamos el dataset 
df = pd.read_csv(r"Datasets/housing_train.csv")  # lee el CSV de entrenamiento

# Identificamos columnas numéricas (candidatas a transformar)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # nombres numéricos

# Diccionarios para registrar resultados y decisiones
best_transform = {}           # método elegido por columna
original_skew = {}            # asimetría original
transformed_skew = {}         # asimetría tras mejor método
lambda_params = {}            # lambda de Box-Cox / Yeo-Johnson si aplica

# Función auxiliar: calcula asimetría ignorando NaNs
def skew_safe(series):
    # calcula skew sin crashear si hay NaNs
    return series.dropna().skew()

# Recorremos cada variable numérica para decidir la mejor transformación
for col in numeric_cols:
    # guardamos la serie original (copia para no modificar df todavía)
    s = df[col].astype(float)  # nos aseguramos tipo float
    # calculamos asimetría original
    sk0 = skew_safe(s)  # skew original
    original_skew[col] = sk0  # almacenamos el valor

    # preparamos un "tablero" de candidatos a probar
    candidates = {}  # dict de nombre_metodo -> (serie_transformada, info_extra)

    # 1) LOG(1+x): requiere valores >= -1; si hay valores <= -1 no se puede
    if (s.min(skipna=True) > -1):  # condición para log1p válida
        s_log = np.log1p(s)  # aplica log(1+x)
        candidates["log1p"] = (s_log, {"note": "log(1+x)"})  # guardamos candidato

    # 2) Raíz cuadrada: requiere valores >= 0; si hay negativos, ajustamos con shift
    if (s.min(skipna=True) >= 0):
        s_sqrt = np.sqrt(s)  # raíz directa
        candidates["sqrt"] = (s_sqrt, {"note": "sqrt(x)"})
    else:
        # desplazamos para evitar negativos: x' = x - min + 1
        shift = -s.min(skipna=True) + 1.0  # desplazamiento mínimo
        s_sqrt_shift = np.sqrt(s + shift)  # raíz tras shift
        candidates["sqrt_shift"] = (s_sqrt_shift, {"note": f"sqrt(x + {shift:.3f})", "shift": shift})

    # 3) Box-Cox: SOLO se puede si todos los valores > 0
    if (s.min(skipna=True) > 0):
        # aplicamos boxcox sobre los valores no nulos y luego reconstruimos el vector
        nonnull = s.dropna().values  # valores sin NaN
        bc_vals, bc_lambda = stats.boxcox(nonnull)  # transform y lambda
        # recomponemos serie con índices originales
        s_bc = pd.Series(index=s.index, dtype=float)  # serie vacía
        s_bc.loc[s.notna()] = bc_vals  # insertamos transformados
        candidates["boxcox"] = (s_bc, {"lambda": bc_lambda, "note": "Box-Cox"})  # guardamos lambda

    # 4) Yeo-Johnson: acepta ceros y negativos (PowerTransformer de sklearn)
    #    Es muy útil cuando hay valores <= 0 (PDF: transformaciones avanzadas).
    pt = PowerTransformer(method="yeo-johnson", standardize=False)  # sin estandarizar para evaluar solo forma
    # ajustamos en valores no nulos para evitar problemas
    s_nonnull = s.dropna().values.reshape(-1, 1)  # columna 2D
    try:
        s_yj_nonnull = pt.fit_transform(s_nonnull).ravel()  # transformamos Y-J
        s_yj = pd.Series(index=s.index, dtype=float)  # serie alineada a índice original
        s_yj.loc[s.notna()] = s_yj_nonnull  # insertamos donde no hay NaN
        # La "lambda" equivalente en sklearn se extrae de pt.lambdas_
        yj_lambda = float(pt.lambdas_[0])  # único valor porque es una sola variable
        candidates["yeo_johnson"] = (s_yj, {"lambda": yj_lambda, "note": "Yeo-Johnson"})
    except Exception as e:
        # si falla por alguna razón numérica, lo ignoramos
        pass

    # Si no hay candidatos (por ejemplo, si la columna es constante), saltamos
    if len(candidates) == 0:
        best_transform[col] = "none"  # no se pudo proponer nada
        transformed_skew[col] = sk0   # dejamos skew igual
        continue  # siguiente columna

    # Evaluamos cada candidato por asimetría absoluta (queremos minimizar |skew|)
    best_name = None  # nombre del mejor método
    best_series = None  # serie transformada del mejor
    best_abs_skew = np.inf  # mejor asimetría absoluta
    best_info = {}  # info del mejor (lambda/shift)

    for name, (s_try, info) in candidates.items():
        sk_try = skew_safe(s_try)  # asimetría del candidato
        if abs(sk_try) < best_abs_skew:  # si mejora (más cercana a 0)
            best_abs_skew = abs(sk_try)  # actualizamos métrica
            best_name = name             # guardamos nombre
            best_series = s_try          # guardamos serie
            best_info = info             # guardamos info extra

    # Registramos la decisión final y aplicamos al DataFrame con un nuevo sufijo
    df[f"{col}__{best_name}"] = best_series  # creamos nueva columna transformada
    best_transform[col] = best_name          # anotamos método elegido
    transformed_skew[col] = skew_safe(best_series)  # guardamos skew final

    # Guardamos lambdas si corresponde (Box-Cox o Yeo-Johnson)
    if "lambda" in best_info:
        lambda_params[col] = best_info["lambda"]  # almacena lambda para consulta futura

# Mostramos un resumen claro de decisiones
print("=== RESUMEN: Mejor transformación por variable ===")
for col in numeric_cols:
    print(f"{col:25s} | skew original = {original_skew.get(col, np.nan):6.3f} "
          f"→ método = {best_transform.get(col,'none'):12s} "
          f"→ skew transformado = {transformed_skew.get(col, np.nan):6.3f}")

# Si deseas guardar el dataset con columnas nuevas:
df.to_csv("train_transformed_power.csv", index=False)  # guarda resultado
print("\nArchivo guardado: 'train_transformed_power.csv'")  # confirmación en consola
