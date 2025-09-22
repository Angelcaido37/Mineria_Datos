# -*- coding: utf-8 -*-
# ==============================================================
# 4a. Reportes automatizados de EDA
# - ydata_profiling (antes pandas_profiling)
# - Sweetviz (comparativo simple si procede)
# Dataset: Datasets/housing_train.csv
# Salidas:
#   - outputs/reports/reporte_ydata_profiling.html
#   - outputs/reports/reporte_sweetviz.html
# ==============================================================

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

import warnings
import os
import pandas as pd
import numpy as np
if not hasattr(np, "VisibleDeprecationWarning"):
    try:
          from numpy._exceptions import VisibleDeprecationWarning
          np.VisibleDeprecationWarning = VisibleDeprecationWarning
    except ImportError:
          np.VisibleDeprecationWarning = DeprecationWarning
          warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
          
             
DATA_PATH = r"Datasets/housing_train.csv"
OUT_REPORTS = r"outputs/reports"

os.makedirs(OUT_REPORTS, exist_ok=True)

# Cargar dataset
df = pd.read_csv(DATA_PATH)

# ---------------- ydata_profiling ----------------
try:
    from ydata_profiling import ProfileReport
    profile = ProfileReport(
        df,
        title="Reporte de Análisis Exploratorio (ydata_profiling)",
        explorative=True,
        minimal=False
    )
    out1 = os.path.join(OUT_REPORTS, "reporte_ydata_profiling.html")
    profile.to_file(out1)
    print("Reporte ydata_profiling generado:", out1)
except Exception as e:
    print("No se pudo generar ydata_profiling. Detalle:", str(e))

# ---------------- Sweetviz ----------------
try:
    import sweetviz as sv
    # Reporte básico (sin división train/test)
    report = sv.analyze(df)
    out2 = os.path.join(OUT_REPORTS, "reporte_sweetviz.html")
    report.show_html(out2)
    print("Reporte Sweetviz generado:", out2)
except Exception as e:
    print("No se pudo generar Sweetviz. Detalle:", str(e))
