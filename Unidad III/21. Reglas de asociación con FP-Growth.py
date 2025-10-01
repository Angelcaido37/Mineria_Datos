# -*- coding: utf-8 -*-
"""
FP-Growth (tabular) para dataset:
BillNo;Itemname;Quantity;Date;Price;CustomerID;Country

- Lee CSV tabular con ';'
- Agrupa por BillNo y usa Itemname como producto
- Filtra Quantity > 0
- One-hot disperso (sparse) para ahorrar RAM
- Genera reglas y guarda CSV
"""

# --- Significado de las métricas en las reglas ---
# support   = cuántas veces (proporción de tickets) aparecen juntos antecedente y consecuente
# confidence = de todos los tickets con el antecedente, qué porcentaje también tiene el consecuente
# lift       = cuántas veces más probable es que aparezcan juntos que por azar (si >1 hay relación positiva)


import os
import math
import gc
from collections import Counter
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ---------- Config ----------
DATA_PATH     = os.path.join("Datasets", "Market_Basket_Optimisation.csv")
OUTDIR        = "outputs"
ENCODINGS     = ("utf-8", "latin-1")
MIN_SUPPORT   = 0.02
RULE_METRIC   = "lift"
MIN_THRESHOLD = 1.0
MAX_LEN       = 3  # None para sin límite

# Forzar modo tabular con tus columnas
TICKET_COL = "BillNo"
ITEM_COL   = "Itemname"
QTY_COL    = "Quantity"

os.makedirs(OUTDIR, exist_ok=True)

def read_tabular_strict(path):
    """
    Lee el CSV asumiendo separador ';'.
    Ignoramos Price aunque tenga '2\\t55'. Si quisieras, se puede limpiar,
    pero no es necesario para FP-Growth.
    """
    last_err = None
    for enc in ENCODINGS:
        try:
            # engine='python' es más tolerante con rarezas
            df = pd.read_csv(
                path, sep=';', engine='python', encoding=enc,
                on_bad_lines='skip'
            )
            return df
        except Exception as e:
            last_err = e
    raise SystemExit(f"No pude leer el CSV tabular con ';'. Último error: {last_err}")

def build_transactions_from_df(df: pd.DataFrame):
    # Validar columnas
    cols = set(df.columns)
    needed = {TICKET_COL, ITEM_COL, QTY_COL}
    missing = needed - cols
    if missing:
        raise SystemExit(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {list(df.columns)}")

    # Mantener solo lo necesario
    df = df[[TICKET_COL, ITEM_COL, QTY_COL]].copy()

    # Limpiar Itemname
    df[ITEM_COL] = df[ITEM_COL].astype(str).str.strip()
    df = df[df[ITEM_COL].str.len() > 0]

    # Quantity > 0 (numérica)
    with pd.option_context('mode.use_inf_as_na', True):
        qty = pd.to_numeric(df[QTY_COL], errors='coerce').fillna(0)
    df = df[qty > 0]

    # Agrupar por BillNo → lista de productos (sin duplicados dentro del mismo ticket)
    tx = (df.groupby(TICKET_COL)[ITEM_COL]
            .apply(lambda s: list(dict.fromkeys(s)))  # mantiene orden y quita dupes
            .tolist())

    return tx

def main():
    df = read_tabular_strict(DATA_PATH)
    transactions = build_transactions_from_df(df)

    if not transactions:
        pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"]).to_csv(
            os.path.join(OUTDIR, "fpgrowth_rules.csv"), index=False
        )
        raise SystemExit("No se pudieron construir transacciones válidas.")

    # ---- Conteo para filtrar ítems infrecuentes (previo al one-hot) ----
    item_counts = Counter()
    for items in transactions:
        item_counts.update(items)
    num_tx = len(transactions)
    min_count = math.ceil(MIN_SUPPORT * num_tx)
    frequent_items = {it for it, cnt in item_counts.items() if cnt >= min_count}
    del item_counts; gc.collect()

    # Filtrar transacciones por ítems frecuentes
    transactions = [[it for it in tx if it in frequent_items] for tx in transactions]
    transactions = [tx for tx in transactions if tx]  # quitar vacías

    if not transactions:
        pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"]).to_csv(
            os.path.join(OUTDIR, "fpgrowth_rules.csv"), index=False
        )
        raise SystemExit(f"No quedaron ítems con soporte >= {MIN_SUPPORT:.2f} tras filtrar.")

    # ---- One-hot disperso ----
    te = TransactionEncoder()
    te_ary_sparse = te.fit(transactions).transform(transactions, sparse=True)
    df_onehot = pd.DataFrame.sparse.from_spmatrix(te_ary_sparse, columns=te.columns_)
    del transactions; gc.collect()

    # ---- FP-Growth ----
    freq_itemsets = fpgrowth(df_onehot, min_support=MIN_SUPPORT, use_colnames=True, max_len=MAX_LEN)
    if freq_itemsets.empty:
        pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"]).to_csv(
            os.path.join(OUTDIR, "fpgrowth_rules.csv"), index=False
        )
        print(f"Sin itemsets frecuentes con soporte >= {MIN_SUPPORT:.2f}."); return

    rules = association_rules(freq_itemsets, metric=RULE_METRIC, min_threshold=MIN_THRESHOLD)
    if rules.empty:
        pd.DataFrame(columns=["antecedents","consequents","support","confidence","lift"]).to_csv(
            os.path.join(OUTDIR, "fpgrowth_rules.csv"), index=False
        )
        print("No se generaron reglas con los umbrales establecidos."); return

    # Ordenar por lift y serializar sets para CSV
    rules = rules.sort_values(by="lift", ascending=False).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

    out_path = os.path.join(OUTDIR, "fpgrowth_rules.csv")
    rules[["antecedents","consequents","support","confidence","lift"]].to_csv(out_path, index=False)

    print("Top 10 reglas (ya sin países ni números):")
    print(rules[["antecedents","consequents","support","confidence","lift"]].head(10))
    print(f"\nGuardado: {out_path}")

if __name__ == "__main__":
    main()
