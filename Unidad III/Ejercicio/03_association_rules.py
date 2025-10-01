
# -*- coding: utf-8 -*-
"""
Script 03: Reglas de asociación con Apriori (implementación simple en pandas).
Calcula soporte, confianza y lift sobre "cestas" definidas a partir de acciones
y resultados. Visualiza las TOP reglas por lift.

Ejecuta: python 03_association_rules.py
Requiere: loan_recovery_dataset.csv en el mismo directorio.
"""

import pandas as pd
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict

# Cargamos el dataset desde el CSV
from pathlib import Path
csv_path = Path("Unidad III") / "Ejercicio" / "loan_recovery_dataset.csv"
df = pd.read_csv(csv_path)

# Construimos "cestas" de items por prestatario para análisis de reglas.
# Incluimos:
# - Segmentos discretizados (ej. DPD alto/bajo, DTI alto/bajo, Score alto/bajo)
# - Acción tomada
# - Resultado pagó/no pagó 30d (como "label" en el conjunto)
def discretize(row):
    items = []
    items.append("DPD_{}".format("ALTO" if row["days_past_due"] >= 30 else "BAJO"))
    items.append("DTI_{}".format("ALTO" if row["dti_percent"] >= 40 else "BAJO"))
    items.append("SCORE_{}".format("ALTO" if row["credit_score"] >= 680 else "BAJO"))
    items.append("CANAL_{}".format(row["action_taken"]))
    items.append("REGION_{}".format(row["region"]))
    items.append("RESP_{}".format(row["last_contact_response"]))
    items.append("OUTCOME_{}".format("PAGO30" if row["paid_within_30d"]==1 else "NOPAGO30"))
    return items

baskets = df.apply(discretize, axis=1).tolist()

# Función para calcular soporte de itemsets
def apriori(baskets, min_support=0.05):
    n = len(baskets)
    # 1) Cálculo de soporte de itemsets de 1 elemento
    item_counts = defaultdict(int)
    for basket in baskets:
        for item in set(basket):
            item_counts[frozenset([item])] += 1
    L1 = {items: cnt/n for items, cnt in item_counts.items() if cnt/n >= min_support}
    Lk = L1
    L_all = dict(L1)
    k = 2
    # 2) Iterativamente generamos candidatos y filtramos por soporte
    while True:
        # Generación de candidatos Ck a partir de Lk-1
        items_prev = list(Lk.keys())
        candidates = set()
        for i in range(len(items_prev)):
            for j in range(i+1, len(items_prev)):
                union = items_prev[i].union(items_prev[j])
                if len(union) == k:
                    candidates.add(union)
        if not candidates:
            break
        # Conteo de soporte
        cand_counts = defaultdict(int)
        for basket in baskets:
            bset = set(basket)
            for cand in candidates:
                if cand.issubset(bset):
                    cand_counts[cand] += 1
        # Filtrado por min_support
        Lk = {items: cnt/n for items, cnt in cand_counts.items() if cnt/n >= min_support}
        if not Lk:
            break
        L_all.update(Lk)
        k += 1
    return L_all

# Ejecutamos Apriori con un soporte mínimo (ajusta según tamaño)
supports = apriori(baskets, min_support=0.06)  # 6%

# Convertimos a DataFrame para ordenar y filtrar reglas
# Construimos reglas del tipo X -> Y donde Y es "OUTCOME_*"
rows = []
support_map = {tuple(sorted(list(k))): v for k, v in supports.items()}

def support_of(itemset):
    key = tuple(sorted(list(itemset)))
    return support_map.get(key, 0.0)

# Generamos reglas para itemsets de tamaño >= 2
for itemset, supp in supports.items():
    if len(itemset) < 2:
        continue
    items = list(itemset)
    for r in range(1, len(items)):
        for antecedent in itertools.combinations(items, r):
            consequent = tuple(sorted(set(items) - set(antecedent)))
            # Solo reglas que concluyan en OUTCOME_*
            if not any(i.startswith("OUTCOME_") for i in consequent):
                continue
            antecedent = tuple(sorted(antecedent))
            supp_x = support_of(antecedent)
            supp_y = support_of(consequent)
            conf = supp / supp_x if supp_x > 0 else 0.0
            lift = conf / supp_y if supp_y > 0 else 0.0
            rows.append({
                "antecedent": antecedent,
                "consequent": consequent,
                "support": supp,
                "confidence": conf,
                "lift": lift
            })

rules = pd.DataFrame(rows)
rules = rules.sort_values(["lift","confidence","support"], ascending=False).reset_index(drop=True)

print("Top 15 reglas por lift (X -> OUTCOME_*):")
print(rules.head(15))

# Graficamos las 15 mejores por lift (lift vs confianza con tamaño por soporte)
top = rules.head(15)
plt.figure()
plt.scatter(top["lift"], top["confidence"], s=top["support"]*1000)
for idx, row in top.iterrows():
    label = " & ".join([a.replace("CANAL_","") for a in row["antecedent"] if a.startswith("CANAL_")]) or "Mix"
    plt.annotate(label, (row["lift"], row["confidence"]))
plt.xlabel("Lift")
plt.ylabel("Confianza")
plt.title("Top reglas (por lift) hacia OUTCOME_*")
plt.tight_layout()
plt.show()
