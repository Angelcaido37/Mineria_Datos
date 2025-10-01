# -*- coding: utf-8 -*-
"""
07_association_rules_apriori_groceries_dataset.py
Versión académica adaptada para el archivo Groceries_dataset.csv

Este script:
- Carga el dataset Groceries_dataset.csv (transacciones cliente-producto)
- Agrupa las compras por cliente (Member_number)
- Transforma las transacciones a formato binario (one-hot encoding)
- Aplica el algoritmo Apriori para obtener itemsets frecuentes
- Genera reglas de asociación y las ordena por lift
- Muestra las 10 reglas más fuertes en consola y en una gráfica
"""

# ==== IMPORTACIONES ====
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ==== PARÁMETROS DEL DATASET ====
CSV_PATH = Path("Datasets/Groceries_dataset.csv")   # Ruta del archivo cargado

if __name__ == "__main__":
    # 1. Cargar dataset
    assert CSV_PATH.exists(), f"No se encontró el archivo {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)

    # 2. Revisar estructura
    print("Primeras filas del dataset:")
    print(df.head())

    # 3. Agrupar transacciones por cliente (Member_number)
    # Cada cliente tendrá una lista de productos que compró
    transactions = df.groupby("Member_number")["itemDescription"].apply(list).tolist()
    print(f"Total de transacciones agrupadas: {len(transactions)}")
    print("Ejemplo de transacción:", transactions[0])

    # 4. Transformar a formato binario con TransactionEncoder
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_bin = pd.DataFrame(te_array, columns=te.columns_)

    print(f"Dimensiones matriz binaria: {df_bin.shape}")  # (n_transacciones, n_productos)

    # 5. Aplicar Apriori (umbral mínimo de soporte 1%)
    frequent_itemsets = apriori(df_bin, min_support=0.01, use_colnames=True)
    print(f"Itemsets frecuentes encontrados: {len(frequent_itemsets)}")

    # 6. Generar reglas de asociación (confianza mínima 30%)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

    # 7. Ordenar reglas por lift
    rules_sorted = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    print(f"Reglas generadas: {len(rules_sorted)}")

    # 8. Mostrar las 5 reglas principales
    print("\nTop 5 reglas más relevantes:\n")
    print(rules_sorted.head(5)[["antecedents", "consequents", "support", "confidence", "lift"]])

    # 9. Seleccionar las Top-10 para graficar
    top = rules_sorted.head(10).copy()
    labels = [
        ", ".join(sorted(list(a))) + " → " + ", ".join(sorted(list(c)))
        for a, c in zip(top["antecedents"], top["consequents"])
    ]

    # 10. Gráfica: Top-10 reglas por lift
    plt.figure(figsize=(9, 6))
    plt.barh(range(len(top)), top["lift"].values, color="skyblue")
    plt.yticks(range(len(top)), labels, fontsize=9)
    plt.gca().invert_yaxis()
    plt.xlabel("Lift")
    plt.title("Top-10 Reglas de Asociación (Groceries Dataset)")
    plt.tight_layout()
    plt.show()
