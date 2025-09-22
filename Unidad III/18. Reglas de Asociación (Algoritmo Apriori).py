# -*- coding: utf-8 -*-
"""
Tema: 9. Reglas de Asociación (Análisis de la Cesta de Mercado)
Librería: mlxtend
Explicación: Las reglas de asociación son una técnica de minería de datos
utilizada para descubrir relaciones interesantes entre variables en grandes
bases de datos. El ejemplo más famoso es el "análisis de la cesta de mercado",
que busca encontrar productos que se compran juntos con frecuencia.
El algoritmo Apriori es el método clásico para esta tarea.

Se basa en tres métricas clave:
- **Soporte (Support):** Qué tan frecuentemente aparece un item o conjunto
  de items en todas las transacciones. Soporte({Pan}) = (Transacciones con Pan) / (Total Transacciones).
- **Confianza (Confidence):** La probabilidad de comprar el item Y, dado que
  se compró el item X. Confianza({X} -> {Y}) = Soporte({X, Y}) / Soporte({X}).
- **Elevación (Lift):** Mide qué tan probable es comprar Y si se compra X,
  mientras se controla qué tan popular es Y.
  - Lift = 1: No hay asociación.
  - Lift > 1: Los items se compran juntos más a menudo de lo esperado (asociación positiva).
  - Lift < 1: Los items se compran juntos menos a menudo de lo esperado (asociación negativa).
"""
# Se necesita instalar mlxtend: pip install mlxtend
# 1. Importar librerías
# ---------------------
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 2. Crear un dataset de transacciones de ejemplo
# -----------------------------------------------
dataset = [
    ['Leche', 'Cebolla', 'Nuez Moscada', 'Frijoles', 'Huevos', 'Yogurt'],
    ['Papas', 'Cebolla', 'Nuez Moscada', 'Frijoles', 'Huevos', 'Yogurt'],
    ['Leche', 'Manzana', 'Frijoles', 'Huevos'],
    ['Leche', 'Maíz', 'Frijoles', 'Yogurt'],
    ['Maíz', 'Cebolla', 'Jugo', 'Queso']
]
print("Dataset de transacciones original:")
print(dataset)

# 3. Transformar los datos al formato requerido
# ---------------------------------------------
# El algoritmo Apriori de mlxtend necesita los datos en un formato de
# "one-hot encoding", donde cada fila es una transacción y cada columna es
# un producto. El valor es True si el producto está en la transacción, False si no.
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print("\nDataset transformado a formato one-hot:")
print(df)

# 4. Aplicar el algoritmo Apriori para encontrar "itemsets" frecuentes
# ---------------------------------------------------------------------
# Un "itemset" es simplemente un conjunto de uno o más items.
# `min_support=0.6` significa que solo nos interesan los itemsets que
# aparecen en al menos el 60% de las transacciones.
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

print("\nItemsets Frecuentes (Soporte >= 0.6):")
print(frequent_itemsets)

# 5. Generar las reglas de asociación a partir de los itemsets frecuentes
# -----------------------------------------------------------------------
# Ahora, a partir de los itemsets frecuentes, generamos las reglas.
# `metric="confidence"` y `min_threshold=0.7` significa que solo queremos
# las reglas que tengan una confianza de al menos 0.7 (70%).
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# 'antecedents' -> 'consequents' se lee como "Si el cliente compra los antecedentes,
# entonces probablemente también comprará los consecuentes".
print("\nReglas de Asociación (Confianza >= 0.7):")
# Mostramos las columnas más relevantes
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 6. Interpretación de una regla de ejemplo
# -----------------------------------------
# Tomemos la regla: {Cebolla, Huevos} -> {Frijoles}
print("\n--- Interpretación de una Regla de Ejemplo ---")
# Filtramos para encontrar esa regla específica
example_rule = rules[
    (rules['antecedents'] == {'Cebolla', 'Huevos'}) &
    (rules['consequents'] == {'Frijoles'})
]

if not example_rule.empty:
    support = example_rule['support'].iloc[0]
    confidence = example_rule['confidence'].iloc[0]
    lift = example_rule['lift'].iloc[0]
    
    print("Regla: Si un cliente compra {Cebolla, Huevos}, también comprará {Frijoles}")
    print(f"- Soporte ({support:.2f}): El 60% de TODAS las transacciones contienen Cebolla, Huevos y Frijoles juntos.")
    print(f"- Confianza ({confidence:.2f}): El 100% de las veces que un cliente compró Cebolla y Huevos, también compró Frijoles.")
    print(f"- Lift ({lift:.2f}): Un cliente tiene 1.25 veces más probabilidades de comprar Frijoles si ya ha comprado Cebolla y Huevos, en comparación con la probabilidad de comprar Frijoles en general.")
    print("  (Como Lift > 1, es una asociación positiva y potencialmente útil para estrategias de marketing).")
else:
    print("La regla de ejemplo no se generó con los umbrales actuales.")
