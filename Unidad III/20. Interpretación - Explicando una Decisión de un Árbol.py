# -*- coding: utf-8 -*-
"""
Tema: 10.2 Interpretación de Modelos - Decisiones Trazables en Árboles
Dataset: Iris
Explicación: La gran ventaja de un solo Árbol de Decisión es su "transparencia"
o interpretabilidad. A diferencia de modelos complejos como las redes neuronales
(que son "cajas negras"), podemos seguir exactamente el camino de decisión que
tomó el árbol para clasificar una muestra específica. Esto es extremadamente
valioso para entender y validar el comportamiento del modelo.
"""

# 1. Importar librerías
# ---------------------
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 2. Cargar datos y entrenar un árbol simple
# ------------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Entrenamos un árbol con una profundidad limitada para que sea fácil de visualizar y seguir.
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# 3. Seleccionar una muestra para explicar
# ----------------------------------------
# Tomaremos una muestra del dataset para ver cómo la clasifica el modelo.
# Podría ser cualquier conjunto de nuevas mediciones.
sample_index = 140 # Un ejemplo de la especie 'virginica'
sample_to_explain = X[sample_index]
true_class = class_names[y[sample_index]]

print("--- Explicando la Predicción para una Muestra Específica ---")
print(f"Índice de la muestra: {sample_index}")
print(f"Características de la muestra: {pd.Series(sample_to_explain, index=feature_names)}")
print(f"Clase real de la muestra: '{true_class}'")

# 4. Obtener la predicción del modelo
# -----------------------------------
prediction_index = model.predict([sample_to_explain])[0]
prediction_class = class_names[prediction_index]
prediction_proba = model.predict_proba([sample_to_explain])

print(f"\nPredicción del modelo: '{prediction_class}'")
print(f"Probabilidades predichas: {dict(zip(class_names, prediction_proba[0]))}")

# 5. Trazar el camino de decisión
# -------------------------------
# `decision_path` nos da los nodos que la muestra atravesó.
# `apply` nos da el ID del nodo hoja final al que llegó la muestra.
decision_path = model.decision_path([sample_to_explain])
leaf_id = model.apply([sample_to_explain])[0]
node_indices = decision_path.indices

# Accedemos a la estructura interna del árbol entrenado
tree = model.tree_

print("\n--- Camino de Decisión que siguió la muestra ---")
for node_id in node_indices:
    # Si no es un nodo hoja
    if leaf_id != node_id:
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_value = sample_to_explain[feature_idx]

        # ¿La muestra fue a la izquierda o a la derecha?
        if feature_value <= threshold:
            direction = "izquierda"
            decision_rule = f"<= {threshold:.2f}"
        else:
            direction = "derecha"
            decision_rule = f"> {threshold:.2f}"
        
        print(f"Nodo {node_id}: {feature_names[feature_idx]} ({feature_value:.2f}) {decision_rule} -> va a la {direction}")

print(f"Nodo Hoja Final {leaf_id}: Se toma la decisión final.")
final_decision_values = tree.value[leaf_id][0]
final_decision_class = class_names[np.argmax(final_decision_values)]
print(f"   Distribución de clases en esta hoja: {dict(zip(class_names, final_decision_values))}")
print(f"   Clase mayoritaria en la hoja: '{final_decision_class}'")


# 6. Visualizar el árbol y resaltar el camino
# --------------------------------------------
# Esto requiere un poco más de código para la visualización avanzada.
plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
plt.title("Camino de Decisión para la Muestra 140")
plt.show()

# 7. Exportar las reglas como texto
# ---------------------------------
# Una forma muy directa de ver todas las reglas que aprendió el modelo.
tree_rules = export_text(model, feature_names=feature_names)
print("\n--- Reglas del Árbol en Formato de Texto ---")
print(tree_rules)