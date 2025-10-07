import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# En este apartado del tema vamos a trabajar los arboles de decisión usando un ejemplo de clasificación
# muy básico: Usando el dataset Iris trataremos de clasificar flores en función de ciertas características
# de sus pétalos y sépalos.

#--------------------------BLOQUE 1: Árbol de Decisión------------------------------
# Objetivo: entrenar, visualizar y evaluar un árbol individual.

# 1. Cargamos los datos y asignamos las variables independientes(X) y la variable respuesta (y)
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Hacemos la partición 80/20 como ya vimos en apartados anteriores
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Creamos el modelo
DecisionTree = DecisionTreeClassifier(random_state=42,max_depth=1) #Establecemos un maximo de profundidad con el fin de evitar sobreajustes

# Y lo entrenamos...
DecisionTree.fit(X_train, y_train)

# 3. Evaluación
y_pred_train = DecisionTree.predict(X_train)
y_pred_test  = DecisionTree.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test  = accuracy_score(y_test,  y_pred_test)

print(f"Accuracy Tree (train): {acc_train:.3f}")
print(f"Accuracy Tree (test) : {acc_test:.3f}")
print(confusion_matrix(y_train, y_pred_train))

"""
ELIMINAR COMILLAS PARA VER EL ÁRBOL
# 4. Visualización
plt.figure(figsize=(12, 8))
plot_tree(
    DecisionTree,
    feature_names=X.columns,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Árbol de decisión (Iris)")
plt.show()
"""
# --------------------------BLOQUE 2-------------------------------
# Objetivo: Ver cómo mejora al hacer un bosque/forest

# Creamos el modelo para 100 arboles. Para poner más parámetros útiles consultar la documentación.
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=1)

# Entrenamos el modelo
rf.fit(X_train, y_train)

# Evaluación
y_pred_train_rf = rf.predict(X_train)
y_pred_test_rf = rf.predict(X_test)

acc_train_rf = accuracy_score(y_train, y_pred_train_rf)
acc_test_rf = accuracy_score(y_test, y_pred_test_rf)

print(f"Accuracy RF (train): {acc_train_rf:.3f}")
print(f"Accuracy RF (test) : {acc_test_rf:.3f}")
print(confusion_matrix(y_train, y_pred_train_rf))

"""
Al comparar resultados, teniendo en cuenta que el dataset es muy simple,
ambos resultados son muy buenos pero si nos ponemos a probar con poca profundidad la mejora del bosque se 
hace mucho más notable.
"""

#-----------------------BLOQUE 3---------------------------
# Objetivo: Entender qué variables importan y cómo afectan al modelo

importances = rf.feature_importances_
order = np.argsort(importances)[::-1]  # de mayor a menor

print("\nImportancias por impureza:")
for idx in order:
    print(f"{X.columns[idx]:>15}: {importances[idx]:.3f}")

# Gráfico de barras
plt.figure(figsize=(6,4))
plt.bar(range(len(importances)), importances[order])
plt.xticks(range(len(importances)), X.columns[order], rotation=45)
plt.ylabel("Importancia (impureza)")
plt.title("Random Forest — Importancia de variables")
plt.tight_layout()
plt.show()

"""
Para este bloque analizamos brevemente la importancia relativa de las variables. La función que usamos 
calcula cuanto baja la impureza ponderado por cuántas muestras pasan por ese nodo. Luego suma esa ganancia

Con los resultados vemos claramente que la anchura de los sépalos en completamente irrelevante.

Investigando un mínimo las plantas del dataset tiene todo el sentido que el programa haya aprendido que lo más relevante
sean los pétalos ya que en las tres plantas sus petalos son muy diferentes entre sí y es fácil identificar
la planta solo con el pétalo.

Sería interesante buscar un dataset en el que hubiera más tipos de plantas y así el programa se vería obligado
a usar la información de los sépalos para clasificar.
"""


















