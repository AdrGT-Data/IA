import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Cargar datos
data = fetch_california_housing(as_frame=True)   #Cargamos los datos. El true hace que vengan
# en dataframes de pandas y no como arrays de Numpy
X = data.data # Guardamos las variables independientes
y = data.target # Guardamos la variable a predecir (precio)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
train_test_split(var inde, var dep, test_size= porcentage del conjunto para test, Semilla)
En este caso usamos 80% train, 20% test
"""

# 2) Pipeline + GridSearchCV
pipe = Pipeline([
    ("scaler", StandardScaler()), #Estandarizar (media = 0, desviación = 1)
    ("model", Ridge()) # Modelo de regresión Ridge en este caso.
])
"""
El modelo de regresión Ridge permite reducir los coeficientes /beta de la formula Y=BX + E
Este método reduce el peso de los coeficientes pero nunca los anula ( al contario que el lasso).
Esto permite dar oportunidad al analista de eliminar si ve necesario las variables con bajo peso.
"""

param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 100]}  # Definimos los /alpha que vamos a probar

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,  #Validación cruzada de 5 divisiones
    scoring="neg_root_mean_squared_error", #Métrica de error MSER
    n_jobs=-1
)

grid.fit(X_train, y_train)  #Entrenamos al modelo

best = grid.best_estimator_  #Nos quedamos con el modelo
print("Mejor alpha:", grid.best_params_["model__alpha"]) # Imprimimos el mejor alpha


# 3) Evaluación en test
y_pred = best.predict(X_test) #Probamos el modelo con test
rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)


print(f"RMSE test: {rmse:.4f}")
print(f"MAE  test: {mae:.4f}")
print(f"R2   test: {r2:.4f}")

# Gráfico y_real vs y_pred
plt.scatter(y_test, y_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c="red")
plt.xlabel("y real")
plt.ylabel("y predicha")
plt.title("Predicho vs Real (Ridge)")
plt.show()

# 4) Interpretación de coeficientes (en espacio estandarizado)
feature_names = X.columns
coefs = best.named_steps["model"].coef_
for name, c in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
    print(f"{name:>12s}: {c:+.3f}")

"""
INTERPRETACIÓN

Este código muestra un modelo al que le aplicamos la tecnica de regularizacióbn Ridge
Para interpretar los resultados deberíamos saber acerca del contexto del problema y el fin del
análisis.

En los resultados podemos ver el RMSE (Root Mean Square Error) y el MAE (Mean Absolute Error) además
del R2 que podríamos decir que es como una métrica que mide como la variabilidad de y es explicada por
el modelo.

Además se nos muestran los coeficientes para cada una de las variables independientes o predictoras.
El hecho de eliminar las variables con bajo coeficiente (~0) deberá de ser espaldado con 
explicaciones que aseguren que es útil y por qué lo és antes de eliminarla.
"""

