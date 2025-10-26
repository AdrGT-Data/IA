"""
En este tema programa haremos un proyecto de regresión aplicando lo visto en el tema 3.
Con ello el objetivo es implementar una regresión lineal simple y = wx + b y que el sistema
por descenso de gradiente aprenda los parámetros w y b.

Dicho queda que no usaremos otra libreria que no sea numpy para realizarlo, sin librerias de ML.

Para facilitar el proyecto, los datos serán sencillos y se describen a continuación.
Para facilitar la repetición de este ejercicio, el enunciado vendra entre las dos lineas
horizontales del inicio
"""
#---------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

#Datos:
x = np.random.randn(10)
y = np.random.randn(10)
# Parametros
w = 0.0    # peso inicial
b = 0.0    # sesgo inicial
alpha = 0.01   # learning rate
iteraciones = 1000  # número de iteraciones
n = len(x)
#---------------------------------------------------------------

#Definimos la función de coste que hará que el algoritmo sepa si esta muy equivocado o no
# Usaremos MSE (Mean Squared Error)

def MSE(y, y_pred):
    return np.mean((y_pred - y)**2)

# Gradiente Descendente
coste_history = []

for i in range(iteraciones):
    y_pred = w*x + b

    #Calculo del coste
    cost= MSE(y, y_pred)
    coste_history.append(cost)

    #Gradiente de la función a minimizar (la de coste) y con los parámetros que queremos optimizar(w,b)
    dw = (-n/2) * np.sum(x*(y-y_pred))
    db = (-n/2) * np.sum(y-y_pred)
    grad = np.array([dw, db])

    #Actualizamos los parámetros
    w = w - alpha * grad[0]
    b = b - alpha * grad[1]
"""
    # Mostrar cada 100 iteraciones
    if iteraciones % 100 == 0:
        print(f"Iter {iteraciones:4d} | Coste={cost:.4f} | w={w:.4f} | b={b:.4f}")
"""
print(f"\nModelo final: y = {w:.2f}x + {b:.2f}")


#Representación de resultado
plt.scatter(x, y, color='blue')
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y_pred, color='red')
plt.title("Resultado final")

plt.show()
















