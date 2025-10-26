from functools import cache

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Fuerza backend gráfico
import matplotlib.pyplot as plt

#------------------------Puerta XOR-------------------------------
#En este miniproyecto crearemos una puerta XOR que aprenda a traves de redes neuronales 2-2-1 y
# aplicando el backpropagation para que ello.

# Entradas (4 combinaciones posibles)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
# Etiquetas (0 si iguales, 1 si diferentes)
y = np.array([[0], [1], [1], [0]])

# Para reproducibilidad
np.random.seed(0)

# Establecemos una funcion que devuelve unos parametros aleatorios iniciales
def init_params():
    W1 = np.random.randn(2, 2) * 0.1
    b1 = np.zeros((2, 1))
    W2 = np.random.randn(1, 2) * 0.1
    b2 = np.zeros((1, 1))
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params()

# Creamos las funciones que necesitaremos

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def binary_cross_entropy(y, y_hat):
    y_hat = np.clip(y_hat, 1e-9, 1 - 1e-9) #clip() limita los valores
    return -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))

#EMpezamos con el forward
def forward_pass(X, W1, b1, W2, b2):
    # Transponemos X para que cada columna sea un ejemplo (2x4)
    X = X.T
    Z1 = W1 @ X + b1          # (2,4)
    H1 = relu(Z1)             # (2,4)
    Z2 = W2 @ H1 + b2         # (1,4)
    Y_hat = sigmoid(Z2)       # (1,4)
    cache = (X, Z1, H1, Z2, Y_hat)
    return Y_hat, cache

#Seguimos con el backpropagation
def backward_pass(y, cache, W1, b1, W2, b2, lr=0.1):
    X, Z1, H1, Z2, Y_hat = cache
    m = X.shape[1]  # nº ejemplos

    # Paso 1: error en salida
    dZ2 = Y_hat - y.T                            # (1,4)
    dW2 = (dZ2 @ H1.T) / m                       # (1,2)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m # (1,1)

    # Paso 2: error capa oculta
    dH1 = W2.T @ dZ2                             # (2,4)
    dZ1 = dH1 * relu_deriv(Z1)                   # (2,4)
    dW1 = (dZ1 @ X.T) / m                        # (2,2)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m # (2,1)

    # Paso 3: actualizar parámetros
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    return W1, b1, W2, b2

#ENTRENAMIENTO

iteraciones= 10000
lr = 0.1
errores=[]

for i in range(iteraciones):
    Y_hat, cache = forward_pass(X, W1, b1, W2, b2)
    #Metemos los errores en una lista
    error = binary_cross_entropy(y,Y_hat)
    errores.append(error)

    # Hacemos el back
    W1, b1, W2, b2 = backward_pass(y, cache, W1, b1, W2, b2, lr)
    if i % 1000 == 0:
        print(f"Epoch {i:5d} | Loss = {error:.4f}")

#Representamos la evolucion d ela perdida
plt.plot(errores)
plt.title("Evolución del error (XOR)")
plt.xlabel("Iteraciones")
plt.ylabel("Binary Cross-Entropy")
plt.show()

#Resultados finales
Y_pred, _ = forward_pass(X, W1, b1, W2, b2)
print("Predicciones finales:\n", np.round(Y_pred.T, 3))


















