import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
"""
#--------------------------BLOQUE 1------------------------------
# Objetivo: entender de forma práctica cómo un modelo “baja la montaña”.

#1.1 Queremos minimizar f(x)= x^2 con una LR y con n iteracciones

lr = 0.1
iteraciones = 10
x = 5

def funcion(x):
    return x**2

def derivada(x):
    return 2*x

for i in range(iteraciones):
    grad = derivada(x)
    x = x - lr * grad
    print(f"Iteración {i+1}: x = {x:.4f}, f(x) = {funcion(x):.4f}")


# 1.2 Ejercicio visual. Usa el siguiente código para ver como cambia el descenso segun alpha(lr)

# Función y derivada
def f(x): return x**2
def df(x): return 2*x

# Parámetros iniciales
x = 5
alpha = 0.01
hist = []

for i in range(10):
    x = x - alpha * df(x)
    hist.append(x)

plt.plot(range(10), [f(xi) for xi in hist], 'o-')
plt.xlabel('Iteración')
plt.ylabel('Valor de f(x)')
plt.title('Descenso del coste con alpha=0.1')
plt.show()
"""
# ------------------------BLOQUE 2-------------------------
#2.1 Queremos minimizar f(x,y) = x^2 + y^2

def funcion2(x,y):
    return x**2 + y**2

def gradiente(x,y):
    return np.array([2*x, 2*y])

# Parametros
alpha = 0.1
w = np.random.random(2) #Punto inicial aleatorio
iteraciones = 10

for i in range(iteraciones):
    grad = gradiente(w[0], w[1])
    w = w - alpha * grad
    print(f"Iteración {i+1}: w1={w[0]:.4f}, w2={w[1]:.4f}, f={funcion2(w[0], w[1]):.4f}")


































