import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
# -------------------------------BLOQUE 1 --------------------------
#1.1 Genera 20 puntos (x1, x2) aleatorios entre 0 y 1 con numpy
"""
X = np.random.rand(20,2)

#1.2 Asigna la etiqueta y= 1 si x1+x2 >1
Y = (X[:,0] + X[:,1]>1).astype(int)
print(Y)

#1.3 Representar los punto en el plano variando el color en funcion de si y=1 o y = 0
plt.scatter(X[:,0],X[:,1], c= Y, cmap="bwr", edgecolors="k")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Dataset")
#plt.show()
"""
# -------------------------------BLOQUE 2:Implementar una neurona -----------------------------

#2.1 Escribe la funcion neuron(x,w,b) que reciba un vector x, otro de pesos w y un bias b y devuelva la salida de
# una neurona aplicando la sigmoide sig(z)=1/ (1+e^-z) para z = (x1w1+...+xnwn)+b

def signoid(z):
    signoid = 1 / (1 + np.exp(-z))
    return signoid

def neuron(x,w,b):
    z = np.dot(x,w) + b
    return signoid(z)

#Ejemplo
"""
x=[0.5, 0.7]
w=[0.1, -0.2]
b= 0.05
"""
# -------------------------------BLOQUE 3: Entrenar la neurona-----------------------------
#3.1 Crea el dataset de la compuerta AND
X= np.array([[0,0],[0,1],[1,0],[1,1]])
Y= np.array([0,0,0,1])

#Entrena la neurona con 20 interacciones
#Inicializamos con unos pesos y bis aleatorios
iteracciones = 200000
w=np.random.rand(2)
b=np.random.rand(1)
lr = 0.1 #Tasa aprendizaje/learning rate
for epoch in range(iteracciones):
    for i in range (len(X)):
        x_i = X[i]
        y_i = Y[i]

        y_pred= neuron(x_i,w,b) #Calculamos la prediccion de la iteracion

        error= y_pred - y_i #Calculo el error

        #Actualizamos los parámetros
        w = w - lr * error * x_i
        b = b - lr * error
    if epoch % 20000==0: #Mostramos los parametros cada ciertos valores
        print(f"Iteraccion {epoch}: w = {w}, b = {b}")

#3.3 Muestra cada 5 iteraciones como van cambiando los pesos y el error
for i in range(len(X)):
    print(f"Entrada: {X[i]}, Predicción: {neuron(X[i],w,b)}")









