import torch, torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

"""
En este proyecto haremos un modelo neuronal con MNIST (numeros incritos a mano) como mini-proyecto del ttema 5.
"""

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Transformación: convertir a tensor y normalizar entre -1 y 1
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar datasets de entrenamiento y prueba
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Crear dataloaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Imágenes de entrenamiento: {len(train_data)}")
print(f"Imágenes de prueba: {len(test_data)}")

print("----------------------------------------------")

#-----------------------------------------------
"""
Hasta aquí solo hemos descargado los dataset y particionado los datos. Además hemos vistoq ue trabajaremos
en la cpu del ordenador.

Los datos descargados son 70000 muestras de imagenes en escala de grises de 28x28 pixeles que
muestran numeros escritos a mano y una etiqueta que le pone el nunmero que es.

Ahora vamos a crear el modelo neuronal que pretenderá predecir que numero es el de una imagen nueva que pongamos.
"""

#-----------------------------------------------

device = torch.device("cpu")  # ya comprobamos que es CPU

# Modelo: 28*28 -> 256 -> 128 -> 10
model = nn.Sequential( #Esta parte permite encadenar las capas secuencialmente una por una
    nn.Flatten(), # -> COnvierte los tensores [1,28,28] en [784] que es 28x28 para que la siguiente linea los pueda procesar
    nn.Linear(28*28, 256), # Crea una capa densa con 784 entradas y 256 neuronas
    nn.ReLU(), #F. Activación
    nn.Linear(256, 128), #Capa intermedia
    nn.ReLU(),
    nn.Linear(128, 10)   # #Capa final con 10 salidas posibles (0-9)
).to(device) #Mueve todo el modelo a cpu

# Función para contar parámetros entrenables
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(model)
print("Parámetros entrenables:", count_params(model))

# Sanity check: pasar un batch por el modelo y ver la forma
X_batch, y_batch = next(iter(train_loader))
X_batch = X_batch.to(device)
logits = model(X_batch)
print("Batch input shape:", X_batch.shape)   # [64, 1, 28, 28]
print("Logits shape:", logits.shape)         # [64, 10]

# ENTRENAMIENTO

# 1) Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()           # para clasificación multiclase con logits
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Adam suele converger rápido

# 2) Función auxiliar: calcular accuracy para evaluar el modelo.
def accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)               # [batch, 10]
            preds = logits.argmax(dim=1)    # clase más probable
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 3) Entrenamiento
epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        # forward
        #El model genera predicciones y calcula el error
        logits = model(X)
        loss = criterion(logits, y)

        # backward
        optimizer.zero_grad() #Reset a gradientes anteriores
        loss.backward() # Calcula derivadas usando la regla de la cadena de la ultima capa a la primera
        optimizer.step() # Actualiza los pesos

        running_loss += loss.item()

    # métricas por época
    train_loss = running_loss / len(train_loader)
    test_acc = accuracy(model, test_loader, device)
    print(f"Epoch {epoch}/{epochs} | loss: {train_loss:.4f} | test_acc: {test_acc:.4f}")