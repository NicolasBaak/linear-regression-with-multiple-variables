import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#======================== Lectura de DB y Normalización ================================
def readDatabase(ruta):
    path = os.path.abspath(os.getcwd() + ruta ) 
    df = pd.read_csv(path)
    df.insert(0, 'Ones', 1)
    cols = df.shape[1] # df = (13000, 9), cols = 9

    # X elimino la ultima columna
    X = df.iloc[:, 0:cols-1] # valores de entrada (13000, 8)

    # y asigno la ultima columna 
    y = df.iloc[:, cols-1:cols]  # valores de salida (13000, 1)

    # Convertir a matrices para usar numpy e inicializar theta
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.zeros(cols-1))
    return df, X, y, theta

def feautureNormalize(X, ini):
    X_norm = X.copy()
    parameters = int(X.shape[1])
    for i in range(parameters-ini):
        minimo = np.min(X[:, i+ini])
        maximo = np.max(X[:, i+ini])
        X_norm[:, i+ini]  = (X_norm[:, i+ini] - minimo)/(maximo - minimo)
    return X_norm

#======================= Funciones de costo, descenso del gradiente ===================================
def computeCost(X,y,theta):
    m = len(X)
    temp = np.power(((X * theta.T) - y), 2)
    return (1/(2 * m))*np.sum(temp) 
         
def gradientDescent(X, y, theta, alpha, iters, X2, y2):
    # x: (13000, 8)
    # y: (13000, 1)
    # theta: (1, 8)
    m = len(y)
    J_history = np.zeros(iters)
    J_history_cv = np.zeros(iters)
   
    for iter in np.arange(iters):
        h = X.dot(theta.T)  #  (13000, 8) * (8, 1) = (13000, 1)
        J_history[iter] = computeCost(X, y,theta)
        J_history_cv[iter] = computeCost(X2, y2, theta)

        for i in np.arange(X.shape[1]):
            theta[0, i] = theta[0, i] - (alpha/m)*np.sum((h-y).T*X[ :, i]) # (13000, 1) * (13000, 1)
    return(theta, J_history, J_history_cv)

def normalEqn(X, y):
    return np.linalg.pinv(X.T@X) @ X.T @ y

#=========================================== Pruebas ============================================================
def evaluarRendimiento(y,y_pred):
    cont=0
    error = []
    for i in range(len(y)): 
        if (y[i][0] == y_pred[i][0]):
            cont+=1
        else:
            error.append([i , y_pred[i][0]])
    return cont,error

#========================================================================================#
#                                   DATA TRAINING                                        #
#========================================================================================#

#===== Cargando valores de Entrenamiento =====
datatraining, X, y, theta = readDatabase('\df_train.csv')       
# Normalizando los valores de entrenamiento 
print("==== Valores de X sin Normalizar ===")
print(X[0:3, 0:8])
X = feautureNormalize(X, 4)
print("==== Valores de X normalizados ====")
print(X[0:3, 0:8])

#===== Cargando valores de la Validación Cruzada =====
datacv2, X2, y2, theta2 = readDatabase('\df_cv.csv') 
# Normalizando los valores de validación cruzada  
X2 = feautureNormalize(X2, 4)


#=============== Entrenamiento ====================
print(50*"=")
print(" Entrenando....\n")
alpha = 0.01
iters = 10000
theta, cost, cost_cv = gradientDescent(X, y, theta, alpha, iters, X2, y2) 

J = computeCost(X, y, theta)
J_cv = computeCost(X2, y2, theta)
print("Costo de entrenamiento: ", J)
print("Costo de validación cruzada: ", J_cv)
print("\nTheta calculada por el descenso del gradiente: \n\n", theta)

# ====== Evaluación =======
print(50*"=")
print(" Graficando el costo....")
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, label="Train")
ax.plot(np.arange(iters), cost_cv, label="Cross Validation")
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
ax.legend()
plt.show()

# ===== Evaluación usando sklearn ========
print(50*"=")
print("==== Evaluación usando sklearn ====\n")

y = np.asarray(y)                       # Entrenamiento
y2 = np.asarray(y2)                     # Validación Cruzada

y_pred = np.asarray(X.dot(theta.T))     # Predicción Entrenamiento
y_pred2 = np.asarray(X2.dot(theta.T) )  # Predicción Validación Cruzada

from sklearn.metrics import r2_score
def print_r2_score(y_train, train_pred, nombre):
    r2_train = r2_score(y_train, train_pred)
    print("Predicción de la regresion lineal (" + nombre + "): "+str(round(100*r2_train, 4))+" %")

print_r2_score(y, y_pred, 'Entrenamiento')
print_r2_score(y2, y_pred2, 'Validación Cruzada')