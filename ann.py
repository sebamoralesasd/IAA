from copy import deepcopy
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.base import is_classifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)

# params
# numero de epocas que entrena cada vez (epocas_por_entrenamiento)
# learning rate (eta)
# momentum (alfa)
# neuronas en la capa oculta (N2)
class MLP:
  def __init__(self, epocas_por_entrenamiento=25, eta=0.01, alfa=0.9, N2=60) -> None:        
    # defino MLP para regresión
    self._regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)
    # defino MLP para clasificación
    self._clasif = MLPClassifier(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=0.0, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)
    # print(self.regr)
  
  @property
  def regr(self):
    return self._regr

  @property
  def clasif(self):
    return self._clasif

class MLPGamma:
  def __init__(self, gamma, epocas_por_entrenamiento=25, eta=0.01, alfa=0.9, N2=60) -> None:
    self._regr = MLPRegressor(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=gamma, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)
    self._clasif = MLPClassifier(hidden_layer_sizes=(N2,), activation='logistic', solver='sgd', alpha=gamma, batch_size=1, learning_rate='constant', learning_rate_init=eta,momentum=alfa,nesterovs_momentum=False,tol=0.0,warm_start=True,max_iter=epocas_por_entrenamiento)


  @property
  def regr(self):
    return self._regr

  @property
  def clasif(self):
    return self._clasif


def medir_error(df: pd.DataFrame, df_entrenado: pd.DataFrame, red):
  if is_classifier(red):
    return sk.metrics.zero_one_loss(df, df_entrenado)
  else:
    return sk.metrics.mean_squared_error(df, df_entrenado)
  
# función que entrena una red ya definida previamente "evaluaciones" veces, cada vez entrenando un número de épocas 
# elegido al crear la red y midiendo el error en train, validación y test al terminar ese paso de entrenamiento. 
# Guarda y devuelve la red en el paso de evaluación que da el mínimo error de validación
# entradas: la red, las veces que evalua, los datos de entrenamiento y sus respuestas, de validacion y sus respuestas, de test y sus respuestas
# salidas: la red entrenada en el mínimo de validación, los errores de train, validación y test medidos en cada evaluación
def entrenar_red(red, evaluaciones, X_train, y_train, X_val, y_val, X_test, y_test):
  error_train_data = []
  error_val_data = []
  error_test_data = []

  best_error_val = 0

  for e in range(evaluaciones):
    red.fit(X_train, y_train)

    train_entrenado = red.predict(X_train)
    val_entrenado = red.predict(X_val)
    test_entrenado = red.predict(X_test)

    current_error_train = medir_error(y_train, train_entrenado, red)
    current_error_val = medir_error(y_val, val_entrenado, red)
    current_error_test = medir_error(y_test, test_entrenado, red)

    error_train_data.append(current_error_train)
    error_val_data.append(current_error_val)
    error_test_data.append(current_error_test)

    if e == 0 or current_error_val < best_error_val:
      best_error_val = current_error_val
      #best_red = deepcopy(red)

  return test_entrenado, error_train_data, error_val_data, error_test_data

# función que grafica curvas de error y las predicciones
# epocas es la cantidad de veces que entreno la red y mido los errores
def graficar_curvas(regr, epocas, X_train, y_train, X_val, y_val, X_test, y_test):
  regr, e_train, e_val, e_test = entrenar_red(regr, epocas, X_train, y_train, X_val, y_val, X_test, y_test)

  plt.plot(range(epocas),e_train,label="train",linestyle=":")
  plt.plot(range(epocas),e_val,label="validacion",linestyle="-.")
  plt.plot(range(epocas),e_test,label="test",linestyle="-")

  plt.legend()
  plt.show()