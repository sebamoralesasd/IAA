from pyclbr import Class
import pandas as pd
import numpy as np
from math import *

from requests import head

class GeneradorDF:
  def __init__(self, std, d, n, centro1, centro2) -> None:
      self.std = std
      self.d = d
      self.n = n
      self.centro1 = centro1
      self.centro2 = centro2

  def generar_puntos(self, centro, mat_cov, size):
    rng = np.random.default_rng()
    return rng.multivariate_normal(centro, mat_cov, size).tolist()

  def taggear(self, lista, val):
    iter_list_tag = map(lambda x: x + [val], lista)
    # Por alguna razon, se sobreescribe list en algun lado y tengo que usar el builtin por defecto.
    return list(iter_list_tag)

  def generar_clases(self):
    cov = self.std ** 2
    mat_cov = np.diag(self.d * [cov])
    size = self.n + 1

    clase1 = self.generar_puntos(self.centro1, mat_cov, size // 2)
    clase1_contag = self.taggear(clase1, 0)

    clase2 = self.generar_puntos(self.centro2, mat_cov, size // 2)
    clase2_contag = self.taggear(clase2, 1)
    lista = clase1_contag + clase2_contag

    colNames = list(range(self.d)) + ["Clase"]

    return pd.DataFrame(lista, columns=colNames)


class GeneradorDFDiagonal:
  def __init__(self, C, d, n) -> None:
      self.C = C
      self.d = d
      self.n = n

  def generar_centro(self, val):
    return np.repeat(val, self.d)

  def generar_clase(self):
    centro1 = self.generar_centro(1)
    centro2 = self.generar_centro(-1)
    # desviación estándar igual a C * SQRT(d)
    generador_df = GeneradorDF(std=self.C*sqrt(self.d), n=self.n, d=self.d, centro1=centro1, centro2=centro2)
    return generador_df.generar_clases()

class GeneradorDFParalelo:
  def __init__(self, C, d, n) -> None:
      self.C = C
      self.d = d
      self.n = n

  def generar_centro(self, val):
    return np.append([val], np.repeat(0, self.d-1))

  def generar_clase(self):
    centro1b = self.generar_centro(1)
    centro2b = self.generar_centro(-1)
    # desviación estandar es igual a C independientemente de d
    generador_df = GeneradorDF(std=self.C, n=self.n, d=self.d, centro1=centro1b, centro2=centro2b)  
    return generador_df.generar_clases()  



import matplotlib.pyplot as plt

class GraficadorDF:
  def __init__(self, df) -> None:
      self.df = df

  # Fuente: https://stackoverflow.com/a/63539077
  def graph_puntos(self, titulo = ''):
    # Separo los puntos por clase
    x0, y0 = self.df[0][self.df.Clase == 0], self.df[1][self.df.Clase == 0]
    x1, y1 = self.df[0][self.df.Clase == 1], self.df[1][self.df.Clase == 1]

    # Calculo los máximos y mínimos para tener límites en x e y del gráfico simétricos
    xmax, xmin = max(max(x0), max(x1)), min(min(x0), min(x1))
    ymax, ymin = max(max(y0), max(y1)), min(min(y0), min(y1))

    xmax = np.ceil(max(xmax, ymax))
    ymax = xmax
    xmin, ymin = xmax * -1, ymax * -1

    # Grafico
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(x0, y0, c = 'blue', label = "Clase 0")
    ax.scatter(x1, y1, c = 'red', label = "Clase 1")

    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')

    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    x_ticks = np.arange(xmin, xmax+1, 1)
    y_ticks = np.arange(ymin, ymax+1, 1)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])
  
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    ax.legend()

    if titulo != '':
      plt.title(titulo, size=14, pad=25)
    plt.show()



from random import uniform
from math import dist
from cmath import polar

class GeneradorDFEspiral:
  def __init__(self, n, radio) -> None:
      self.n = n
      self.radio = radio

  def curva_uno(self, theta):
    return theta/(4*np.pi)

  def curva_dos(self, theta):
    return (theta + np.pi)/(4*np.pi)

  def punto_uniforme(self):    
      x, y = uniform(-self.radio, self.radio), uniform(-self.radio, self.radio)

      if dist([0, 0], [x, y]) <= self.radio:
        return [x, y] 
      else: # Se regenera el punto en caso de obtener uno fuera del circulo.
        return self.punto_uniforme() 

  def generar_puntos_curva(self):
    puntos = [self.punto_uniforme() for _ in range(self.n)]

    for i, (x, y) in enumerate(puntos):
      r, theta = polar(complex(x, y))

      es_clase_cero = self.curva_uno(theta) < r < self.curva_dos(theta) \
        or self.curva_uno(theta) + 0.5 < r < self.curva_dos(theta) + 0.5 \
        or self.curva_uno(theta) + 1 < r < self.curva_dos(theta) + 1
      
      if es_clase_cero: 
        puntos[i].append(0) 
      else: 
        puntos[i].append(1)

    colNames = [0, 1, "Clase"] 
    return pd.DataFrame(puntos, columns=colNames)


class GeneradorDFCSV:
  def __init__(self, filename, names=None, custom=False, delim_whitespace=False, skipinitialspace=False, header=None) -> None:
    if custom:
      self.df_data = self.read_csv_custom(filename, "data", names, delim_whitespace, skipinitialspace, header)
      self.df_test = self.read_csv_custom(filename, "test", names, delim_whitespace, skipinitialspace, header)
    else:
      self.df_data = self.read_csv(filename, "data")
      self.df_test = self.read_csv(filename, "test")
  
  def read_csv(self, name, df_type):
    return pd.read_csv(f"./datasets/{name}.{df_type}", names=[0, 1, "Clase"])

  def read_csv_custom(self, filename, df_type, names, delim_whitespace, skipinitialspace, header):
    return pd.read_csv(f"./datasets/{filename}.{df_type}", names=names, 
    delim_whitespace=delim_whitespace, skipinitialspace=skipinitialspace, header=header)