# Databricks notebook source
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from dataprep.eda import create_report
from sklearn import preprocessing 
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.sql import functions as F

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre procesamiento de datos con Pandas
# MAGIC Inicialmente se realizara el pre procesamiento de datos utilizandos pandas en el archivo Datamart_Proyecto generado del query de integracion, posteriomente para la generacion del modelo se utilizara Spark
# MAGIC

# COMMAND ----------

df_electro = spark.read.csv('/FileStore/tables/CasoElectro/DATAMART_PROYECTO.csv', header=True, inferSchema=True)
df_datamart = df_electro.toPandas()
df_datamart.head()

# COMMAND ----------

df_datamart.dtypes 

# COMMAND ----------

# Eliminacion de columnas no necesarias
df_datamart = df_datamart.drop(['ID_PRO'], axis=1)

# COMMAND ----------

# Columnas a Object
df_datamart["ID_PROVEEDOR"] = df_datamart["ID_PROVEEDOR"].astype('object')
df_datamart["ID_PLAN_INVERSION"] = df_datamart["ID_PLAN_INVERSION"].astype('object')
df_datamart.describe()

# COMMAND ----------

df_datamart.dtypes 

# COMMAND ----------

#Tamaño del Set de datos para entrenamiento y test
len(df_datamart)

# COMMAND ----------

create_report(df_datamart).show()

# COMMAND ----------

# Analisis Bivariado
df_datamart.groupby(['OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       }).sort_values(['OPERATIVO'], ascending = True )  

# COMMAND ----------

df_datamart.groupby(['ESTADO_SOLICITUD','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['SISTEMA','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['NIVEL_TENSION','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['PROPIEDAD','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['PROPIEDAD_INSTALACION','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['PROYECTO','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['TIPO_PROYECTO','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean,  'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.groupby(['TIPO_SERVICIO','OPERATIVO']).agg({'CANT_TIPO_PRODUCTO_1':np.mean, 'CANT_TIPO_PRODUCTO_2':np.mean, 'CANT_TIPO_PRODUCTO_3':np.mean, 'CANT_TIPO_PRODUCTO_4':np.mean, \
                                                      'CANT_TIPO_PRODUCTO_5':np.mean, 'CANT_TIPO_PRODUCTO_6':np.mean, 'PRECIO_TOTAL':np.mean
                                       })

# COMMAND ----------

df_datamart.columns

# COMMAND ----------

#Tratamiento de Datos Nulos
df_datamart.isnull().sum()

# COMMAND ----------

sns.heatmap(df_datamart.isnull(), cbar=False)

# COMMAND ----------

# Verificamos el tamaño inicial
df_datamart.shape

# COMMAND ----------

# Eliminamos las filas con valor nulo en el precio total
columns_to_check = ['PRECIO_TOTAL']
df_datamart.dropna(subset=columns_to_check, how='any', inplace=True)

# COMMAND ----------

# Rellenamos el valor nulo de la columna Tipo Proyecto con la moda
most_frequent_value = df_datamart['TIPO_PROYECTO'].mode()[0] 
df_datamart['TIPO_PROYECTO'].fillna(most_frequent_value, inplace=True)

# COMMAND ----------

df_datamart.isnull().sum()

# COMMAND ----------

# Verificamos el tamaño
df_datamart.shape

# COMMAND ----------

# Tratamiento de Datos Atipicos con Z-Score
z = np.abs(stats.zscore(df_datamart[['CANT_TIPO_PRODUCTO_1', 'CANT_TIPO_PRODUCTO_2', 'CANT_TIPO_PRODUCTO_3', 'CANT_TIPO_PRODUCTO_4',
                                       'CANT_TIPO_PRODUCTO_5', 'CANT_TIPO_PRODUCTO_6', 'PRECIO_TOTAL']]))
z.head()

# COMMAND ----------

(z<3).all(axis=1)

# COMMAND ----------

# Reduccion de 7054 clientes a 6674
df_datamart = df_datamart[(z<3).all(axis=1)]
df_datamart.shape

# COMMAND ----------

# Variables a ser reescaladas con MinMax Encoder
features_mimmax = ['CANT_TIPO_PRODUCTO_1', 'CANT_TIPO_PRODUCTO_2', 'CANT_TIPO_PRODUCTO_3', 'CANT_TIPO_PRODUCTO_4',
                                       'CANT_TIPO_PRODUCTO_5', 'CANT_TIPO_PRODUCTO_6', 'PRECIO_TOTAL']

objeto_scaler = MinMaxScaler()
df_datamart[features_mimmax] = objeto_scaler.fit_transform(df_datamart[features_mimmax])
df_datamart.head()

# COMMAND ----------

# Variables cualitativas nominales a convertirse en Dummies
features = ['ESTADO_SOLICITUD', 'SISTEMA', 'PROPIEDAD', 'PROPIEDAD_INSTALACION'
            , 'PROYECTO', 'TIPO_PROYECTO', 'TIPO_SERVICIO']

dummies = pd.get_dummies(df_datamart[features])
electro_proccessed = pd.concat([df_datamart.drop(features, axis=1), dummies], axis=1)
electro_proccessed.head()

# COMMAND ----------

# Variables cualitativas ordinales a ser procesadas en ordinal encoder
enc = OrdinalEncoder()
electro_proccessed[["NIVEL_TENSION"]] = enc.fit_transform(electro_proccessed[["NIVEL_TENSION"]])
electro_proccessed.head()

# COMMAND ----------

# Modificacion de variable objetivo a enteros
electro_proccessed['OPERATIVO'] = electro_proccessed['OPERATIVO'].replace('FALSE', 0).replace('TRUE', 1)

# COMMAND ----------

#Tabla Minable
electro_proccessed.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ## A partir de aqui se utilizara spark para el modelo
# MAGIC

# COMMAND ----------


