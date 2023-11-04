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

# MAGIC %md
# MAGIC #### Analisis Visual de los datos utilizando fundamentos estadisticos

# COMMAND ----------

create_report(df_datamart).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analisis Univariado y Bivariado de los datos

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

# MAGIC %md
# MAGIC #### Tratamiento de Datos Nulos

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

# MAGIC %md
# MAGIC #### Tratamiento de Outliers

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

# MAGIC %md
# MAGIC #### Re escalamiento de datos

# COMMAND ----------

# Variables a ser reescaladas con MinMax Encoder
features_mimmax = ['CANT_TIPO_PRODUCTO_1', 'CANT_TIPO_PRODUCTO_2', 'CANT_TIPO_PRODUCTO_3', 'CANT_TIPO_PRODUCTO_4',
                                       'CANT_TIPO_PRODUCTO_5', 'CANT_TIPO_PRODUCTO_6', 'PRECIO_TOTAL']

objeto_scaler = MinMaxScaler()
df_datamart[features_mimmax] = objeto_scaler.fit_transform(df_datamart[features_mimmax])
df_datamart.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tratamiento de variables cualitativas nominales y ordinales

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

electro_proccessed['OPERATIVO'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Desarrollo del Modelo con SPARK
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Hasta esta parte del codigo se ha realizado la limpìeza de los datos (utilizando pandas) generados de la integracion de diferentes dimensiones y una factica desde SQL (Datamart), generando finalmente una tabla minable
# MAGIC
# MAGIC Para la fase de modelado se buscar implementar 2 modelos:
# MAGIC - Un modelo de clustering para segmentar los proyectos en clusteres con mayor afinidad y asi asignarlos a los 3 Gerentes de proyectos, logrando asi que estos tengan un trabajo mas especializado y efciente con proyectos similares. 
# MAGIC - Un modelo de clasficacion binaria utilizando como target el flag 'OPERATIVO', ya que la informacion del datamart representa la estructura de proyectos, pero mucho de estos nunca llegan a ejecutarse, por lo que se realizara un submuestreo balanceando la data para los proyectos que si esten operativos, posteriormente se aplicara el modelo generada en la data restante.
# MAGIC

# COMMAND ----------

spark_datamart = spark.createDataFrame(electro_proccessed)
display(spark_datamart)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modelo con Kmeans

# COMMAND ----------

spark_datamart_cluster = spark_datamart.drop('OPERATIVO')


# COMMAND ----------

columns = ['NIVEL_TENSION',
 'CANT_TIPO_PRODUCTO_1',
 'CANT_TIPO_PRODUCTO_2',
 'CANT_TIPO_PRODUCTO_3',
 'CANT_TIPO_PRODUCTO_4',
 'CANT_TIPO_PRODUCTO_5',
 'CANT_TIPO_PRODUCTO_6',
 'PRECIO_TOTAL',
 'ESTADO_SOLICITUD_CARPETAARMADA',
 'ESTADO_SOLICITUD_CARPETACONGESTOR',
 'ESTADO_SOLICITUD_CERRARSOLICITUD',
 'ESTADO_SOLICITUD_CIERRECONTABLE',
 'ESTADO_SOLICITUD_CONCLUSIONOBRA',
 'ESTADO_SOLICITUD_CONTTAASIGNADO',
 'ESTADO_SOLICITUD_EJECMATMOAPROBADO',
 'ESTADO_SOLICITUD_EJECMATMOOBSERV',
 'ESTADO_SOLICITUD_ENACTIVACION',
 'ESTADO_SOLICITUD_ENCONSTRUCCION',
 'ESTADO_SOLICITUD_ENTREGAASBUILT',
 'ESTADO_SOLICITUD_INFCONSTAPROBADO',
 'ESTADO_SOLICITUD_REVCONTROLADM',
 'ESTADO_SOLICITUD_REVINFCONSTRUCCION',
 'ESTADO_SOLICITUD_SOLICITUDCERRADA',
 'ESTADO_SOLICITUD_SOLREVEJECMATMO',
 'SISTEMA_Rural',
 'SISTEMA_Tropico',
 'SISTEMA_Urbano',
 'SISTEMA_ValleAlto',
 'SISTEMA_ValleBajo',
 'SISTEMA_ValleCentral',
 'PROPIEDAD_Alcaldia',
 'PROPIEDAD_Cliente',
 'PROPIEDAD_Electro',
 'PROPIEDAD_Gobernacion',
 'PROPIEDAD_INSTALACION_Electro',
 'PROPIEDAD_INSTALACION_Exclusivo',
 'PROPIEDAD_INSTALACION_Interesado',
 'PROYECTO_ALUMBRADO PUBLICO',
 'PROYECTO_AMPLIACION',
 'PROYECTO_AMPLIACION SUBTERRANEA',
 'PROYECTO_ANEXADO',
 'PROYECTO_EQUIPOS',
 'PROYECTO_MANTENIMIENTO BAJA TENSION',
 'PROYECTO_MANTENIMIENTO MEDIA TENSION',
 'PROYECTO_REFORMA MANTENIMIENTO',
 'PROYECTO_REFORMA REDES DISTRIBUCION',
 'PROYECTO_SUBESTACION MT',
 'TIPO_PROYECTO_GAS',
 'TIPO_PROYECTO_INV',
 'TIPO_SERVICIO_AP',
 'TIPO_SERVICIO_EH',
 'TIPO_SERVICIO_EV',
 'TIPO_SERVICIO_PS',
 'TIPO_SERVICIO_RE']

assembler = VectorAssembler(inputCols=columns, outputCol="features")
df_cluster = assembler.transform(spark_datamart_cluster)

# COMMAND ----------

train_data, test_data = df_cluster.randomSplit([0.7,0.3])

# COMMAND ----------

kmeans = KMeans().setK(3).setSeed(2023)
model = kmeans.fit(train_data)

# COMMAND ----------

predictions = model.transform(test_data)

# COMMAND ----------

predictions_pandas = predictions.sample(fraction=0.5).toPandas()
predictions_pandas.head()

# COMMAND ----------

predictions.groupBy(F.col('prediction')).count().show()

# COMMAND ----------

evaluador = ClusteringEvaluator()

# COMMAND ----------

silhouette = evaluador.evaluate(predictions)
print("El coeficiente Silhouette usando distancias euclidianas al cuadrado es = " + str(silhouette))

# COMMAND ----------

# MAGIC %md
# MAGIC Se puede observar que se han generado 3 clusteres (Cluster 0 con 183 item, Cluster 1 con 1229 proyectos, Cluster 2 con 597 proyectos), no se realizo una evaluacion inicial para la cantidad de clusteres dado que la empresa unicamente cuenta con 3 gerentes de proyectos, sin embargo se puede observar que los proyectos se encuentran desbalanceados entre los gerentes, por lo que sera recomendable agregar un 4 gerente y generar nuevamente el modelo con 4 clusteres.

# COMMAND ----------


