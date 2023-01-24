# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Case Petz
# MAGIC 
# MAGIC Esse notebook contém a resolução do case de predição de demanda da Petz.
# MAGIC 
# MAGIC 
# MAGIC ## Sumário
# MAGIC 
# MAGIC 1. Bibliotecas
# MAGIC 2. Importação dos Dados
# MAGIC 3. Análise Exploratória
# MAGIC     1. Decomposição das séries temporais
# MAGIC     2. Correlação e Autocorrelação
# MAGIC 4. Feature Engineering
# MAGIC     1. Anos Abertos
# MAGIC     2. One hot encoding
# MAGIC 5. Modelagem
# MAGIC     1. Separação componentes da data
# MAGIC     2. Join nas tabelas
# MAGIC     3. Criação dos datasets de treino e teste
# MAGIC     4. Baseline: modelo ingênuo
# MAGIC     5. Regressão Linear
# MAGIC     6. Random Forest
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Bibliotecas

# COMMAND ----------

from pyspark.sql.functions import (
    col,
    isnan,
    when,
    count,
    regexp_replace,
    year,
    month,
    dayofmonth,
    dayofweek,
    dayofyear,
    weekofyear,
    concat,
    udf,
    percent_rank,
    pandas_udf,
    PandasUDFType,
    lag,
    lit
)
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Window

from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StringIndexer,
    OneHotEncoder,
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation

import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# from xgboost.spark import SparkXGBRegressor
from prophet import Prophet

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Dados

# COMMAND ----------

canais = spark.read.table('canais')
lojas = spark.read.table('lojas')
produtos = spark.read.table('produtos')
unidades_negocios = spark.read.table('unidades_negocios')
vendas = spark.read.table('vendas')

# COMMAND ----------

vendas = vendas.withColumn('qtde_venda', regexp_replace('qtde_venda', ',', '.'))
vendas = vendas.withColumn('qtde_venda', vendas['qtde_venda'].cast("float"))
vendas = vendas.withColumn('valor_venda', regexp_replace('valor_venda', ',', '.'))
vendas = vendas.withColumn('valor_venda', vendas['valor_venda'].cast("float"))
vendas = vendas.withColumn('valor_imposto', regexp_replace('valor_imposto', ',', '.'))
vendas = vendas.withColumn('valor_imposto', vendas['valor_imposto'].cast("float"))
vendas = vendas.withColumn('valor_custo', regexp_replace('valor_custo', ',', '.'))
vendas = vendas.withColumn('valor_custo', vendas['valor_custo'].cast("float"))
vendas = vendas.withColumn('id_data',vendas.id_data.cast(DateType()))

# COMMAND ----------

print(vendas.dropna().count())
print(vendas.count())
print(vendas.na.drop(how='any').count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Análise Exploratória

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decomposição das Séries Temporais
# MAGIC 
# MAGIC #### Vendas Gerais

# COMMAND ----------

vendas = vendas.withColumn('ano_mes', concat(year(vendas.id_data), F.lit('-'),month(vendas.id_data)))
vendas_por_mes = vendas.select('ano_mes', 'qtde_venda', 'valor_venda').groupBy('ano_mes').agg({'qtde_venda': 'sum', 'valor_venda': 'sum'}).toPandas()
vendas_por_mes['ano_mes'] = pd.to_datetime(vendas_por_mes['ano_mes'], format='%Y-%m') + MonthEnd(0)

# COMMAND ----------

ax = sns.lineplot(data=vendas_por_mes, x='ano_mes', y='sum(qtde_venda)')
ax.tick_params(axis='x', rotation=45)

# COMMAND ----------

result_add = seasonal_decompose(vendas_por_mes.set_index('ano_mes')['sum(qtde_venda)'].sort_index(), model='multiplicative')
figure = result_add.plot()
figure.set_figheight(10)
figure.set_figwidth(15)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vendas por loja

# COMMAND ----------

vendas_por_loja = vendas.select('ano_mes', 'id_loja', 'qtde_venda').groupBy('ano_mes', 'id_loja').sum().toPandas()
vendas_por_loja['ano_mes'] = pd.to_datetime(vendas_por_loja['ano_mes'], format='%Y-%m') + MonthEnd(0)
vendas_por_loja_merged = pd.merge(vendas_por_loja, lojas.toPandas(), how='inner', on='id_loja')

# COMMAND ----------

ax = sns.lineplot(data=vendas_por_loja_merged, x='ano_mes', y='sum(qtde_venda)', hue='cod_loja')
ax.tick_params(axis='x', rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlação e Autocorrelação

# COMMAND ----------

colunas_numericas = [ 'qtde_venda',  'valor_venda','valor_imposto','valor_custo']

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=colunas_numericas, outputCol=vector_col)
df_vector = assembler.transform(vendas).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = colunas_numericas, index=colunas_numericas) 
corr_matrix_df .style.background_gradient(cmap='coolwarm').set_precision(2)

# COMMAND ----------

vendas_por_dia = vendas.select('id_data', 'qtde_venda').groupBy('id_data').agg({'qtde_venda': 'sum'}).toPandas()

vendas_por_dia = vendas_por_dia.set_index('id_data').sort_index()

ax = pd.plotting.autocorrelation_plot(vendas_por_dia)

ax.set_xlim([0, 550])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anos Abertos

# COMMAND ----------

lojas = lojas.withColumn('anos_abertos', 2019 - col('ano_abertura'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### One hot encoding

# COMMAND ----------

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="cod_loja", outputCol="loja_index")
indexed_df = string_indexer.fit(lojas).transform(lojas)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="loja_index", outputCol="loja_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="regional", outputCol="regional_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="regional_index", outputCol="regional_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="distrito", outputCol="distrito_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="distrito_index", outputCol="distrito_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="cidade", outputCol="cidade_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="cidade_index", outputCol="cidade_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="uf", outputCol="uf_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="uf_index", outputCol="uf_vec")
encoded_lojas = one_hot_encoder.fit(indexed_df).transform(indexed_df)

# COMMAND ----------

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="fornecedor", outputCol="fornecedor_index")
indexed_df = string_indexer.fit(produtos).transform(produtos)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="fornecedor_index", outputCol="fornecedor_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

string_indexer = StringIndexer(inputCol="produto_nome", outputCol="produto_nome_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="produto_nome_index", outputCol="produto_nome_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

string_indexer = StringIndexer(inputCol="categoria", outputCol="categoria_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="categoria_index", outputCol="categoria_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

string_indexer = StringIndexer(inputCol="sub_categoria", outputCol="sub_categoria_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="sub_categoria_index", outputCol="sub_categoria_vec")
encoded_produtos = one_hot_encoder.fit(indexed_df).transform(indexed_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelagem

# COMMAND ----------

colunas = ['id_data', 'id_loja', 'id_produto', 'qtde_venda']

colunas_group_by = ['id_data', 'id_loja', 'id_produto']

colunas_agg = {'qtde_venda': 'sum'}

vendas_grouped = vendas.select(colunas).groupBy(colunas_group_by).agg(colunas_agg)

# COMMAND ----------

print(vendas.count())
print(vendas_grouped.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Separação ano, mês, dia, número da semana, número do dia na semana

# COMMAND ----------

vendas_grouped = vendas_grouped.withColumn('ano', year(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn('mes', month(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn('dia_do_mes', dayofmonth(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn('dia_da_semana', dayofweek(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn('dia_do_ano', dayofyear(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn('semana_do_ano', weekofyear(vendas_grouped.id_data))
vendas_grouped.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join nas tabelas

# COMMAND ----------

vendas_merged = vendas_grouped \
    .join(encoded_lojas, 'id_loja', how='inner') \
    .join(encoded_produtos, vendas_grouped.id_produto==encoded_produtos.produto, how='inner')

vendas_merged.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criação dos datasets de treino e teste.

# COMMAND ----------

selected_columns = [
    "ano",
    "mes",
    "dia_do_mes",
    "dia_da_semana",
    "dia_do_ano",
    "semana_do_ano",
    "anos_abertos",
    "loja_vec",
    "regional_vec",
    "distrito_vec",
    "cidade_vec",
    "uf_vec",
    "fornecedor_vec",
    "produto_nome_vec",
    "categoria_vec",
    "sub_categoria_vec"
]

target_column = ['sum(qtde_venda)']

vendas_selected = vendas_merged.select(selected_columns + target_column + ['id_data', 'id_loja', 'id_produto'])

# COMMAND ----------

# unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

df = vendas_selected.alias('df')

for i in selected_columns:
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i],outputCol=i+"_Vect")

    # MinMaxScaler Transformation
    scaler = StandardScaler(inputCol=i+"_Vect", outputCol=i+"_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    df = pipeline.fit(df).transform(df).drop(i+"_Vect")


# COMMAND ----------

# features_columns = [i+"_Scaled" for i in selected_columns]
features_columns = ['ano', 'mes', 'loja_vec', 'produto_nome_vec']
assembler = VectorAssembler(inputCols=features_columns, outputCol='features')
vendas_vectorized = assembler.transform(df)

# COMMAND ----------

train_test_data = vendas_vectorized.withColumn("rank", percent_rank().over(Window.partitionBy().orderBy("id_data")))

train = train_test_data.where("rank <= .8").drop("rank")
test = train_test_data.where("rank > .8").drop("rank")

# COMMAND ----------

print("Train antes: ", train.count(), train.rdd.getNumPartitions())
print("Test antes: ", test.count(), test.rdd.getNumPartitions())
train_partitions = train.repartition(15)
test_partitions = test.repartition(10)
print("Train depois: ", train_partitions.count(), train_partitions.rdd.getNumPartitions())
print("Test depois: ", test_partitions.count(), test_partitions.rdd.getNumPartitions())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline: modelo ingênuo
# MAGIC 
# MAGIC Para esse modelo, vamos pegar a média de quantidades vendidas dos produtos por loja e por dia-mês.

# COMMAND ----------

train_naive_fit = train_partitions.select('ano', 'mes', 'id_loja', 'id_produto', 'sum(qtde_venda)').groupBy('ano', 'mes', 'id_loja', 'id_produto').agg({'sum(qtde_venda)': 'mean'})

# COMMAND ----------

train_naive_predict = train_partitions.select('ano', 'mes', 'id_loja', 'id_produto', 'sum(qtde_venda)').join(train_naive_fit, ['ano', 'mes', 'id_loja', 'id_produto']).withColumnRenamed('avg(sum(qtde_venda))',"prediction")

# COMMAND ----------

test_naive_predict = test_partitions.select('ano', 'mes', 'id_loja', 'id_produto', 'sum(qtde_venda)').join(train_naive_fit, ['ano', 'mes', 'id_loja', 'id_produto']).withColumnRenamed('avg(sum(qtde_venda))',"prediction")

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="sum(qtde_venda)", predictionCol="prediction", metricName="rmse")

train_rmse = evaluator.evaluate(train_naive_predict)
test_rmse = evaluator.evaluate(test_naive_predict)
print(train_rmse)
print(test_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regressão Linear

# COMMAND ----------

lr_classifier = LinearRegression(featuresCol='features', labelCol='sum(qtde_venda)').fit(train_partitions)
lr_train_predictions = lr_classifier.transform(train_partitions)
lr_test_predictions = lr_classifier.transform(test_partitions)

evaluator = RegressionEvaluator(labelCol="sum(qtde_venda)", predictionCol="prediction", metricName="rmse")

lr_train_rmse = evaluator.evaluate(lr_train_predictions)
lr_test_rmse = evaluator.evaluate(lr_test_predictions)
print(lr_train_rmse)
print(lr_test_rmse)

# COMMAND ----------

window = Window.partitionBy('id_loja', 'id_produto').orderBy("id_data")

train_lagged = train_partitions.withColumn("lag",lag("sum(qtde_venda)",1).over(window))

# COMMAND ----------

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="id_loja", outputCol="id_loja_idx")
indexed_df = string_indexer.fit(train_partitions).transform(train_partitions)

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="id_produto", outputCol="id_produto_idx")
indexed_df = string_indexer.fit(indexed_df).transform(indexed_df)

# COMMAND ----------

schema = StructType([
                     StructField('id_loja_idx', FloatType()),
                     StructField('id_produto_idx', FloatType()),
                     StructField('ds', TimestampType()),
                     StructField('y', FloatType()),
                     StructField('yhat', DoubleType()),
                     StructField('yhat_upper', DoubleType()),
                     StructField('yhat_lower', DoubleType()),
])

# COMMAND ----------

# define the Pandas UDF
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def apply_model(store_pd):
    # instantiate the model and set parameters
    model = Prophet(
        interval_width=0.95,
        growth="linear",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
    )
    # fit the model to historical data
    model.fit(store_pd)
    # Create a data frame that lists 90 dates starting from Jan 1 2018
    future = model.make_future_dataframe(periods=90, freq="d", include_history=True)
    # Out of sample prediction
    future = model.predict(future)
     # Create a data frame that contains store, item, y, and yhat
    f_pd = future[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]
    f_pd['ds'] = pd.to_datetime(f_pd['ds'])
    st_pd = store_pd[['ds', 'id_loja_idx', 'id_produto_idx', 'y']]
    st_pd['ds'] = pd.to_datetime(st_pd['ds'])
    result_pd = f_pd.join(st_pd.set_index('ds'), on='ds', how='left')
    # fill store and item
    result_pd['id_loja_idx'] = store_pd['id_loja_idx'].iloc[0]
    result_pd['id_produto_idx'] = store_pd['id_produto_idx'].iloc[0]
    return result_pd[['id_loja_idx', 'id_produto_idx', 'ds', 'y', 'yhat',
                    'yhat_upper', 'yhat_lower']]


# Apply the function to all store-items
results = (
    indexed_df.select("id_data", "sum(qtde_venda)", 'id_loja_idx', 'id_produto_idx')
    .withColumnRenamed("id_data", "ds")
    .withColumnRenamed("sum(qtde_venda)", "y")
    .groupBy('id_loja_idx', 'id_produto_idx')
    .apply(apply_model)
)

# COMMAND ----------

indexed_df.select('id_data', 'sum(qtde_venda)').orderBy(col('id_data').desc()).show()

# COMMAND ----------

results.orderBy(col('ds').desc()).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

rf_classifier = RandomForestRegressor(featuresCol='features', labelCol='sum(qtde_venda)', numTrees=5).fit(train_partitions)

# COMMAND ----------

train_predictions = rf_classifier.transform(train_partitions)
test_predictions = rf_classifier.transform(test_partitions)

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="sum(qtde_venda)", predictionCol="prediction", metricName="rmse")

train_rmse = evaluator.evaluate(train_predictions)
test_rmse = evaluator.evaluate(test_predictions)
print(train_rmse)
print(test_rmse)
