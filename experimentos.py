# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Case Petz
# MAGIC 
# MAGIC ## Bibliotecas

# COMMAND ----------

from pyspark.sql.functions import col,isnan, when, count, regexp_replace, year, month, dayofmonth, dayofweek, dayofyear, weekofyear, concat
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

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

result_add = seasonal_decompose(vendas_por_mes.set_index('ano_mes')['sum(valor_venda)'].sort_index())
figure = result_add.plot()
figure.set_figheight(8)
figure.set_figwidth(15)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vendas por loja

# COMMAND ----------

vendas_por_loja = vendas.select('ano_mes', 'id_loja', 'qtde_venda').groupBy('ano_mes', 'id_loja').sum().toPandas()
vendas_por_loja['ano_mes'] = pd.to_datetime(vendas_por_loja['ano_mes'], format='%Y-%m') + MonthEnd(0)

# COMMAND ----------

vendas_por_loja_merged = pd.merge(vendas_por_loja, lojas.toPandas(), how='inner', on='id_loja')

# COMMAND ----------

vendas_por_loja_merged.head()

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

lojas.show()

# COMMAND ----------

lojas = lojas.withColumn('anos_abertos', 2019 - col('ano_abertura')).show(1)

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

encoded_lojas.show()

# COMMAND ----------

# Create a StringIndexer to convert the categorical feature to an indexed column
string_indexer = StringIndexer(inputCol="fornecedor", outputCol="fornecedor_index")
indexed_df = string_indexer.fit(produtos).transform(produtos)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="fornecedor_index", outputCol="fornecedor_vec")
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

encoded_produtos.show()

# COMMAND ----------

vendas.show()

# COMMAND ----------

vendas.groupBy('id_tipo_cliente').count().orderBy(col('count').desc()).show(100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pré-processamento

# COMMAND ----------

#colunas = ['id_data', 'id_loja', 'id_produto', 'qtde_venda', 'valor_venda', 'valor_imposto', 'valor_custo', 'ano', 'mes', 'dia_do_mes', 'dia_da_semana', 'dia_do_ano', 'semana_do_ano']
colunas = ['id_data', 'id_loja', 'id_produto', 'qtde_venda', 'valor_venda', 'valor_imposto', 'valor_custo']

colunas_group_by = ['id_data', 'id_loja', 'id_produto']

colunas_sum = ['qtde_venda', 'valor_venda', 'valor_imposto', 'valor_custo']

colunas_unique = ['ano', 'mes', 'dia_do_mes', 'dia_da_semana', 'dia_do_ano', 'semana_do_ano']

vendas.select(colunas).groupBy(colunas_group_by).agg().show()

# COMMAND ----------

vendas.createOrReplaceTempView("vendas_temp")

sql_str="select id_data, id_loja, id_produto," \
"sum(qtde_venda)," \
"sum(valor_venda)," \
"sum(valor_imposto)," \
"sum(valor_custo)" \
" from vendas_temp "  \
" group by id_data, id_loja, id_produto"

vendas_grouped = spark.sql(sql_str)

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
vendas_grouped.show() # diminui tamanho pela metade

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join nas tabelas

# COMMAND ----------

vendas_merged = vendas_grouped \
    .join(encoded_lojas, 'id_loja', how='inner') \
    .join(encoded_produtos, vendas_grouped.id_produto==encoded_produtos.produto, how='inner')

# COMMAND ----------

vendas_merged.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criação dos datasets de treino e teste

# COMMAND ----------

from pyspark.sql.functions import percent_rank
from pyspark.sql import Window

# Criando algo parecido com TimeSeriesSplit do sci-kit learn
folds = []

# Define the number of splits you want
n_splits = 5
  
# Calculate count of each dataframe rows
each_len = vendas_df.count() // n_splits
  
# Create a copy of original dataframe
copy_df = prod_df
  
# Iterate for each dataframe
i = 0
while i < n_splits:
  
    # Get the top `each_len` number of rows
    temp_df = copy_df.limit(each_len)
  
    # Truncate the `copy_df` to remove
    # the contents fetched for `temp_df`
    copy_df = copy_df.subtract(temp_df)
    
    temp_df.append(folds)
  
    # Increment the split number
    i += 1

res = []
for fold in folds:
    train, test = fold.randomSplit([0.80,0.20])
    model.train(train)
    res.append(model.evaluate(test))

# COMMAND ----------

vendas_grouped.show()
