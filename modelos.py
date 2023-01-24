# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC <img src="https://s3.amazonaws.com/gupy5/production/companies/1130/career/1714/images/2022-07-29_17-08_mainImage.jpg">
# MAGIC 
# MAGIC # Case Petz
# MAGIC 
# MAGIC Olá! Seja muito bem vindo à minha resolução do case da Petz. O objetivo desse estudo é criar previsões de demanda para os próximos 30, 60 e 90 dias (a partir da data final do dataset).
# MAGIC 
# MAGIC Para isso, iremos utilizar diversas ferramentas e técnicas diferentes. Iniciamos com a importação das biliotecas e dos dados. Em seguida analisamos os dados para capturar tendências e features importantes. Com isso em mente criamos novas features e, finalmente, criamos os modelos de predição.
# MAGIC 
# MAGIC 
# MAGIC ## Sumário
# MAGIC 
# MAGIC 1. Bibliotecas.
# MAGIC 2. Importação dos Dados.
# MAGIC 3. Análise Exploratória.
# MAGIC     1. Decomposição das séries temporais.
# MAGIC     2. Correlação e Autocorrelação.
# MAGIC 4. Feature Engineering.
# MAGIC     1. Anos Abertos.
# MAGIC     2. One hot encoding.
# MAGIC 5. Modelagem.
# MAGIC     1. Separação componentes da data.
# MAGIC     2. Join nas tabelas.
# MAGIC     3. Criação dos datasets de treino e teste.
# MAGIC     4. Baseline: modelo simples.
# MAGIC     5. Regressão Linear.
# MAGIC     6. Random Forest.
# MAGIC 6. Predição próximos 90 dias.
# MAGIC 7. Considerações Finais.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Bibliotecas
# MAGIC 
# MAGIC As tecnologias usadas nesse estudo foram: `pyspark`, `pandas`, `matplotlib`, `seaborn`, `statsmodels` e o `prophet`.

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
    lit,
    sequence,
    to_date,
    explode,
    col,
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

canais = spark.read.table("canais")
lojas = spark.read.table("lojas")
produtos = spark.read.table("produtos")
unidades_negocios = spark.read.table("unidades_negocios")
vendas = spark.read.table("vendas")

# COMMAND ----------

# MAGIC %md
# MAGIC Antes de prosseguirmos com a análise, devemos fazer um breve tratamento dos dados da tabela de vendas. As colunas `qtde_venda`, `valor_venda`, `valor_imposto`, `valor_custo` tem como separador da casa decimal o caracter ",". Substituiremos por "." e transformaremos a coluna de `StringType` para `FloatType`.
# MAGIC 
# MAGIC Por fim, a coluna `id_data`  deve ser transformada para `DataType`.

# COMMAND ----------

vendas = vendas.withColumn("qtde_venda", regexp_replace("qtde_venda", ",", "."))
vendas = vendas.withColumn("qtde_venda", vendas["qtde_venda"].cast("float"))
vendas = vendas.withColumn("valor_venda", regexp_replace("valor_venda", ",", "."))
vendas = vendas.withColumn("valor_venda", vendas["valor_venda"].cast("float"))
vendas = vendas.withColumn("valor_imposto", regexp_replace("valor_imposto", ",", "."))
vendas = vendas.withColumn("valor_imposto", vendas["valor_imposto"].cast("float"))
vendas = vendas.withColumn("valor_custo", regexp_replace("valor_custo", ",", "."))
vendas = vendas.withColumn("valor_custo", vendas["valor_custo"].cast("float"))
vendas = vendas.withColumn("id_data", vendas.id_data.cast(DateType()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Checamos para dados faltantes no nosso dataset principal (`vendas`). Note que o tamanho do dataset não muda após remover observações com dados faltantes.

# COMMAND ----------

print("Dataset completo: ", vendas.count())
print("Dataset sem NaN: ", vendas.na.drop(how="any").count())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Análise Exploratória

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decomposição das Séries Temporais
# MAGIC 
# MAGIC Começamos a análise exploratória dos dados decompondo a série temporal de interesse (`qtde_venda`) pela tendência, sazonalidade e ruído. Essa análise é importante por uma série de motivos. Tome, por exemplo, o modelo de regressão linear, que perfoma melhor quando a série é estacionária, i.e., não há tendência positiva ou negativa.
# MAGIC Não obstante, a sazonalidade dá mais informações para o modelo e melhora sua robustez de predição, ao passo que certas observações podem ser mais facilmente previstas se obedecerem à alguma regra de sazonalidade. Um exemplo claro disso é o aumento de vendas que ocorre no setor de varejo no fim de ano.
# MAGIC 
# MAGIC Iniciaremos observando o comportamento da série por si só.
# MAGIC 
# MAGIC #### Vendas Gerais

# COMMAND ----------

vendas = vendas.withColumn(
    "ano_mes", concat(year(vendas.id_data), F.lit("-"), month(vendas.id_data))
)
vendas_por_mes = (
    vendas.select("ano_mes", "qtde_venda", "valor_venda")
    .groupBy("ano_mes")
    .agg({"qtde_venda": "sum", "valor_venda": "sum"})
    .toPandas()
)
vendas_por_mes["ano_mes"] = pd.to_datetime(
    vendas_por_mes["ano_mes"], format="%Y-%m"
) + MonthEnd(0)

# COMMAND ----------

ax = sns.lineplot(data=vendas_por_mes, x="ano_mes", y="sum(qtde_venda)")
ax.tick_params(axis="x", rotation=45)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Ao olho "nu", observamos que há uma tendência positiva. Confirmamos isso ao decompor a série. Vide gráfico abaixo.

# COMMAND ----------

result_add = seasonal_decompose(
    vendas_por_mes.set_index("ano_mes")["sum(qtde_venda)"].sort_index(),
    model="multiplicative",
)
figure = result_add.plot()
figure.set_figheight(10)
figure.set_figwidth(15)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Confirmando nossa hipótese inicial, há tendência positiva na série entre os meses de 2018-08 e 2019-07. Nota-se, além disso, alguns compostamentos sazonais. A cada 12 meses, há um aumento das vendas em janeiro seguido de uma queda nos meses de fevereiro e março. O aumento das vendas pode ser desencadeado pelos feriados de fim de ano e pela época de recebimento do 13º.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Vendas por loja
# MAGIC 
# MAGIC Outro fator possivelmente muito importante de ser capturado pelos modelos é o efeito das diferentes lojas. Abaixos iremos decompor a série original para cada tipo de loja.

# COMMAND ----------

vendas_por_loja = (
    vendas.select("ano_mes", "id_loja", "qtde_venda")
    .groupBy("ano_mes", "id_loja")
    .sum()
    .toPandas()
)
vendas_por_loja["ano_mes"] = pd.to_datetime(
    vendas_por_loja["ano_mes"], format="%Y-%m"
) + MonthEnd(0)
vendas_por_loja_merged = pd.merge(
    vendas_por_loja, lojas.toPandas(), how="inner", on="id_loja"
)

# COMMAND ----------

ax = sns.lineplot(
    data=vendas_por_loja_merged, x="ano_mes", y="sum(qtde_venda)", hue="cod_loja"
)
ax.tick_params(axis="x", rotation=45)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Esse gráfico nos dá algumas informações:
# MAGIC 
# MAGIC 1) Nem todas as lojas tem a mesma quantidade de demanda. Nota-se que a Loja 3 vende mais, enquanto a Loja 2 vende menos. E as demais gravitam no meio dessas duas.
# MAGIC 2) Nem todas as lojas foram criadas no mesmo ano. Note que a Loja 5 surje no ano de 2018, enquanto as demais já existiam anteriormente. Iremos capturar essa informação na seção de ***Feature Engineering***, criando a coluna de Anos Abertos.
# MAGIC 3) As lojas obedecem, em média, as variações de sazonalidade. Quando há uma tendência de aumento da demanda, todas aumentam. As explicações para isso são diversas, mas dentre elas uma possível é que existam variáveis exógenas ao modelo (data é um feriado, variações na taxa de desemprego).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlação e Autocorrelação
# MAGIC 
# MAGIC Iremos analisar as correlações entre algumas variáveis e a correlação da série de interesse com ela mesma. A primeira etapa é importante para removermos variáveis do modelo que possuem alta correlação entre si, evitando problemas de multicolinearidade. Esse tipo de problema afeta modelos como o de regressão linear. 
# MAGIC 
# MAGIC Não obstante, a autocorrelação irá servir para termos uma ideia geral de como as observações passadas impactam o valor atual. Isso é importante para criarmos novas features (`lags`) e também para melhor ajustar modelos como o ARIMA.

# COMMAND ----------

colunas_numericas = ["qtde_venda", "valor_venda", "valor_imposto", "valor_custo"]

# convert to vector column first
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=colunas_numericas, outputCol=vector_col)
df_vector = assembler.transform(vendas).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)
matrix = Correlation.corr(df_vector, "corr_features").collect()[0][0]
corr_matrix = matrix.toArray().tolist()
corr_matrix_df = pd.DataFrame(
    data=corr_matrix, columns=colunas_numericas, index=colunas_numericas
)
corr_matrix_df.style.background_gradient(cmap="coolwarm").set_precision(2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Note que, apesar de pouca correlação entre a quantidade vendida e as demais variáveis númericas do dataset, há forte correlação entre as variáveis que não são de interesse, i.e., `valor_venda`, `valor_imposto`, `valor_custo`. Isso faz sentido ao passo que o imposto é uma função do valor da venda (% da venda, por exemplo). E também porque o valor da venda é função do valor do custo, já que a empresa provavelmente define um mark-up (% acima do valor de custo do produto).
# MAGIC 
# MAGIC Assim, podemos deixar em nosso modelo, **ao máximo**, uma das variáveis que não a de interesse. Ou `valor_venda`, ou `valor_imposto` ou `valor_custo`.
# MAGIC 
# MAGIC Abaixo, prosseguimos para a análise da autocorrelação de `qtde_venda`.

# COMMAND ----------

vendas_por_dia = (
    vendas.select("id_data", "qtde_venda")
    .groupBy("id_data")
    .agg({"qtde_venda": "sum"})
    .toPandas()
)

vendas_por_dia = vendas_por_dia.set_index("id_data").sort_index()

ax = pd.plotting.autocorrelation_plot(vendas_por_dia)

ax.set_xlim([0, 550])

# COMMAND ----------

# MAGIC %md
# MAGIC Note que há autocorrelação estatisticamente significativa até a defasagem de ordem 500. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Feature Engineering
# MAGIC 
# MAGIC Iremos agora criar novas colunas com base nas existentes para extrairmos mais informações dos dados fornecidos. Começando com uma das primeiras informações de nossa análise exploratória: a existência de lojas com idades diferentes.
# MAGIC 
# MAGIC Observe, entretanto, que o processo exploratório é iterativo e mais adiante, fora desta seção, criaremos novas features.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Anos Abertos

# COMMAND ----------

lojas = lojas.withColumn("anos_abertos", 2019 - col("ano_abertura"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### One hot encoding
# MAGIC 
# MAGIC Os datasets `lojas` e `produtos` trazem novas e relevantes informações sobre os estabelecimentos e das mercadorias.

# COMMAND ----------

lojas.printSchema()
produtos.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Note que cada loja apresenta uma região, distrito, cidade e unidade federativa diferentes. Entretanto, como estão (`StringType`) não podem ser processados pelos modelos. Devemos, portanto, realizar o One Hot Encoding para transformar nossas variáveis categoricas em númericas sem que se crie relação de ordem entre elas.
# MAGIC 
# MAGIC O mesmo ocorre para os dados de produtos.

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
one_hot_encoder = OneHotEncoder(
    inputCol="produto_nome_index", outputCol="produto_nome_vec"
)
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

string_indexer = StringIndexer(inputCol="categoria", outputCol="categoria_index")
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(inputCol="categoria_index", outputCol="categoria_vec")
encoded_df = one_hot_encoder.fit(indexed_df).transform(indexed_df)

string_indexer = StringIndexer(
    inputCol="sub_categoria", outputCol="sub_categoria_index"
)
indexed_df = string_indexer.fit(encoded_df).transform(encoded_df)

# Create a OneHotEncoder to convert the indexed column to a one-hot encoded column
one_hot_encoder = OneHotEncoder(
    inputCol="sub_categoria_index", outputCol="sub_categoria_vec"
)
encoded_produtos = one_hot_encoder.fit(indexed_df).transform(indexed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelagem
# MAGIC 
# MAGIC Iniciaremos nossa modelagem reduzindo a dimensão dos dados, selecionando as colunas relevantes até agora.

# COMMAND ----------

colunas = ["id_data", "id_loja", "id_produto", "qtde_venda"]

colunas_group_by = ["id_data", "id_loja", "id_produto"]

colunas_agg = {"qtde_venda": "sum"}

vendas_grouped = vendas.select(colunas).groupBy(colunas_group_by).agg(colunas_agg)

# COMMAND ----------

print(vendas.count())
print(vendas_grouped.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Separação ano, mês, dia, número da semana, número do dia na semana
# MAGIC 
# MAGIC Com o dataset dessa maneira fica mais fácil criar novas features que contém informações relevantes da data. Por isso reservamos essa etapa para agora. As informações que conseguimos extrair são: ano, mês, dia do mês, dia da semana, dia do ano e semana do ano.

# COMMAND ----------

vendas_grouped = vendas_grouped.withColumn("ano", year(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn("mes", month(vendas_grouped.id_data))
vendas_grouped = vendas_grouped.withColumn(
    "dia_do_mes", dayofmonth(vendas_grouped.id_data)
)
vendas_grouped = vendas_grouped.withColumn(
    "dia_da_semana", dayofweek(vendas_grouped.id_data)
)
vendas_grouped = vendas_grouped.withColumn(
    "dia_do_ano", dayofyear(vendas_grouped.id_data)
)
vendas_grouped = vendas_grouped.withColumn(
    "semana_do_ano", weekofyear(vendas_grouped.id_data)
)
vendas_grouped.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join nas tabelas
# MAGIC 
# MAGIC Com as informações de data das vendas ja tratados, podemos inserir as informações de lojas e produtos das demais tabelas. Faremos isso a partir de um join entre `vendas`, `lojas` e `produtos`.

# COMMAND ----------

vendas_merged = vendas_grouped.join(encoded_lojas, "id_loja", how="inner").join(
    encoded_produtos, vendas_grouped.id_produto == encoded_produtos.produto, how="inner"
)

vendas_merged.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Criação dos datasets de treino e teste.
# MAGIC 
# MAGIC Com os dados tratados, iremos separar nossa amostra em treino e teste. A rigor, para dados em painel, o ideal é separarmos os dados seguindo a ordem cronológica. O motivo de não usarmos `RandomSplit`, por exemplo, é porque isso pode adicionar viés em nosso modelo, onde ele irá capturar informações do futuro que, naquela data, ainda não existiam.
# MAGIC 
# MAGIC Aliado à isso, para termos o melhor modelo, idealmente teriamos que fazer algo como um *rolling time series cross validation*. Ou seja, separar o dataset de treino e teste com base em uma janela movel, respeitando a ordem das observações. Essa abordagem também cria um espaço entre os datasets para predições mais verídicas, já que não estamos apenas interessados em predições de apenas 1 período a frente. Abaixo uma ilustração.
# MAGIC 
# MAGIC Entretanto, devido à restrição de tempo, não pude fazer isso aqui. No sklearn é possível fazer o mesmo com a função `TimeSeriesSplit()`.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://user-images.githubusercontent.com/28893120/154545460-d1629e8a-22a4-494d-affc-2df3cb95ade7.png">

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
    "sub_categoria_vec",
]

target_column = ["sum(qtde_venda)"]

vendas_selected = vendas_merged.select(
    selected_columns + target_column + ["id_data", "id_loja", "id_produto"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Como temos dados em diferentes proporções, iremos normalizar cada coluna para que a robustez do modelo melhore, usando `StandardScaler`.

# COMMAND ----------

# unlist = udf(lambda x: round(float(list(x)[0]),3), DoubleType())

df = vendas_selected.alias("df")

for i in selected_columns:
    # VectorAssembler Transformation - Converting column to vector type
    assembler = VectorAssembler(inputCols=[i], outputCol=i + "_Vect")

    # MinMaxScaler Transformation
    scaler = StandardScaler(inputCol=i + "_Vect", outputCol=i + "_Scaled")

    # Pipeline of VectorAssembler and MinMaxScaler
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fitting pipeline on dataframe
    df = pipeline.fit(df).transform(df).drop(i + "_Vect")

# COMMAND ----------

# features_columns = [i+"_Scaled" for i in selected_columns]
features_columns = ["ano", "mes", "loja_vec", "produto_nome_vec"]
assembler = VectorAssembler(inputCols=features_columns, outputCol="features")
vendas_vectorized = assembler.transform(df)

# COMMAND ----------

train_test_data = vendas_vectorized.withColumn(
    "rank", percent_rank().over(Window.partitionBy().orderBy("id_data"))
)

train = train_test_data.where("rank <= .8").drop("rank")
test = train_test_data.where("rank > .8").drop("rank")

# COMMAND ----------

# MAGIC %md
# MAGIC Abaixo noto que, após a divisão do dataset, o número de partições cai para 1, reduzindo significativamente a performance do código. Trato isso aumentando o número de partições.

# COMMAND ----------

print("Train antes: ", train.count(), train.rdd.getNumPartitions())
print("Test antes: ", test.count(), test.rdd.getNumPartitions())
train_partitions = train.repartition(15)
test_partitions = test.repartition(10)
print(
    "Train depois: ", train_partitions.count(), train_partitions.rdd.getNumPartitions()
)
print("Test depois: ", test_partitions.count(), test_partitions.rdd.getNumPartitions())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline: modelo simples
# MAGIC 
# MAGIC Em um primeiro momento, nossa estratégia inicial será criar um modelo simples, onde a predição para um produto `i` de uma loja `j` será apenas a média das quantidades vendidas para a combinação mês-ano `t`. Em termos algébricos:
# MAGIC 
# MAGIC $$ \frac{\sum_{}^{}a_{ij}^{t}}{n} $$
# MAGIC 
# MAGIC Onde,
# MAGIC 
# MAGIC t = mes-ano.  
# MAGIC i = produto.  
# MAGIC j = loja.  
# MAGIC n = número de observações para determinada combinação de loja, produto e mes-ano.

# COMMAND ----------

train_naive_fit = (
    train_partitions.select("ano", "mes", "id_loja", "id_produto", "sum(qtde_venda)")
    .groupBy("ano", "mes", "id_loja", "id_produto")
    .agg({"sum(qtde_venda)": "mean"})
)

# COMMAND ----------

train_naive_predict = (
    train_partitions.select("ano", "mes", "id_loja", "id_produto", "sum(qtde_venda)")
    .join(train_naive_fit, ["ano", "mes", "id_loja", "id_produto"])
    .withColumnRenamed("avg(sum(qtde_venda))", "prediction")
)

# COMMAND ----------

test_naive_predict = (
    test_partitions.select("ano", "mes", "id_loja", "id_produto", "sum(qtde_venda)")
    .join(train_naive_fit, ["ano", "mes", "id_loja", "id_produto"])
    .withColumnRenamed("avg(sum(qtde_venda))", "prediction")
)

# COMMAND ----------

evaluator = RegressionEvaluator(
    labelCol="sum(qtde_venda)", predictionCol="prediction", metricName="rmse"
)

train_rmse = evaluator.evaluate(train_naive_predict)
test_rmse = evaluator.evaluate(test_naive_predict)
print("RMSE (treino): ", train_rmse)
print("RMSE (teste): ", test_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Obtemos com esse estimador um valor de RMSE de 1.5 para o dataset de treino e 1.76 para o dataset de teste. Os valores estão de acordo com o que era de se esperar, já que com dados novos nosso modelo performa "pior".

# COMMAND ----------

# MAGIC %md
# MAGIC ### Regressão Linear
# MAGIC 
# MAGIC Iremos agora realizar o modelo de regressão linear.

# COMMAND ----------

features_lr = [
    "ano_Scaled",
    "mes_Scaled",
    "dia_do_mes_Scaled",
    "dia_da_semana_Scaled",
    "dia_do_ano_Scaled",
    "semana_do_ano_Scaled",
    "anos_abertos_Scaled",
    "loja_vec_Scaled",
    "regional_vec_Scaled",
    "distrito_vec_Scaled",
    "cidade_vec_Scaled",
    "uf_vec_Scaled",
    "fornecedor_vec_Scaled",
    "produto_nome_vec_Scaled",
    "categoria_vec_Scaled",
    "sub_categoria_vec_Scaled",
]
assembler = VectorAssembler(inputCols=features_lr, outputCol="features_lr")
train_partitions = assembler.transform(train_partitions)

# COMMAND ----------

test_partitions = assembler.transform(test_partitions)

# COMMAND ----------

lr_classifier = LinearRegression(
    featuresCol="features_lr", labelCol="sum(qtde_venda)"
).fit(train_partitions)
lr_train_predictions = lr_classifier.transform(train_partitions)
lr_test_predictions = lr_classifier.transform(test_partitions)

evaluator = RegressionEvaluator(
    labelCol="sum(qtde_venda)", predictionCol="prediction", metricName="rmse"
)

lr_train_rmse = evaluator.evaluate(lr_train_predictions)
lr_test_rmse = evaluator.evaluate(lr_test_predictions)
print(lr_train_rmse)
print(lr_test_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Note que a performance desse modelo foi pior do que o modelo simples, e os motivos disso são claros. Primeiro, para modelos de regressão linear o ideal é tirar a primeira diferença e ter uma série estacionária. Não obstante, no dataset de teste, para prever a primeira amostra, bastaria pegar o valor real da variável de interesse da última observação e usar como feature para a predição. Para a segunda observação, porém, teriamos que pegar o resultado da primeira predição e usa-lo como feature. E assim sucessivamente. Ou seja, fazer um modelo autoregressivo.
# MAGIC 
# MAGIC Não tive tempo hábil de decifrar como fazer isso no pyspark de maneira eficiente. Fiquei bloqueado em como fazer a predição autoregressiva. O statsmodel oferece o modelo ARIMA, mas nativamente não roda em paralelo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest
# MAGIC 
# MAGIC Por fim, testaremos o modelo random forest. Usaremos 20 para o hiperparâmetro número de árvores e 5 de profundidade máxima.

# COMMAND ----------

rf_classifier = RandomForestRegressor(
    featuresCol="features_lr", labelCol="sum(qtde_venda)", numTrees=20, maxDepth = 5
).fit(train_partitions)

# COMMAND ----------

train_predictions = rf_classifier.transform(train_partitions)
test_predictions = rf_classifier.transform(test_partitions)

# COMMAND ----------

evaluator = RegressionEvaluator(
    labelCol="sum(qtde_venda)", predictionCol="prediction", metricName="rmse"
)

train_rmse = evaluator.evaluate(train_predictions)
test_rmse = evaluator.evaluate(test_predictions)
print(train_rmse)
print(test_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC O modelo performou pior que o modelo de baseline. Por motivos de velocidade na execução, não foi feito cross validation e isso explica a performance ruim.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predição próximos 90 dias.
# MAGIC 
# MAGIC Usaremos o modelo de baseline para prever a demanda para cada produto nos próximos 90 dias. Para isso, iremos prever a demanda de cada produto em cada loja a cada dia e agregar a quantidade demandada por loja.
# MAGIC 
# MAGIC **Passos**:
# MAGIC 1. Criar dataframe com 90 novos dias, a partir da data final.
# MAGIC 2. Fazer cross join entre datas novas, lojas distintas e produtos distintos.
# MAGIC 3. Criar features de ano e mês.
# MAGIC 4. Prever novos valores.
# MAGIC 5. Salvar como csv.
# MAGIC 
# MAGIC Como não temos dados para o ano de 2020 no modelo simples, iremos usar os dados do ano anterior. Dessa maneira, criamos uma coluna `id_data_temp` como auxiliar para essa operação.

# COMMAND ----------

to_predict = spark.sql("SELECT sequence(to_date('2019-01-01'), to_date('2019-03-30'), interval 1 day) as date").withColumn("id_data_temp", explode(col("date")))

# COMMAND ----------

lojas_distinct = lojas.select('id_loja').distinct()
produtos_distinct = produtos.select(col('produto').alias('id_produto')).distinct()

# COMMAND ----------

to_predict = to_predict.crossJoin(lojas_distinct).crossJoin(produtos_distinct)

# COMMAND ----------

to_predict = to_predict.drop('date')

# COMMAND ----------

to_predict = to_predict.withColumn("ano", year(to_predict.id_data_temp))
to_predict = to_predict.withColumn("mes", month(to_predict.id_data_temp))

# COMMAND ----------

# Aqui devemos fazer um left join entre to_predict e o "modelo" treinado (train_naive_fit) para não
# perdermos dados que queremos prever.

predictions = (
    to_predict
    .join(train_naive_fit, ["ano", "mes", "id_loja", "id_produto"], how='left')
    .withColumnRenamed("avg(sum(qtde_venda))", "prediction")
)

# COMMAND ----------

predictions = predictions.withColumn('id_data', F.add_months('id_data_temp', 12))

# COMMAND ----------

predictions = predictions.fillna(0)

# COMMAND ----------

predictions_final = predictions.select('id_data', 'id_loja', 'id_produto', 'prediction').groupBy('id_data', 'id_produto').agg({'prediction': 'sum'}).orderBy(col('id_data').desc())

# COMMAND ----------

predictions_final.toPandas().to_csv('predictions.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Considerações finais
# MAGIC 
# MAGIC **Como melhorar os modelos?**
# MAGIC 
# MAGIC Devido às restrições de tempo, não foi possível realizar algumas melhorias que talvez melhorassem a performance do modelo. Discorremos sobre essas melhorias ao longo do texto.
# MAGIC 
# MAGIC - Rolling Time Series Cross Validation.
# MAGIC - Inserir Lags para capturar efeitos de momentum e sazonalidade.
# MAGIC - Tirar a primeira diferença para tratar a tendência positiva.
# MAGIC - Feature hashing nas colunas que possuem muitas categorias diferentes, por exemplo, cliente, tipo de cliente, etc.
# MAGIC - Adicionar dados de externos, como feriados e nível de desemprego.
# MAGIC 
# MAGIC **Outros modelos que queria testar**
# MAGIC - **ARIMA**: A motivação de usar esse modelo se dá por nossa série principal (qtde_venda), quando olhada individualmente, ser do tipo série temporal. O modelo ARIMA soluciona 3 problemas que identificamos nos dados: i. A existência de autocorrelação. Nesse caso por ser um modelo autoregressivo (AR), a predição levaria em consideração dados do passado; ii. A existência de tendências de longo prazo. Nesse caso é tratado pela existência do componente de média móvel (MA); e, por fim, iii. A não estacionaridade da amostra, que é tratada pelo componente (I) do modelo, ou seja, quantas vezes tirar a primeira diferença.
# MAGIC - **LSTM**: A motivação de usar um modelo de Deep Learning neste caso se dá pelo fato de termos muitos dados, cenário onde tais tipos de algorítmos performam melhor, sobretudo o LSTM. Não obstante, estamos tratando de dados em painel, ou seja, onde há ordem nos dados com base no tempo, e o LSTM leva em consideração dados de dias anterirores.
# MAGIC - **Um modelo para cada janela de predição**: Por fim, outro teste válido a se fazer é o de criar diferentes modelos para diferentes janelas de predição (30d, 60d ou 90d). Isso serve pois predições para os próximos 30d devem levar mais em consideração flutuações de curto prazo, enquanto que modelos de médio/longo prazo (90d) devem dar mais importância a tendências de longo prazo.
