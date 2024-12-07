#Introdução

Python tem se mostrado uma ferramenta poderosa para análise de dados financeiros. Sua ampla biblioteca de pacotes permite que investidores e analistas desenvolvam soluções para processar grandes volumes de dados, identificar tendências de mercado e realizar análises quantitativas. Este projeto aplica Python para a clusterização de empresas listadas na bolsa de valores, utilizando dados financeiros para identificar grupos com características semelhantes, destacando as melhores opções de investimento.

#Objetivo

O objetivo deste projeto é identificar as melhores empresas para investir na bolsa de valores brasileira. Para isso, foram utilizadas técnicas de machine learning para agrupar empresas em clusters com base em indicadores financeiros como receita líquida, lucro líquido, EBITDA, margem líquida, ROE, fluxo de caixa livre, alavancagem e dívida líquida. Este agrupamento visa destacar os grupos de empresas que apresentam maior potencial de retorno.

#Metodologia

##Obtenção de Dados

Os dados foram obtidos utilizando a biblioteca yfinance, que permite acessar informações financeiras de empresas listadas em bolsas de valores. Os principais passos para a coleta de dados foram:

###Extração dos tickers das empresas listadas na B3 (bolsa brasileira) a partir de um arquivo CSV.

###Coleta de dados históricos de preços e de indicadores financeiros, como balanços patrimoniais, demonstrativos de resultados e fluxos de caixa.

###Organização dos dados em um DataFrame para análise posterior.

##Limpeza e Processamento de Dados

Os dados coletados passaram por uma etapa de limpeza para:

###Remover entradas com valores ausentes.

###Escalar os dados para normalizar as variações entre diferentes métricas financeiras, utilizando o StandardScaler da biblioteca scikit-learn.

##Clusterização

Foi utilizado o algoritmo K-Means para agrupar as empresas. O processo incluiu:

###Determinação do número ideal de clusters:

Aplicação do método do cotovelo, que analisa a soma dos erros ao quadrado (inércia) para diferentes números de clusters.

Uso do Silhouette Score para avaliar a qualidade do agrupamento.

###Execução do K-Means:

Definição do número ideal de clusters com base nas análises anteriores.

Atribuição de cada empresa a um cluster.

##Visualização e Análise

As médias dos indicadores financeiros foram calculadas para cada cluster, permitindo identificar as principais características de cada grupo.

Gráficos de mapa de calor foram gerados para visualizar as diferenças entre os clusters.

##Resultados

O projeto identificou um cluster com as empresas que apresentaram os melhores indicadores financeiros, como maior lucro líquido, EBITDA e fluxo de caixa livre. Estas empresas foram destacadas como opções promissoras para investimento.

###Empresas em Destaque

As empresas do cluster de maior potencial foram salvas em um arquivo CSV, permitindo uma análise posterior detalhada.

##Conclusão

Este projeto demonstra o poder do Python na análise financeira e no suporte à tomada de decisões de investimento. A utilização de técnicas de clusterização permitiu agrupar empresas com base em seus indicadores financeiros, destacando aquelas com maior potencial. Este método pode ser expandido para incluir outros mercados ou incorporar variáveis macroeconômicas, oferecendo uma base robusta para futuras análises quantitativas.
