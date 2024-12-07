# Importando as bibliotecas
import yfinance as yf
import pandas as pd
from datetime import datetime

# Carregar o CSV com os tickers da B3
df_empresas = pd.read_csv("C:/Users/Usuario/Documents/Material de Estudo/Python aplicado à Finanças/acoes-listadas-b3.csv")

# Extrair a coluna "Ticker" e adicionar o sufixo ".SA"
tickers = df_empresas['Ticker'] + ".SA"

# Converter para lista
tickers = tickers.tolist()
print(tickers)

# Definir período
start_date = "2019-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Para armazenar os dados de preços históricos
data_dict = {}

# Iterar sobre cada ticker e baixar dados históricos
for ticker in tickers:
    print(f"Processando dados de preço para {ticker}")
    try:
        # Baixar dados históricos
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df['Ticker'] = ticker  # Adicionar uma coluna com o ticker
            data_dict[ticker] = df
        else:
            print(f"Nenhum dado encontrado para {ticker}")
    except Exception as e:
        print(f"Erro ao processar {ticker}: {e}")

# Verificar se há dados no dicionário
if data_dict:
    all_data = pd.concat(data_dict.values(), ignore_index=True)
    print(all_data.head())
else:
    print("Nenhum dado disponível para concatenar.")

# In[]:

# Inicializar a lista para armazenar os indicadores financeiros
indicadores_financeiros_lista = []    

# Iterar sobre os tickers para extrair os indicadores financeiros
for ticker in tickers:
    print(f"Processando indicadores financeiros para {ticker}")
    try:
        # Obter dados financeiros do yfinance
        acao = yf.Ticker(ticker)

        # DRE (Demonstrativo de Resultados)
        dre = acao.financials
        receita_liquida = dre.loc['Total Revenue'].iloc[0] if 'Total Revenue' in dre.index else None
        lucro_liquido = dre.loc['Net Income'].iloc[0] if 'Net Income' in dre.index else None
        ebitda = dre.loc['EBIT'].iloc[0] if 'EBIT' in dre.index else None

        # Balanço Patrimonial
        balance_sheet = acao.balance_sheet
        patrimonio_liquido = balance_sheet.loc['Ordinary Shares Number'].iloc[0] if 'Ordinary Shares Number' in balance_sheet.index else None
        ativos_totais = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None
        passivos_totais = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None

        # Fluxo de Caixa
        cashflow = acao.cashflow
        fluxo_caixa_operacional = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else None
        capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else None
        free_cash_flow = fluxo_caixa_operacional - capex if fluxo_caixa_operacional and capex else None

        # Cálculos adicionais
        divida_liquida = passivos_totais - ativos_totais if passivos_totais and ativos_totais else None
        margem_liquida = (lucro_liquido / receita_liquida * 100) if lucro_liquido and receita_liquida else None
        roe = (lucro_liquido / patrimonio_liquido * 100) if lucro_liquido and patrimonio_liquido else None
        alavancagem = (passivos_totais / patrimonio_liquido) if passivos_totais and patrimonio_liquido else None

        # Adicionar os dados à lista
        indicadores_financeiros_lista.append({
            'Ticker': ticker,
            'Receita Líquida': receita_liquida,
            'Lucro Líquido': lucro_liquido,
            'EBITDA': ebitda,
            'Ativos Totais': ativos_totais,
            'Passivos Totais': passivos_totais,
            'Patrimônio Líquido': patrimonio_liquido,
            'Dívida Líquida': divida_liquida,
            'Fluxo Caixa Operacional': fluxo_caixa_operacional,
            'CAPEX': capex,
            'Free Cash Flow': free_cash_flow,
            'Margem Líquida (%)': margem_liquida,
            'ROE (%)': roe,
            'Alavancagem Financeira': alavancagem
        })
    except Exception as e:
        print(f"Erro ao processar os dados financeiros para {ticker}: {e}")

# In[]:

    # Exibir os resultados
    df_indicadores = pd.DataFrame(indicadores_financeiros_lista)
    print(df_indicadores) 
    
    # Converter a lista de indicadores financeiros em um DataFrame
    df_indicadores = pd.DataFrame(indicadores_financeiros_lista)

    # Exibir o DataFrame antes da limpeza
    print("Antes da limpeza:")
    print(df_indicadores)

    # Remover linhas onde há pelo menos um valor ausente
    # Caso deseje remover apenas empresas onde TODOS os indicadores estão ausentes, altere `how='any'` para `how='all'`
    df_indicadores_limpo = df_indicadores.dropna(how='any')

    # Exibir o DataFrame após a limpeza
    print("\nApós a limpeza (linhas com valores ausentes foram removidas):")
    print(df_indicadores_limpo)
        
# In[]: 

#Nessa parte do código, realizaremos o processo de clusterização, para ficar clara a distinção dentre as empresas listadas na bolsa, assim gerando grupos de destaque para a análise
    
# Importar bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Usar o DataFrame com os dados limpos
X = df_indicadores_limpo.copy()

# Remover a coluna 'Ticker' para análise de cluster
X = X.drop(['Ticker'], axis=1)

# Verificar valores faltantes
print(f"Valores faltantes: {X.isnull().sum().sum()}")

# Escalar os dados para normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar o número ideal de clusters usando o método do cotovelo
distorsions = []
max_loop = 15  # Número máximo de clusters

for k in range(2, max_loop):  # Loop para testar de 2 a (max_loop - 1) clusters
    try:
        k_means = KMeans(n_clusters=k, random_state=10)
        k_means.fit(X_scaled)
        distorsions.append(k_means.inertia_)
        print(f"k={k}, Inércia={k_means.inertia_}")  # Diagnóstico
    except Exception as e:
        print(f"Erro no cálculo para k={k}: {e}")

# Diagnóstico do comprimento de distorsions
print(f"Distorsions calculadas: {len(distorsions)} elementos")
print(f"Comprimentos - x: {len(range(2, max_loop))}, y: {len(distorsions)}")

# Plotar o método do cotovelo
if len(distorsions) == len(range(2, max_loop)):
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_loop), distorsions, marker='o')
    plt.xticks(range(2, max_loop))
    plt.xlabel("Número de clusters")
    plt.ylabel("Soma dos Erros ao Quadrado (Inércia)")
    plt.title("Método do Cotovelo")
    plt.grid(True)
    plt.show()
else:
    print("Erro: O comprimento de 'distorsions' não corresponde ao esperado.")

# Silhouette Score para validar a escolha do número de clusters
silhouette_scores = []
for k in range(2, max_loop):
    kmeans = KMeans(n_clusters=k, random_state=10, n_init=10)
    kmeans.fit(X_scaled)
    silhouette_scores.append(metrics.silhouette_score(X_scaled, kmeans.labels_))

# Plotar Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(range(2, max_loop), silhouette_scores, marker='o', color='orange')
plt.xticks(range(2, max_loop))
plt.xlabel("Número de clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score por Número de Clusters")
plt.grid(True)
plt.show()

# Definir o número ideal de clusters (por exemplo, a partir dos gráficos)
n_clusters = 5

# Executar KMeans com o número ideal de clusters
k_means = KMeans(n_clusters=n_clusters, random_state=10)
k_means.fit(X_scaled)

# Adicionar os rótulos dos clusters ao DataFrame original
df_indicadores_limpo['Cluster'] = k_means.labels_

# Analisar os clusters
numeric_columns = df_indicadores_limpo.select_dtypes(include=['number']).columns
df_numeric = df_indicadores_limpo[numeric_columns]

cluster_summary = df_numeric.groupby(df_indicadores_limpo['Cluster']).mean()
print("\nResumo dos clusters:")
print(cluster_summary)

# Visualizar as médias dos clusters em um mapa de calor
plt.figure(figsize=(15, 8))
sns.heatmap(cluster_summary.T, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Média dos Indicadores por Cluster")
plt.show()

# Opcional: Salvar os resultados
df_indicadores_limpo.to_csv("resultados_clusterizacao.csv", index=False)

# In[]: 

# Selecionar os indicadores relevantes
indicadores_relevantes = [
    'Receita Líquida', 'Lucro Líquido', 'EBITDA', 'Margem Líquida (%)',
    'ROE (%)', 'Free Cash Flow', 'Dívida Líquida', 'Alavancagem Financeira'
]

df_cluster = df_indicadores_limpo[indicadores_relevantes].copy()

# Inverter os indicadores que devem ser baixos (Dívida Líquida e Alavancagem)
df_cluster['Dívida Líquida'] = -df_cluster['Dívida Líquida']
df_cluster['Alavancagem Financeira'] = -df_cluster['Alavancagem Financeira']

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# Determinar o número ideal de clusters com o método do cotovelo
distorsions = []
max_loop = 15  # Número máximo de clusters

for k in range(2, max_loop):  # Loop para testar de 2 a (max_loop - 1) clusters
    try:
        k_means = KMeans(n_clusters=k, random_state=10)
        k_means.fit(X_scaled)  # Certifique-se de que X_scaled foi definido corretamente
        distorsions.append(k_means.inertia_)
        print(f"k={k}, Inércia={k_means.inertia_}")  # Verifique a saída
    except Exception as e:
        print(f"Erro no cálculo para k={k}: {e}")

# Plotar o método do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(range(2, max_loop), distorsions, marker='o')
plt.xticks(range(2, max_loop))
plt.xlabel("Número de clusters")
plt.ylabel("Soma dos Erros ao Quadrado (Inércia)")
plt.title("Método do Cotovelo")
plt.grid(True)
plt.show()

# Definir o número ideal de clusters
n_clusters = 5

# Aplicar K-Means
k_means = KMeans(n_clusters=n_clusters, random_state=10)
k_means.fit(X_scaled)

# Adicionar os clusters ao DataFrame original
df_indicadores_limpo['Cluster'] = k_means.labels_

# Filtrar apenas as colunas numéricas para calcular as médias por cluster
df_cluster_numerico = df_indicadores_limpo[indicadores_relevantes].select_dtypes(include=['float64', 'int64'])

# Analisar os clusters
cluster_summary = df_cluster_numerico.groupby(df_indicadores_limpo['Cluster']).mean()
print("\nResumo dos clusters:")
print(cluster_summary)

# Identificar o cluster com as melhores empresas (maiores médias)
melhor_cluster = cluster_summary['Lucro Líquido'].idxmax()
print(f"\nO cluster com as melhores empresas é o Cluster {melhor_cluster}")

# Filtrar as empresas do melhor cluster
empresas_destaque = df_indicadores_limpo[df_indicadores_limpo['Cluster'] == melhor_cluster]
print("\nEmpresas em destaque:")
print(empresas_destaque[['Ticker'] + indicadores_relevantes])

# Salvar os resultados
empresas_destaque.to_csv("empresas_destaque.csv", index=False)