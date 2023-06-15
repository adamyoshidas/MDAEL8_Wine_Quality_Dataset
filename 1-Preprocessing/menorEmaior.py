import pandas as pd

# Define o nome do arquivo .data
nome_arquivo = '0-Datasets/WineQTClearColums.data'

# Lê o arquivo .data e define o separador (se necessário)
dados = pd.read_csv(nome_arquivo, delimiter=',')

# Itera sobre as colunas dos dados
for coluna in dados.columns:
    # Obtém o menor e maior valor da coluna atual
    menor_valor = dados[coluna].min()
    maior_valor = dados[coluna].max()
    
    # Imprime os resultados
    print(f"Coluna: {coluna}")
    print(f"Mínimo: {menor_valor}")
    print(f"Máximo: {maior_valor}")
    print()
