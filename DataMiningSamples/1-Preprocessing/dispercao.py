import pandas as pd
import numpy as np


# Leitura do arquivo Excel usando o pandas
input_file = '0-Datasets/WineQTClear.data'
names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality','id']
df = pd.read_csv(input_file, names = names)

# Selecionando a coluna desejada do DataFrame
coluna = df['total sulfur dioxide']
# Calculando a média usando a função mean() do pandas
media = coluna.mean()
# Calculando a média usando a função mean() do pandas
amplitude = np.max(coluna) - np.min(coluna)
# Calcule o desvio padrão dos dados
desvio_padrao = np.std(coluna)

# Imprimindo os resultados
print("total sulfur dioxide")
print("Média:", media)
print("Amplitude:", amplitude)
print("Desvio padrão:", desvio_padrao)