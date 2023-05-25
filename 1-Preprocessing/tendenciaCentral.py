import pandas as pd
import statistics as stats

input_file = '0-Datasets/WineQTClear.data'
names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
df = pd.read_csv(input_file, names = names)

coluna = df['density']
media = coluna.mean()
mediana = coluna.median()
moda = stats.mode(coluna)
ponto_medio = (min(coluna) + max(coluna)) / 2

# Imprimindo os resultados
print("Coluna: ", coluna.name)
print("Média:", media)
print("Mediana:", mediana)
print("Moda:", moda)
print("Ponto Médio:", ponto_medio)