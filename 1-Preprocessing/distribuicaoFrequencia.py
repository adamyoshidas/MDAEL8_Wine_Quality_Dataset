import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Leitura do arquivo Excel usando o pandas
input_file = '0-Datasets/WineQTClear.data'
names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality','id']
df = pd.read_csv(input_file, names = names)

# Selecionando a coluna desejada do DataFrame
dados = df['quality']

num_classes = 6

# Calcular a amplitude da classe
amplitude = (max(dados) - min(dados)) / num_classes

# Calcular os limites das classes
limites_inferiores = [min(dados) + i * amplitude for i in range(num_classes)]
limites_superiores = [limite_inf + amplitude for limite_inf in limites_inferiores]

# Inicializar as contagens de frequência
frequencias = [0] * num_classes

# Contar a frequência de cada valor
for valor in dados:
    for i in range(num_classes):
        if valor >= limites_inferiores[i] and valor < limites_superiores[i]:
            frequencias[i] += 1
            break

# Imprima os resultados
print("Numero de classes escolhido:", num_classes)
print("Amplitude:", amplitude)
print("Frequência:", frequencias)
print("Limites inferiores:", limites_inferiores)
print("Limites superiores:", limites_superiores)

# Imprimir a tabela de distribuição de frequência
print("Tabela de distribuição de frequência")
print("-----------------------------------")
print("Classes\t\tLimites\t\tFrequências")
print("-----------------------------------")
for i in range(num_classes):
    print(f"{i+1}\t\t{limites_inferiores[i]:.2f} - {limites_superiores[i]:.2f}\t\t{frequencias[i]}")
print("-----------------------------------")