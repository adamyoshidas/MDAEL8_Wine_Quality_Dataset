import pandas as pd
import matplotlib.pyplot as plt

input_file = '0-Datasets/WineQTClear.data'
names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
df = pd.read_csv(input_file, names = names)

coluna = df['quality']

media = coluna.mean()
desvio_padrao = coluna.std()

escore_z = coluna.apply(lambda x: (x - media) / desvio_padrao)

quantil = coluna.quantile([0.25, 0.5, 0.75])
# Imprimindo os resultados
print("Escore Z:\n", escore_z)
print("Quantis:\n", quantil)

plt.boxplot(coluna)
plt.title('Gráfico do Quantil')
plt.xlabel(coluna.name)
plt.ylabel('Valores')
plt.show()