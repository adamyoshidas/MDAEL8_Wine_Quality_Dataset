import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '0-Datasets/WineQTClear.data'
names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
df = pd.read_csv(input_file, names = names)

coluna1 = df['fixed acidity']
coluna2 = df['density']

covariancia = coluna1.cov(coluna2)
correlacao = coluna1.corr(coluna2)

# Imprimindo os resultados

print("Correlação e covariancia entre:", coluna1.name, "e" , coluna2.name)
print("Covariância:", covariancia)
print("Correlação:", correlacao)

sns.regplot(x=coluna1, y=coluna2)
plt.title('Gráfico de Dispersão')
plt.xlabel(coluna1.name)
plt.ylabel(coluna2.name)
plt.show()

sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Matriz de Correlação')
plt.show()