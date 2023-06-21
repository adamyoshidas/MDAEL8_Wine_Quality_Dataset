import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/WineQTClean.data'
    names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    df = pd.read_csv(input_file, names=names)

    # Medidas de tendência central
    print("---------------------")
    print("MEDIDAS DE TENDENCIA CENTRAL")
    print("---------------------")
    print("Teor Alcoolico")
    print("Media:")
    print(df['alcohol'].mean())
    print("Mediana:")
    print(df['alcohol'].median())
    print("Ponto médio:")
    print((df['alcohol'].max() + df['alcohol'].min()) / 2)
    print("Moda:")
    print(df['alcohol'].mode())
    print("---------------------")

    print("pH")
    print("Media:")
    print(df['pH'].mean())
    print("Mediana:")
    print(df['pH'].median())
    print("Ponto médio:")
    print((df['pH'].max() + df['pH'].min()) / 2)
    print("Moda:")
    print(df['pH'].mode())
    print("---------------------")

    print("Acidez Citrica")
    print("Media:")
    print(df['citric acid'].mean())
    print("Mediana:")
    print(df['citric acid'].median())
    print("Ponto médio:")
    print((df['citric acid'].max() + df['citric acid'].min()) / 2)
    print("Moda:")
    print(df['citric acid'].mode())
    print("---------------------")

    #print("Gosto")
    #print("Moda:")
    #print(df['Odor'].mode())
    #print("---------------------")

    #print("Odor")
    #print("Moda:")
    #print(df['Taste'].mode())
    #print("---------------------")

    #print("Gordura")
    #print("Moda:")
    #print(df['Fat'].mode())
    #print("---------------------")

    #print("Turbidez")
    #print("Moda:")
    #print(df['Turbidity'].mode())


    # Medidas de dispersão
    print("MEDIDAS DE DISPERSAO")
    print("---------------------")
    print("Teor Alcoolico")
    print(df['alcohol'].max() - df['alcohol'].min())
    print("Amplitude:")
    print("Desvio padrão:")
    print(df['alcohol'].std())
    print("Variância:")
    print(df['alcohol'].var())
    print("Coeficiente de variação:")
    print((df['alcohol'].std() / df['alcohol'].mean()) * 100)
    print("---------------------")

    print("pH")
    print("Amplitude:")
    print(df['pH'].max() - df['pH'].min())
    print("Desvio padrão:")
    print(df['pH'].std())
    print("Variância:")
    print(df['pH'].var())
    print("Coeficiente de variação:")
    print((df['pH'].std() / df['pH'].mean()) * 100)
    print("---------------------")

    print("Acidez Citrica")
    print("Amplitude:")
    print(df['citric acid'].max() - df['citric acid'].min())
    print("Desvio padrão:")
    print(df['citric acid'].std())
    print("Variância:")
    print(df['citric acid'].var())
    print("Coeficiente de variação:")
    print((df['citric acid'].std() / df['citric acid'].mean()) * 100)
    print("---------------------")

    # Medidas de posição relativa
    print("MEDIDAS DE POSICAO RELATIVA")
    print("---------------------")
    print("Teor Alcoolico")
    print("Quartis:")
    print(df['alcohol'].quantile([0.25, 0.5, 0.75]))
    z_score = (df['alcohol'] - df['alcohol'].mean()) / df['alcohol'].std()
    print("Escore-z:")
    print(z_score)
    print("---------------------")

    # Boxplot
    plt.boxplot(df['alcohol'])
    plt.xlabel('Teor Alcoolico')
    plt.title('Boxplot - Teor Alcoolico')
    plt.show()

    # Medidas de associação
    print("MEDIDAS DE ASSOCIACAO")
    print("---------------------")
    print("Covariância entre pH e Teor alcoólico:")
    cov = df['pH'].cov(df['alcohol'])
    print(cov)
    correlation = df['pH'].corr(df['alcohol'])
    print("Correlação entre pH e Teor alcoólico:")
    print(correlation)
    plt.scatter(df['pH'], df['alcohol'])
    plt.xlabel('pH')
    plt.ylabel('alcohol')
    plt.title('Gráfico de Dispersão - pH vs alcohol')
    plt.show()
    sns.regplot(x='pH', y='alcohol', data=df)
    plt.xlabel('pH')
    plt.ylabel('alcohol')
    plt.title('Gráfico de Dispersão - pH vs alcohol')
    plt.show()

if __name__ == "__main__":
    main()
