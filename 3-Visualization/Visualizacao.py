import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/WineQTClean.data'
    names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    df = pd.read_csv(input_file, names=names)

    plt.hist(df['pH'], bins=10, edgecolor='black')
    plt.xlabel('pH')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência do pH')
    x_ticks = np.arange(df['pH'].min(), df['pH'].max() + 0.5, 0.5) #Mostrar a cada 0,5 no eixo x
    plt.xticks(x_ticks)
    plt.show()
    plt.savefig('histograma_pH.jpeg')
    plt.close()

    plt.hist(df['alcohol'], bins=10, edgecolor='black')
    plt.xlabel('Alcool')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência do teor alcoolico ')
    plt.show()
    plt.savefig('histograma_alcool.jpeg')
    plt.close()

    plt.hist(df['citric acid'], bins=10, edgecolor='black')
    plt.xlabel('Acidez Citrica')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência da acidez citrica')
    plt.show()
    plt.savefig('histograma_acidezcitrica.jpeg')
    plt.close()

    plt.hist(df['residual sugar'], bins=10, edgecolor='black')
    plt.xlabel('Açucar residual')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Frequência do açucar residual')
    plt.show()
    plt.savefig('histograma_acucar.jpeg')
    plt.close()


if __name__ == "__main__":
    main()
