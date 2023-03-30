from tokenize import group
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/WineQTClear.data'
    names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    df = pd.read_csv(input_file, names = names) # Nome das colunas     
    #df['date'] = df['date'].astype('datetime64[ns]')                   
    ShowInformationDataFrame(df,"Dataframe original")
    
    mini = df['pH'].min()
   
    ampli =(df['pH'].max() - df['pH'].min())/6
    df['pH_group'] = pd.cut(df['pH'], bins=[mini, mini+ampli, mini+ampli*2, mini+ampli*3, mini+ampli*4, mini+ampli*5, mini+ampli*6], include_lowest=True)
    ShowCO2Frequency(df, "pH tabela")

    freq = df['pH_group'].value_counts().sort_index()
    ShowCO2Frequency(freq, "pH frequencia")

    intervalos = ['2.739-2.952', '2.952-3.163', '3.163-3.375', '3.375-3.587', '3.587-3.798', '3.798-4.01',]
    plt.bar(intervalos, freq, color="blue")
    plt.xticks(intervalos)
    plt.ylabel('Frequência')
    plt.xlabel('Niveis de pH')
    plt.title('Distribuição de frequência para os niveis de pH')
    plt.show()

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def ShowFrequencia(df, message=""):
    print(message+"\n")
    print(df.head(20))
    print("\n")
    
def ShowCO2Frequency(df, message = ""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")


if __name__ == "__main__":
    main()