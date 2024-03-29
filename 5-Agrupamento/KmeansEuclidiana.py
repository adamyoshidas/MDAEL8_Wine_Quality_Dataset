import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/WineQTClear.data'
    names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
    target = 'quality'
    df = pd.read_csv(input_file, names=names)

    x = df.loc[:, features].values
    y = df.loc[:, [target]].values

    # Min-max normalization
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized2Df = pd.DataFrame(data=x_minmax, columns=features)
    normalized2Df = pd.concat([normalized2Df, df[[target]]], axis=1)

    # PCA
    pca = PCA(n_components=2)  # Defina o número de componentes principais desejados
    principal_components = pca.fit_transform(x_minmax)

    # Cria um novo dataframe para armazenar as componentes principais
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df = pd.concat([pca_df, df[[target]]], axis=1)

    # Aplicando K-means aos dados após a aplicação do PCA
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(principal_components)
    labels = kmeans.labels_

    # Visualização dos clusters obtidos
    visualize_clusters(principal_components, labels)

def visualize_clusters(data, labels):
    plt.figure(figsize=(8, 6))

    # Plota cada grupo separadamente e atribui uma cor diferente a cada grupo
    for group in np.unique(labels):
        plt.scatter(data[labels == group, 0], data[labels == group, 1], label=f'Grupo {group+1}', cmap='viridis')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters K-means após PCA')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
