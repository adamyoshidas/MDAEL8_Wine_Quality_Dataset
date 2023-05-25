#Implementation of Kmeans from scratch and using sklearn
#Loading the required modules 
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt

'''
def show_digitsdataset(digits):
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        # label the image with the target value
        ax.text(0, 7, str(digits.target[i]))
'''

def show_wine_quality_dataset(df):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    #ax.scatter(df['alcohol'], df['quality'], c='b', marker='o', alpha=0.5)
    ax.scatter(df['fixed acidity'], df['quality'], c='b', marker='o', alpha=0.5)
    #ax.scatter(df['alcohol'], df['quality'], c='b', marker='o', alpha=0.5)
    #ax.set_xlabel('Alcohol')
    ax.set_xlabel('Fixed Acidity')
    ax.set_ylabel('Quality')
    ax.set_title('Wine Quality Dataset')
    #fig.show()

def plot_samples(projected, labels, title):    
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

def main():
    #Load dataset Digits
    #digits = load_digits()
    #show_digitsdataset(digits)
    input_file = '0-Datasets/WineQTClear.data'
    names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    df = pd.read_csv(input_file, names = names)
    #Transform the data using PCA
    show_wine_quality_dataset(df)

    pca = PCA(2)

    projected = pca.fit_transform(df.values)
    print(pca.explained_variance_ratio_)
    #print(df.data.shape)
    print(projected.shape)    
    plot_samples(projected, df['quality'], 'Original Labels') 
    
    #Applying sklearn GMM function
    gm  = GaussianMixture(n_components=10).fit(projected)
    print(gm.weights_)
    print(gm.means_)
    x = gm.predict(projected)

    #Visualize the results sklearn
    plot_samples(projected, x, 'Clusters Labels GMM')

    plt.show()

if __name__ == "__main__":
    main()