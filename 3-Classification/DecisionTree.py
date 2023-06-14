from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd 

def main():

    input_file = '0-Datasets/WineQTClear.data'
    names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    
    df = pd.read_csv(input_file, names = names)

    X = df
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = DecisionTreeClassifier(max_leaf_nodes=5)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()
    
    
if __name__ == "__main__":
    main()