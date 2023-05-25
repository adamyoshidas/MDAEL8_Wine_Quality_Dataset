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

    #iris = load_iris()
    #print(iris.data)
    #print(iris.target)
    #X = iris.data
    #y = iris.target
    
    X = df
    y = df['quality']

    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)    
    #iris = load_iris()
    #print(iris.data)
    #print(iris.target)
    #X = iris.data
    #y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = DecisionTreeClassifier(max_leaf_nodes=3)
    clf.fit(X_train, y_train)
    tree.plot_tree(clf)
    plt.show()
    
    


if __name__ == "__main__":
    main()