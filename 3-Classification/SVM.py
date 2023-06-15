# Initial imports
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def main():
    #load dataset
    col_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    data = pd.read_csv("0-Datasets/WineQTClear.data", header=None, names=col_names)

    # Separate X and y data
    X = data.drop('residual sugar', axis=1)
    y = data.pH
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # TESTS USING SVM classifier from sk-learn    
    svm = SVC(kernel='poly') # poly, rbf, linear
    # training using train dataset
    svm.fit(X_train, y_train)
    # get support vectors
    print(svm.support_vectors_)
    # get indices of support vectors
    print(svm.support_)
    # get number of support vectors for each class
    print("Qtd Support vectors: ")
    print(svm.n_support_)
    # predict using test dataset
    y_hat_test = svm.predict(X_test)

     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy SVM from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score SVM from sk-learn: {:.2f}".format(f1))

    # Get test confusion matrix    
    #cm = confusion_matrix(y_test, y_hat_test)        
    #plot_confusion_matrix(cm, data['quality'].unique(), False, "Confusion Matrix - SVM sklearn")      
    #plot_confusion_matrix(cm, data['quality'].unique(), True, "Confusion Matrix - SVM sklearn normalized" )  
    #plt.show()


if __name__ == "__main__":
    main()