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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    


def main():
    #load dataset
    col_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    data = pd.read_csv("0-Datasets/WineQTClear.data", header=None, names=col_names)

    # Separate X and y data
    X = data.drop('quality', axis=1)
    y = data.quality
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create SVM classifier
    svm = SVC(kernel='poly')  # poly, rbf, linear

    # Perform cross-validation
    num_folds = 10  # Número de folds desejado
    accuracies = cross_val_score(svm, X_train, y_train, cv=num_folds)
    
    # Imprimir acurácia de cada fold
    for i, accuracy in enumerate(accuracies):
        print("Fold {}: {:.2f}%".format(i+1, accuracy * 100))
    
    # Calcular e imprimir média da acurácia
    average_accuracy = np.mean(accuracies) * 100
    print("Average Accuracy: {:.2f}%".format(average_accuracy))


if __name__ == "__main__":
    main()