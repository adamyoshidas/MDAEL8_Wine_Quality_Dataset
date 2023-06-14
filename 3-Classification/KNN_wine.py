import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from collections import Counter

# Calculate distance between two points
def minkowski_distance(a, b, p=1):    
    # Store the number of dimensions
    dim = len(a)    
    # Set initial distance to 0
    distance = 0
    
    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)    
    return distance

def find_best_k(X_train, y_train):
    k_values = [1, 3, 5, 7, 9]  # Valores de K a serem testados
    best_k = None
    best_accuracy = 0

    for k in k_values:
        # Criar um classificador k-NN com o valor atual de K
        knn = KNeighborsClassifier(n_neighbors=k)

        # Calcular a acurácia média usando validação cruzada com 5 folds
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        accuracy = scores.mean()

        # Verificar se a acurácia atual é a melhor até agora
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k

def knn_predict(X_train, X_test, y_train, y_test, k, p):    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test

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
    # Load data from the .data file
    col_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']
    data = pd.read_csv("0-Datasets/WineQTClear.data", header=None, names=col_names)

    # Separate X and y data
    X = data.drop('quality', axis=1)
    y = data.quality
    print("Total samples: {}".format(X.shape[0]))

    # Split the data - 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    # STEP 1 - TESTS USING knn classifier write from scratch    
    # Make predictions on test dataset using knn classifier
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=2)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}%".format(f1))

    # Get test confusion matrix
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, data['quality'].unique(), False, "Confusion Matrix - K-NN")      
    plot_confusion_matrix(cm, data['quality'].unique(), True, "Confusion Matrix - K-NN normalized")  

    # STEP 2 - TESTS USING knn classifier from scikit-learn
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)

     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Accuracy K-NN from scikit-learn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scikit-learn: {:.2f}%".format(f1))

    # Get test confusion matrix    
    cm = confusion_matrix(y_test, y_hat_test)        
    plot_confusion_matrix(cm, data['quality'].unique(), False, "Confusion Matrix - K-NN scikit-learn")      
    plot_confusion_matrix(cm, data['quality'].unique(), True, "Confusion Matrix - K-NN scikit-learn normalized")  
    plt.show()

    # Encontrar o melhor valor de K
    best_k = find_best_k(X_train, y_train)
    print("Best K value:", best_k)

    # Usar o melhor valor de K para treinar e testar o modelo
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)


if __name__ == "__main__":
    main()