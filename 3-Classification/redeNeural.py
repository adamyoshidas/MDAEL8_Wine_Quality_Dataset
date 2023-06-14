import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o arquivo .data usando o pandas
data = pd.read_csv('0-Datasets/WineQTClear.data')

# Separar os dados em features (X) e target (y)
X = data.drop('fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality', axis=1)
y = data['quality']

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Padronizar os dados de treinamento e teste
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir a arquitetura da rede neural
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=1)

# Treinar a rede neural
mlp.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = mlp.predict(X_test)

# Calcular a acurácia das previsões
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
