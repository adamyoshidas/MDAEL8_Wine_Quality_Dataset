import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o arquivo .data usando o pandas
data = pd.read_csv('0-Datasets/WineQTClearColums.data')

# Separar os dados em features (X) e target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Dividir os dados em treinamento e teste utilizando holdout (70% treinamento, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Padronizar os dados de treinamento e teste
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir a arquitetura da rede neural com valores padrão
mlp = MLPClassifier()

# Treinar a rede neural
mlp.fit(X_train, y_train)

# Fazer previsões nos dados de teste
y_pred = mlp.predict(X_test)

# Calcular a acurácia das previsões
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Exibir a arquitetura da rede neural
print("Arquitetura da Rede Neural:")
print("Número de neurônios em cada camada escondida:", mlp.hidden_layer_sizes)
print("Função de ativação:", mlp.activation)
