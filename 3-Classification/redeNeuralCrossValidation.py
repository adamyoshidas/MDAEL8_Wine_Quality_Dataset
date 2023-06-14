import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Carregar o arquivo .data usando o pandas
data = pd.read_csv('0-Datasets/WineQTClearColums.data')

# Separar os dados em features (X) e target (y)
X = data.drop('quality', axis=1)
y = data['quality']

# Padronizar os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Definir a arquitetura da rede neural com valores padrão
mlp = MLPClassifier(max_iter=1000, learning_rate="constant", learning_rate_init=0.01)

# Realizar a validação cruzada com k=10
cv_scores = cross_val_score(mlp, X, y, cv=10)

# Calcular a acurácia média das previsões
accuracy = cv_scores.mean()
print("Accuracy:", accuracy)

# Exibir a arquitetura da rede neural
print("Arquitetura da Rede Neural:")
print("Número de neurônios em cada camada escondida:", mlp.hidden_layer_sizes)
print("Função de ativação:", mlp.activation)
