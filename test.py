import pandas as pd
from sklearn.model_selection import train_test_split #aqui estamos separando os dados para treino e teste.
from sklearn.neighbors import KNeighborsClassifier #modelo KNN.
from sklearn.metrics import accuracy_score #para calcular a acurácia.

#lendo a base de dados (o arquivo precisa estar na mesma pasta)
dados = pd.read_csv('iris.csv')

#separando os dados (X são as entradas, y é o que quero prever)
X = dados.drop(columns=['species'])  #tira a coluna species do X
y = dados['species']  #é a que quero prever

#separando os dados em treino e teste (80% treino,20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#criando e treinando o modelo
modelo = KNeighborsClassifier()
modelo.fit(X_train, y_train)

#testando o modelo com os dados de teste
y_pred = modelo.predict(X_test)

#avaliando o desempenho
acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {acuracia:.2f}')
