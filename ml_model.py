import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Carregando os dados
with open('exemplo_intencoes.json', 'r', encoding='utf-8') as f:
    dados = json.load(f)

perguntas = [item['pergunta'] for item in dados]
intencoes = [item['intencao'] for item in dados]

# Vetorizando as perguntas
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(perguntas)

# Treinando o modelo
model = MultinomialNB()
model.fit(X, intencoes)

# Salvando o modelo e o vetor
with open('modelo_chatbot.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("Modelo treinado e salvo com sucesso!")