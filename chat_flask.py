from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import random
import re

app = Flask(__name__)
CORS(app)

# Carregar modelo treinado
with open('modelo_chatbot.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

# Respostas para cada intenção
respostas_por_intencao = {
    "cumprimento": "Olá! Como posso te ajudar hoje? 😊",
    "sobre_site": " A COMPASS foi criada com o objetivo de diminuir o grande índice de desperdícios em grandes corporações, direcionado para onde precisa realizando uma mediação entre as grandes empresas e pessoas em situações de desemprego que tenham acesso à internet ou, ongs que buscam recursos como alimentos para fazerem suas ações beneficentes. Na COMPASS, acreditamos no poder da colaboração e da solidariedade. Esse projeto além de ajudar os beneficiários também gera incentivos fiscais para as empresas que participam dessa colaboração solidária. Venha fazer parte dessa missão e nos ajude a criar um mundo melhor.",
    "criacao": "O projeto foi criado por um grupos de aluno da Universidade Nove de Julho no ano de 2024",
    "despedida": "Tchau! Volte sempre ao Chatbot Compass!"
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuação
    return text

def get_bot_response(user_input):
    user_input = preprocess_text(user_input)
    X_input = vectorizer.transform([user_input])
    probs = model.predict_proba(X_input)[0]
    max_prob = max(probs)
    intencao_predita = model.classes_[probs.argmax()]

    if max_prob < 0.3:
        return "Desculpe, ainda estou aprendendo e não entendi bem."

    resposta = respostas_por_intencao.get(intencao_predita)
    if isinstance(resposta, list):
        return random.choice(resposta)
    return resposta or "Desculpe, ainda estou aprendendo e não entendi bem."


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_message = data.get('message', '')
    bot_reply = get_bot_response(user_message)
    return jsonify({'reply': bot_reply})


if __name__ == "__main__":
    app.run(debug=True)