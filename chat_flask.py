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
    "cumprimento": [
    "Olá! Como posso te ajudar hoje? 😊",
    "Oi! Estou aqui para tirar suas dúvidas sobre o projeto Compass.",
    "Seja bem-vindo(a)! Como posso ajudar?"
],
  "sobre_site": [
    "A Compass é uma plataforma que conecta empresas que desejam doar com ONGs e pessoas que precisam. Nosso objetivo é combater o desperdício e ajudar quem mais precisa.",
    "A Compass surgiu para facilitar a doação de recursos por empresas, promovendo impacto social e reduzindo o desperdício.",
    "Somos uma ponte entre solidariedade e oportunidade. A Compass conecta doadores e beneficiários para transformar vidas!"
],
"funcionamento": [
    "Empresas cadastradas informam os itens que desejam doar, e ONGs ou beneficiários podem se cadastrar para receber. A Compass faz essa mediação.",
    "O site permite que empresas publiquem suas doações e que ONGs ou pessoas em vulnerabilidade social se cadastrem para receber.",
    "Tudo é feito de forma simples: quem quer doar cadastra os itens, e quem precisa se inscreve para receber. A Compass conecta essas pontas."
],
"criacao": [
    "O projeto Compass foi criado por estudantes da Universidade Nove de Julho como parte de um trabalho acadêmico.",
    "A Compass nasceu em 2024 como um projeto universitário com propósito social.",
    "Criado por um grupo de alunos da UNINOVE, o projeto tem como foco a solidariedade e o combate ao desperdício."
],
"impacto_social": [
    "A Compass contribui diretamente para a redução do desperdício de alimentos e materiais, promovendo inclusão e solidariedade.",
    "O impacto é grande: recursos que seriam descartados chegam a quem realmente precisa.",
    "A plataforma conecta empresas a ONGs e pessoas em situação de vulnerabilidade, promovendo responsabilidade social."
],
"beneficios_empresa": [
    "As empresas que doam através da Compass podem receber incentivos fiscais e também reforçar sua imagem como socialmente responsáveis.",
    "Doar através da Compass pode gerar benefícios fiscais, além de fortalecer o compromisso social da empresa.",
    "Sim! Além de contribuir com a sociedade, a empresa pode aproveitar benefícios legais e tributários ao fazer doações."
],
"contato": [
    "Você pode entrar em contato conosco pelas redes sociais ou pelo e-mail disponível no rodapé do site oficial.",
    "Fale com a nossa equipe por meio das redes sociais ou pelo formulário de contato no site.",
    "Estamos à disposição! Visite o final da página para encontrar nossos canais de contato."
],
"despedida": [
    "Tchau! Volte sempre ao Chatbot Compass!",
    "Até logo! Qualquer dúvida, estou por aqui. 😊",
    "Obrigado por conversar com a gente. Até a próxima!"
],
"ajuda": [
  "Claro! Posso te explicar sobre o projeto, como doar, como funciona a plataforma e muito mais.",
  "Me pergunte sobre o site, como ele funciona, ou como entrar em contato.",
  "Estou aqui para ajudar! Pergunte sobre doações, impacto social ou como participar."
]

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

    if max_prob < 0.05:
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