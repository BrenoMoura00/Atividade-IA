import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from openai import OpenAI
import streamlit as st
import json
import os

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

df = pd.read_csv("train.csv")
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt']
target = 'SalePrice'
dados = df[features + [target]].copy()
dados = dados.fillna(dados.median())
X = dados[features]
y = dados[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def prever_modelo(GrLivArea, BedroomAbvGr, FullBath, YearBuilt):
    entrada = pd.DataFrame([[GrLivArea, BedroomAbvGr, FullBath, YearBuilt]], columns=features)
    previsao = model.predict(entrada)[0]
    return f"Valor estimado do imóvel: ${previsao:,.2f}"

def buscar_contexto(query):
    return "O mercado imobiliário local valoriza imóveis mais novos e com maior área construída. Bairros nobres possuem um prêmio de 20% no valor final."

tools = [
    {
        "type": "function",
        "function": {
            "name": "prever_modelo",
            "description": "Prevê o preço de uma casa com base em suas características.",
            "parameters": {
                "type": "object",
                "properties": {
                    "GrLivArea": {"type": "number"},
                    "BedroomAbvGr": {"type": "integer"},
                    "FullBath": {"type": "integer"},
                    "YearBuilt": {"type": "integer"}
                },
                "required": ["GrLivArea", "BedroomAbvGr", "FullBath", "YearBuilt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_contexto",
            "description": "Busca informações sobre o mercado imobiliário.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    }
]

def agente(pergunta_usuario, max_steps=5):
    messages = [
        {"role": "system", "content": "Você é um corretor e avaliador de imóveis. Use as ferramentas disponíveis para gerar um laudo de avaliação completo em linguagem natural para o cliente, explicando os valores."},
        {"role": "user", "content": pergunta_usuario}
    ]
    
    for _ in range(max_steps):
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
        msg = resp.choices[0].message
        
        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "prever_modelo":
                    resultado = prever_modelo(**args)
                elif tool_call.function.name == "buscar_contexto":
                    resultado = buscar_contexto(**args)
                else:
                    resultado = "Ferramenta não encontrada."
                
                messages.append({"role": "tool", "content": str(resultado), "tool_call_id": tool_call.id})
        else:
            return msg.content
    return "Limite de passos atingido."

st.title("Avaliador de Imóveis IA")

area = st.number_input("Área Habitável (sqft)", value=1500)
quartos = st.number_input("Quantidade de Quartos", value=3)
banheiros = st.number_input("Quantidade de Banheiros", value=2)
ano = st.number_input("Ano de Construção", value=2000)

if st.button("Gerar Laudo"):
    with st.spinner("Analisando dados e gerando laudo..."):
        pergunta = f"Gere um laudo para uma casa com as seguintes características: {area} sqft de área, {quartos} quartos, {banheiros} banheiros, construída no ano de {ano}. Busque contexto do mercado se necessário."
        resposta = agente(pergunta)
        st.write(resposta)