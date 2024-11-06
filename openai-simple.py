from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY")

numero_de_dias = 7
numero_de_pessoas = 2
local = 'Natal / RN'

prompt = f"Crie um roteiro de viagem de {numero_de_dias} dias para {numero_de_pessoas} pessoas em {local}"
print(prompt)

cliente = OpenAI(api_key=apiKey)
resposta = cliente.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

print(resposta)

roteiro_viagem = resposta.choices[0].message.content
print(roteiro_viagem)