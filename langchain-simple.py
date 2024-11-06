from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
apiKey = os.getenv("OPENAI_API_KEY")

numero_de_dias = 7
numero_de_pessoas = 2
local = "Natal / RN"

template = PromptTemplate.from_template(
    "Crie um roteiro de viagem de {numero_de_dias} dias para {numero_de_pessoas} pessoas em {local}"
)

prompt = template.format(
    numero_de_dias=numero_de_dias, numero_de_pessoas=numero_de_pessoas, local=local
)
print(prompt)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=apiKey,
)

resposta = llm.invoke(prompt)
print(resposta.content)
