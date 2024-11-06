import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()
set_debug(True)
apiKey = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=apiKey,
)

mensagens = [
    "Quero visitar algum lugar no Brasil famoso por suas praias e cultura. Pode me recomendar?",
    "Qual o melhor período para visitar em termos de clima?",
    "Quais são os melhores restaurantes locais?",
    "Quais tipos de atividades ao ar livre estão disponíveis?",	
]

# mremoriza as duas últimas mensagens, funciona como o FIFO
memory = ConversationBufferWindowMemory(k=2)

conversation = ConversationChain(llm=llm, verbose=True, memory=memory)

for mensagem in mensagens:
    respostas = conversation.predict(input=mensagem)

    print(respostas)
    # quando dictionary vazio printa todas as variáveis da memória
    print(memory.load_memory_variables({}))
