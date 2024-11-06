from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import os

apiKey = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=apiKey,
)

# Zero-shot learning
# No zero-shot learning, o prompt é fornecido sem nenhum exemplo anterior. 
# O modelo usa o conhecimento pré-existente para responder à pergunta ou cumprir a tarefa. 
# É útil quando se deseja uma resposta direta do modelo sem influenciar sua resposta com exemplos anteriores.

# One-shot learning
# One-shot learning envolve fornecer um único exemplo para o modelo antes de fazer a pergunta. 
# Isso ajuda o modelo a entender o contexto ou o formato da resposta esperada. 
# É particularmente útil para orientar o modelo sobre como responder de maneira específica.

# Few-shot learning
# Few-shot learning, exemplificado pelo código abaixo, utiliza múltiplos exemplos para guiar o modelo na produção de respostas. 
# Isso é especialmente útil para tarefas complexas, em que vários exemplos podem ajudar o modelo a compreender melhor a tarefa e gerar resultados mais precisos.

examples = [
    {
        "question": "Quem viveu mais, Muhammad Ali ou Alan Turing?",
        "answer": """
        São necessárias perguntas de acompanhamento: Sim.
        Pergunta: Quantos anos Muhammad Ali tinha quando morreu?
        Resposta intermediária: Muhammad Ali tinha 74 anos quando morreu.
        Pergunta: Quantos anos Alan Turing tinha quando morreu?
        Resposta intermediária: Alan Turing tinha 41 anos quando morreu.
        Então a resposta final é: Muhammad Ali
        """,
    },
    # Outros exemplos aqui...
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Pergunta: {question}\n{answer}"
)

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Pergunta: {input}",
    input_variables=["input"],
)

prompt = prompt_template.format(input="Quem foi o pai de Mary Ball Washington?")
resposta = llm.invoke(prompt)
print(resposta)
