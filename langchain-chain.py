from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
set_debug(True)
apiKey = os.getenv("OPENAI_API_KEY")


# define a classe Destino
class Destino(BaseModel):
    cidade: str = Field("cidade a visitar")
    motivo: str = Field("motivo pelo qual é interessante visitar a cidade")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=apiKey,
)

# inicializa o parser do JSON com a classe modelo do pydantic
parseador = JsonOutputParser(pydantic_object=Destino)

modelo_cidade = ChatPromptTemplate(
    [
        """Sugira uma cidade dado meu interesse por {interesse}",
            {formatacao_de_saida}
            """
    ],
    input_variables=["interesse"],
    partial_variables={"formatacao_de_saida": parseador.get_format_instructions()},
)

modelo_restaurantes = ChatPromptTemplate.from_template(
    "Sugira restaurantes populares entre locais em {cidade}"
)

modelo_cultural = ChatPromptTemplate.from_template(
    "Sugira atividades culturais em {cidade}"
)

cadeia_cidade = LLMChain(prompt=modelo_cidade, llm=llm)
cadeia_restaurantes = LLMChain(prompt=modelo_restaurantes, llm=llm)
cadeia_cultural = LLMChain(prompt=modelo_cultural, llm=llm)

cadeia = SimpleSequentialChain(
    verbose=True,
    chains=[cadeia_cidade, cadeia_restaurantes, cadeia_cultural],
)


resultado = cadeia.invoke("praias")
print(resultado)
