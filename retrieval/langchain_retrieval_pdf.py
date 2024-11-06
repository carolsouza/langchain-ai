import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from langchain_text_splitters import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
set_debug(True)
apiKey = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=apiKey,
)

carregadores = [
    PyPDFLoader("assets/GTB_standard_Nov23.pdf"),
    PyPDFLoader("assets/GTB_gold_Nov23.pdf"),
    PyPDFLoader("assets/GTB_platinum_Nov23.pdf"),
]
documentos = []
for carregador in carregadores:
    documentos.extend(carregador.load())

quebrador = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
textos = quebrador.split_documents(documentos)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(textos, embeddings)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

pergunta = "Como devo proceder caso tenha um item comprado roubado"
resultado = qa_chain.invoke({"query": pergunta})
