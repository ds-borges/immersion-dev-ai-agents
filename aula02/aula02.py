import os
import re
import pathlib
from pathlib import Path
from typing import List, Dict

# Carrega as variáveis de ambiente do arquivo .env na raiz do projeto
from dotenv import load_dotenv
load_dotenv()

# Lê a chave da Gemini API do .env
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# Caminho dos arquivos PDF
DOCS_PATH = "enterprise_politics/"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 0
SEARCH_TYPE = "similarity_score_threshold"
SEARCH_KWARGS = {"score_threshold": 0.3, "k": 4}

docs: List = []

# Funções utilitárias
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela // 2), min(len(txt), pos + janela // 2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source", "")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]

def perguntar_politica_rag(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
    txt = (answer or "").strip()
    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
    return {"answer": txt, "citacoes": formatar_citacoes(docs_relacionados, pergunta), "contexto_encontrado": True}

# Carregamento dos documentos PDF
for n in Path(DOCS_PATH).glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Arquivo carregado: {n.name}, com sucesso")
    except Exception as e:
        print(f"Erro com o arquivo: {e}")
print(f"Total de documentos carregados: {len(docs)}\n")

# Processamento dos documentos em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)

# Embeddings e VectorStore
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type=SEARCH_TYPE, search_kwargs=SEARCH_KWARGS)

# Configuração do LLM e prompt
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    api_key=GOOGLE_API_KEY
)
prompt_rag = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
        "Responda SOMENTE com base no contexto fornecido. "
        "Se não houver base suficiente, responda apenas 'Não sei'."
    ),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])
document_chain = create_stuff_documents_chain(llm, prompt_rag)

# Testes
testes = [
    "Posso reembolsar a internet ?",
    "Quero mais 5 dias de trabalho remoto, posso ?",
    "Posso reembolsar cursos ou treinamento da alura ?",
    "Quantas capivaras tem no rio pinheiro ?"
]

for msg_teste in testes:
    resposta = perguntar_politica_rag(msg_teste)
    print(f"PERGUNTA: {msg_teste}")
    print(f"RESPOSTA: {resposta['answer']}")
    if resposta['contexto_encontrado']:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")
