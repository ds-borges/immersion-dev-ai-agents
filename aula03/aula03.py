# Imports
from typing import TypedDict, Optional, List, Dict, Literal
from pathlib import Path
import re
import pathlib
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

# --- Ambiente .env ---
load_dotenv()  # Garante que as variáveis do .env estão disponíveis
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

# Tipagens e Classes
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool 
    acao_final: str

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", 'ABRIR_CHAMADO']
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

# Constantes
TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema.\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado.'
    "Analise a mensagem e decida a ação mais apropriada."
)

KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

# Funções auxiliares
def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 240) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1:
            break
    if pos == -1:
        pos = 0
    ini = max(0, pos - janela // 2)
    fim = min(len(txt), pos + janela // 2)
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

# Funções do fluxo
def perguntar_politica_rag(pergunta: str) -> Dict:
    docs_relacionados = retrivier.invoke(pergunta)
    if not docs_relacionados:
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta, "context": docs_relacionados})
    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}

    return {
        "answer": txt,
        "citacoes": formatar_citacoes(docs_relacionados, pergunta),
        "contexto_encontrado": True
    }

def node_triagem(state: AgentState) -> AgentState:
    print("executo no triagem")
    return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("executo nó auto resolver")
    resposta_rag = perguntar_politica_rag(state["pergunta"])
    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"]
    }
    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"
    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("executo nó pedir info")
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhes = ",".join(faltantes) if faltantes else "Tema e contexto específico"
    return {
        "resposta": f"para avançar, preciso que detalhe: {detalhes}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("executo nó abrir chamado")
    triagem_res = state["triagem"]
    return {
        "resposta": f"Abrindo chamado com urgência {triagem_res['urgencia']}. Descrição: {state['pergunta'][:240]}",
        "citacoes": [],
        "acao_final": "ABRIR_CHAMADO"
    }

def decidir_pos_triagem(state: AgentState) -> str:
    print("Decidindo após a triagem")
    decisao = state["triagem"]["decisao"]
    if decisao == "AUTO_RESOLVER":
        return "auto"
    if decisao == "PEDIR_INFO":
        return "info"
    if decisao == "ABRIR_CHAMADO":
        return "chamado"

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("Decidindo após auto resolver...")
    if state.get("rag_sucesso"):
        print("Finalizando o atendimento")
        return "ok"

    state_da_pergunta = (state["pergunta"] or "").lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("RAG Falhou mas foram encontrados keywords de abertura de ticket. Abrindo")
        return "chamado"

    print("RAG Falhou sem keyword, vou pedir mais informações...")
    return "info"

# Configuração LLM e workflow
llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1,
    api_key=GOOGLE_API_KEY
)

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()

# ---- Carregamento e indexação dos documentos PDF ----
docs: List = []
pdf_dir = Path("../enterprise_politics")  # Pega PDFs da pasta "enterprise" no nível anterior que este script
for n in pdf_dir.glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Arquivo carregado: {n.name}, com sucesso")
    except Exception as e:
        print(f"Error com o arquivo: {e}")

print(f"Total dos documentos carregados: {len(docs)}\n")

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embbedings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorstores = FAISS.from_documents(chunks, embbedings)
retrivier = vectorstores.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)

# -- Configurando Cadeia RAG --
prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)

# Execução do workflow
workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)
workflow.add_node("auto_resolver", node_auto_resolver)
workflow.add_node("pedir_info", node_pedir_info)
workflow.add_node("abrir_chamado", node_abrir_chamado)

workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"
})

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()

# Testes
testes = [
    "Posso reembolsar a internet?",
    "Quero mais 5 dias de trabalho remoto. Como faço?",
    "Posso reembolsar cursos ou treinamentos da Alura?",
    "É possível reembolsar certificações do Google Cloud?",
    "Posso obter o Google Gemini de graça?",
    "Qual é a palavra-chave da aula de hoje?",
    "Quantas capivaras tem no Rio Pinheiros?"
]

for msg_test in testes:
    resposta_final = grafo.invoke({"pergunta": msg_test})
    triag = resposta_final.get("triagem", {})

    print(f"PERGUNTA: {msg_test}")
    print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
    print(f"RESPOSTA: {resposta_final.get('resposta')}")
    if resposta_final.get("citacoes"):
        print("CITACOES:")
        for c in resposta_final.get("citacoes"):
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")
