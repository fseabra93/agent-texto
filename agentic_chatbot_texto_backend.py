"""
Backend do Agente de Blog — fluxo Orchestrator-Workers.

Fluxo:
  [input: link, PDF ou texto]
      │
      ▼
  LLM1 (Reader) ──► identifica o tema da notícia/artigo e resume
      │
      ▼
  LLM2 (Orchestrator) ──► usa Tavily para buscar notícias relacionadas
      │
      ├──► LLM3 ─┐
      ├──► LLM4 ─┤
      ├──► LLM5 ─┼──► (paralelo) extrai informações-chave de cada notícia
      ├──► LLM6 ─┤
      └──► LLM7 ─┘
      │
      ▼
  LLM8 (Synthesizer) ──► texto jornalístico sóbrio, ~1 página A4
      │
      ▼
  LLM9a (Fact-Checker) ──► checagem de fatos, ciência e consistência
      │
      ▼
  LLM9b (Revisor) ──► revisão gramatical e ortográfica em PT-BR
      │
      ▼
  [output: arquivo .txt salvo em outputs/texto_final_blog.txt]
"""
import os
import re
import sqlite3
from typing import Annotated, TypedDict, Literal
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB_PATH = os.getenv("CHATBOT_DB_PATH", "chatbot.db")


# ============================================================
# 1. FÁBRICA DE LLM (múltiplos provedores)
# ============================================================
def get_llm(provider: str = "openai", model: str | None = None, temperature: float = 0.3):
    """Fábrica de LLM com alternância dinâmica entre OpenAI e Gemini."""
    provider = (provider or "openai").lower()

    if provider == "openai":
        model = model or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temperature, streaming=True)

    if provider in ("gemini", "google"):
        model = model or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, streaming=True)

    raise ValueError(f"Provedor desconhecido: {provider}")


# ============================================================
# 2. FUNÇÕES AUXILIARES DE ENTRADA
# ============================================================
def fetch_url_content(url: str, max_chars: int = 8000) -> str:
    """Baixa o conteúdo textual de uma URL (notícia) para análise."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        text = resp.text

        # Limpeza simples de HTML
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 50:
            return "ERRO: O conteúdo da página parece muito curto ou não pôde ser extraído."
        return text[:max_chars]
    except Exception as e:
        return f"ERRO ao baixar URL ({url}): {e}"


def read_pdf(file_path: str, max_chars: int = 12000) -> str:
    """Lê o conteúdo textual de um arquivo PDF local."""
    try:
        reader = PdfReader(file_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            return "ERRO: O PDF está vazio ou consiste em imagens escaneadas sem camada de texto."
        return text[:max_chars]
    except Exception as e:
        return f"ERRO ao ler PDF: {e}"


# ============================================================
# 3. ESTADO DO GRAFO
# ============================================================
class BlogState(TypedDict):
    """Estado compartilhado do fluxo Orchestrator-Workers."""
    messages: Annotated[list[BaseMessage], add_messages]

    # input
    source_type: Literal["url", "pdf", "text"]
    source_value: str
    provider: str
    model: str

    # LLM1
    theme: str
    source_summary: str

    # LLM2
    related_news: list[dict]   # [{"title":..., "url":..., "content":...}]

    # LLM3..7
    extracted_articles: list[str]

    # LLM8
    draft_text: str

    # LLM9
    final_text: str
    output_path: str


# ============================================================
# 4. NÓS DO GRAFO
# ============================================================
def node_reader(state: BlogState) -> dict:
    """LLM1 — Lê o link, PDF ou texto e entende o tema."""
    provider = state.get("provider", "openai")
    model = state.get("model") or ("gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash")
    llm = get_llm(provider, model, temperature=0.2)

    stype = state.get("source_type", "url")
    sval = state.get("source_value", "")

    if stype == "pdf":
        content = read_pdf(sval)
    elif stype == "url":
        content = fetch_url_content(sval)
    else:
        content = sval

    if content.startswith("ERRO"):
        raise ValueError(f"Falha na leitura da fonte: {content}")

    prompt = (
        "Você é um jornalista analítico experiente. A partir do conteúdo abaixo, extraia:\n"
        "1) O TEMA CENTRAL: um termo ou frase concisa (máximo 8 palavras), factual, "
        "ideal para pesquisar no Google/Tavily Notícias sobre o assunto.\n"
        "2) RESUMO: um resumo factual em até 5 linhas.\n\n"
        f"CONTEÚDO:\n{content[:8000]}\n\n"
        "Responda estritamente no seguinte formato:\n"
        "TEMA: <termo de busca conciso>\n"
        "RESUMO: <resumo factual>"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    text = resp.content.strip()

    theme = ""
    summary = ""
    for line in text.split("\n"):
        clean_line = line.strip().lstrip("#*- ").strip()
        if clean_line.upper().startswith("TEMA:"):
            theme = clean_line[5:].strip().strip('"*')
        elif clean_line.upper().startswith("RESUMO:"):
            summary = clean_line[7:].strip()

    if not theme:
        for line in text.split("\n"):
            if line.strip():
                theme = line.strip()[:100]
                break

    if not theme:
        theme = "notícias sobre o conteúdo analisado"

    if not summary:
        summary = text[:500]

    return {
        "messages": [AIMessage(content=f"[LLM1/Reader] Tema identificado: '{theme}'")],
        "theme": theme,
        "source_summary": summary,
    }


def node_orchestrator(state: BlogState) -> dict:
    """LLM2 — Orquestrador: usa Tavily para buscar até 5 notícias relacionadas."""
    query = state.get("theme", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY")
    news = []

    if tavily_key:
        try:
            client = TavilyClient(api_key=tavily_key)
            search_query = query if query else "notícias recentes"
            results = client.search(query=search_query, max_results=5, topic="news")
            raw_items = results.get("results", []) if isinstance(results, dict) else []
            for item in raw_items[:5]:
                title = item.get("title", "").strip()
                content = item.get("content", "").strip()
                url = item.get("url", "").strip()
                if title or content:
                    news.append({"title": title, "url": url, "content": content})
        except Exception as e:
            news = [{"title": "Aviso de busca", "url": "", "content": f"Busca externa indisponível ({e}). Usando resumo original."}]
    else:
        news = [{"title": "Aviso", "url": "", "content": "TAVILY_API_KEY não configurada. Usando apenas conteúdo base."}]

    if not news:
        news = [{"title": "Conteúdo original", "url": "", "content": state.get("source_summary", "")}]

    return {
        "messages": [AIMessage(content=f"[LLM2/Orchestrator] {len(news)} notícia(s) relacionada(s) para aprofundamento.")],
        "related_news": news,
    }


def _worker_extract(news_item: dict, worker_id: int, provider: str, model: str | None) -> str:
    """Função executada por cada worker para extrair dados chave."""
    title = news_item.get("title", "")
    content = news_item.get("content", "")
    if not content and not title:
        return f"### Worker {worker_id} — (Vazio)"

    llm = get_llm(provider=provider, model=model, temperature=0.2)
    prompt = (
        f"Você é o Worker {worker_id}. Extraia os pontos factuais mais relevantes "
        f"da notícia abaixo. Seja objetivo, factual e imparcial.\n\n"
        f"TÍTULO: {title}\n"
        f"URL: {news_item.get('url', '')}\n"
        f"CONTEÚDO:\n{content[:4000]}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return f"### Informações da Fonte {worker_id}: {title}\n{resp.content}"


def node_workers_parallel(state: BlogState) -> dict:
    """LLM3..LLM7 — workers em paralelo extraem informações de cada notícia."""
    news = state.get("related_news", [])
    provider = state.get("provider", "openai")
    model = state.get("model") or ("gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash")

    if not news:
        return {
            "messages": [AIMessage(content="[LLM3-7/Workers] Nenhuma notícia adicional para extrair.")],
            "extracted_articles": [f"Resumo da fonte original:\n{state.get('source_summary', '')}"]
        }

    valid_news = [item for item in news[:5] if item.get("title") or item.get("content")]
    if not valid_news:
        valid_news = news[:1]

    with ThreadPoolExecutor(max_workers=min(len(valid_news), 5)) as executor:
        futures = [
            executor.submit(_worker_extract, valid_news[i], i + 1, provider, model)
            for i in range(len(valid_news))
        ]
        extracted = [f.result() for f in futures]

    return {
        "messages": [AIMessage(content=f"[LLM3-7/Workers] Extração paralela concluída com {len(extracted)} fontes.")],
        "extracted_articles": extracted,
    }


def node_synthesizer(state: BlogState) -> dict:
    """LLM8 — Sintetizador: monta o texto jornalístico sóbrio (~1 página A4)."""
    provider = state.get("provider", "openai")
    model = state.get("model") or ("gpt-4o" if provider == "openai" else "gemini-2.5-flash")
    llm = get_llm(provider, model, temperature=0.4)

    joined = "\n\n".join(state.get("extracted_articles", []))
    prompt = (
        "Você é um jornalista sênior de um renomado veículo de comunicação.\n"
        "Com base no tema e nas extrações das fontes abaixo, redija um ARTIGO JORNALÍSTICO COMPLETO E SÓBRIO, "
        "em português do Brasil, com aproximadamente uma página A4 (~400 a 600 palavras).\n\n"
        "Estrutura recomendada:\n"
        "1. TÍTULO: informativo, impactante e sério (sem sensacionalismo).\n"
        "Resumo: um resumo chamativo sobre o artigo, em 2 linhas"
        "2. LEAD: abertura direta explicando o fato central (o que aconteceu, quem está envolvido, onde e quando).\n"
        "3. CORPO DA MATÉRIA: contexto aprofundado, dados relevantes das fontes cruzadas e desdobramentos.\n"
        "4. FECHAMENTO: impacto geral e perspectivas futuras.\n\n"
        "Diretrizes de estilo:\n"
        "- Linguagem formal, sóbria, clara e precisa, com um leve tom de humor e que seja entendido por publico leigo a assuntos relacionados a TI e IA.\n"
        "- Total fidelidade aos fatos apurados; não invente dados.\n\n"
        f"TEMA:\n{state.get('theme', '')}\n\n"
        f"RESUMO DA FONTE ORIGINAL:\n{state.get('source_summary', '')}\n\n"
        f"DADOS DAS FONTES COLETADAS:\n{joined[:12000]}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="[LLM8/Synthesizer] Rascunho da matéria jornalística gerado.")],
        "draft_text": resp.content,
    }


def node_fact_check(state: BlogState) -> dict:
    """LLM9a — Checagem factual e coerência científica/jornalística."""
    provider = state.get("provider", "openai")
    model = state.get("model") or ("gpt-4o" if provider == "openai" else "gemini-2.5-flash")
    llm = get_llm(provider, model, temperature=0.1)

    prompt = (
        "Você é um editor sênior e checador de fatos (fact-checker).\n"
        "Analise criticamente o texto abaixo, verificando a consistência dos fatos, "
        "coerência lógica e neutralidade jornalística.\n"
        "Faça os ajustes necessários diretamente no texto para corrigir eventuais falhas ou imprecisões.\n"
        "Mantenha a estrutura, o tom jornalístico e a extensão.\n"
        "ATENÇÃO: Retorne APENAS o texto revisado na íntegra, sem comentários introdutórios ou notas explicativas.\n\n"
        f"TEXTO ORIGINAL:\n{state['draft_text']}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="[LLM9a/Fact-Check] Checagem de consistência e fatos concluída.")],
        "draft_text": resp.content,
    }


def node_proofread(state: BlogState) -> dict:
    """LLM9b — Revisão ortográfica e gramatical final."""
    provider = state.get("provider", "openai")
    model = state.get("model") or ("gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash")
    llm = get_llm(provider, model, temperature=0.0)

    prompt = (
        "Você é um revisor de texto profissional da língua portuguesa (Novo Acordo Ortográfico).\n"
        "Revise o texto abaixo corrigindo pontuação, concordância, regência, acentuação e fluidez frasal.\n"
        "Mantenha integralmente o estilo e o sentido do texto.\n"
        "ATENÇÃO: Retorne APENAS o texto final revisado, sem notas, cumprimentos ou introduções.\n\n"
        f"TEXTO:\n{state['draft_text']}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="[LLM9b/Proofread] Revisão ortográfica e gramatical final concluída.")],
        "final_text": resp.content,
    }


def node_save_file(state: BlogState) -> dict:
    """Salva o texto final em .txt e devolve o caminho."""
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "texto_final_blog.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(state.get("final_text", ""))
    return {
        "messages": [AIMessage(content=f"[Save] Arquivo gerado em: {path}")],
        "output_path": path,
    }


# ============================================================
# 5. CONSTRUÇÃO DO GRAFO
# ============================================================
def build_blog_graph():
    """Constrói o StateGraph Orchestrator-Workers com memória SQLite."""
    builder = StateGraph(BlogState)

    builder.add_node("reader", node_reader)
    builder.add_node("orchestrator", node_orchestrator)
    builder.add_node("workers", node_workers_parallel)
    builder.add_node("synthesizer", node_synthesizer)
    builder.add_node("fact_check", node_fact_check)
    builder.add_node("proofread", node_proofread)
    builder.add_node("save_file", node_save_file)

    builder.add_edge(START, "reader")
    builder.add_edge("reader", "orchestrator")
    builder.add_edge("orchestrator", "workers")
    builder.add_edge("workers", "synthesizer")
    builder.add_edge("synthesizer", "fact_check")
    builder.add_edge("fact_check", "proofread")
    builder.add_edge("proofread", "save_file")
    builder.add_edge("save_file", END)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return builder.compile(checkpointer=checkpointer)


# ============================================================
# 6. PONTO DE ENTRADA PARA STREAMLIT (streaming de eventos)
# ============================================================
def stream_blog_generation(
    source_type: str,
    source_value: str,
    thread_id: str = "default",
    provider: str = "openai",
    model: str | None = None
):
    """
    Executa o grafo em modo streaming, emitindo o estado a cada nó concluído.
    """
    graph = build_blog_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: BlogState = {
        "messages": [HumanMessage(content=f"Gerar texto a partir de {source_type}: {source_value[:200]}")],
        "source_type": source_type,
        "source_value": source_value,
        "provider": provider,
        "model": model or "",
        "theme": "",
        "source_summary": "",
        "related_news": [],
        "extracted_articles": [],
        "draft_text": "",
        "final_text": "",
        "output_path": "",
    }

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        yield event