"""
Backend do Agente de Blog — fluxo Orchestrator-Workers.

Fluxo:
  [input: link, PDF ou texto]
      │
      ▼
  LLM1 (Reader) ──► identifica o tema da notícia/artigo e resume
      │
      ▼
  LLM2 (Orchestrator) ──► usa Tavily para buscar notícias com suas respectivas fontes
      │
      ├──► LLM3 ─┐
      ├──► LLM4 ─┤
      ├──► LLM5 ─┼──► (paralelo) cada worker recebe a notícia + fonte e extrai dados
      ├──► LLM6 ─┤
      └──► LLM7 ─┘
      │
      ├──► [textos_fonte.txt: união dos textos crus dos agentes LLM3 a LLM7 com fontes]
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
  [output: salva outputs/texto_final_blog.txt e outputs/textos_fonte.txt]
"""
import os
import re
import sqlite3
from urllib.parse import urlparse
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

        # Limpeza de scripts, estilos e tags HTML
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


def compile_raw_sources_text(
    extracted_articles: list[str],
    source_type: str = "",
    source_value: str = "",
    theme: str = ""
) -> str:
    """
    Função dedicada a reunir o texto cru extraído pelos agentes
    LLM 3, LLM 4, LLM 5, LLM 6 e LLM 7 com suas respectivas fontes.
    """
    separator = "=" * 80
    header = [
        separator,
        "COMPILAÇÃO DE TEXTOS E FONTES BRUTAS DOS AGENTES (LLM 3 A LLM 7)",
        separator,
        f"TEMA CENTRAL APURADO: {theme}",
        f"FONTE ORIGINAL DE ENTRADA ({source_type.upper()}): {source_value[:250]}",
        f"TOTAL DE AGENTES EXTRATORES EXECUTADOS: {len(extracted_articles)}",
        separator,
        "",
    ]
    body = "\n\n".join(extracted_articles)
    return "\n".join(header) + "\n\n" + body


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
    related_news: list[dict]   # [{"title":..., "url":..., "content":..., "source_name":...}]

    # LLM3..7
    extracted_articles: list[str]

    # Textos crus compilados com fontes (LLM 3 a LLM 7)
    raw_sources_text: str
    sources_output_path: str

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
    """
    LLM2 — Orquestrador: usa Tavily para buscar até 5 notícias relacionadas,
    garantindo que cada notícia contenha sua respectiva fonte (URL e nome do veículo).
    """
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

                # Extrai domínio como nome da fonte se disponível
                source_name = urlparse(url).netloc.replace("www.", "") if url else "Web"

                if title or content:
                    news.append({
                        "title": title,
                        "url": url,
                        "content": content,
                        "source_name": source_name
                    })
        except Exception as e:
            news.append({
                "title": f"Aviso de busca: {query}",
                "url": state.get("source_value", "Fonte Original"),
                "content": f"Busca externa temporariamente indisponível ({e}). Baseado no resumo: {state.get('source_summary', '')}",
                "source_name": "Origem Primária"
            })

    # Garante que os 5 agentes (LLM 3 a LLM 7) tenham notícias e fontes atribuídas
    orig_url = state.get("source_value", "")
    orig_type = state.get("source_type", "texto")
    orig_domain = urlparse(orig_url).netloc.replace("www.", "") if orig_url.startswith("http") else f"Arquivo/Entrada ({orig_type})"

    while len(news) < 5:
        idx = len(news) + 1
        news.append({
            "title": f"Fonte de Contexto #{idx}: {state.get('theme', 'Notícia Base')}",
            "url": orig_url if orig_url else f"Documento Base #{idx}",
            "content": state.get("source_summary", "Conteúdo extraído da fonte inicial."),
            "source_name": orig_domain
        })

    return {
        "messages": [AIMessage(content=f"[LLM2/Orchestrator] 5 notícias com fontes mapeadas para os agentes LLM 3 a LLM 7.")],
        "related_news": news[:5],
    }


def _worker_extract(news_item: dict, worker_id: int, provider: str, model: str | None) -> str:
    """
    Função executada pelos agentes LLM 3, LLM 4, LLM 5, LLM 6 e LLM 7.
    Cada agente recebe explicitamente a notícia COM a sua respectiva fonte.
    """
    title = news_item.get("title", "").strip() or "Notícia sem título"
    content = news_item.get("content", "").strip()
    url = news_item.get("url", "").strip() or "Fonte original"
    source_name = news_item.get("source_name", "").strip() or urlparse(url).netloc or "Fonte Externa"

    llm = get_llm(provider=provider, model=model, temperature=0.2)
    prompt = (
        f"Você é o agente LLM {worker_id} (Worker especializado em apuração jornalística).\n"
        f"Você recebeu a seguinte notícia acompanhada de sua respectiva FONTE.\n\n"
        f"DADOS DA FONTE E NOTÍCIA:\n"
        f"- FONTE / VEÍCULO: {source_name}\n"
        f"- URL / LINK: {url}\n"
        f"- TÍTULO: {title}\n"
        f"- CONTEÚDO BRUTO:\n{content[:4000]}\n\n"
        "Sua tarefa:\n"
        "1. Identifique e extraia detalhadamente os fatos principais, dados numéricos, datas, "
        "declarações e desdobramentos presentes nesta notícia.\n"
        "2. Mantenha total fidelidade factual ao texto fornecido pela fonte.\n"
        "3. Indique expressamente a fonte e o título no início da sua resposta.\n\n"
        "Responda no formato:\n"
        f"AGENTE RESPONSÁVEL: LLM {worker_id}\n"
        f"FONTE: {url} ({source_name})\n"
        f"TÍTULO DA MATÉRIA: {title}\n"
        "PONTOS-CHAVE E EXTRAÇÃO FACTUAL:\n"
        "<listar os pontos factuais com clareza e riqueza de detalhes>"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])

    return (
        f"================================================================================\n"
        f"AGENTE: LLM {worker_id} (Worker {worker_id - 2})\n"
        f"FONTE: {url} | VEÍCULO: {source_name}\n"
        f"TÍTULO: {title}\n"
        f"================================================================================\n"
        f"{resp.content.strip()}"
    )


def node_workers_parallel(state: BlogState) -> dict:
    """
    LLM 3, LLM 4, LLM 5, LLM 6 e LLM 7:
    Cinco workers executam em paralelo via ThreadPoolExecutor. Cada um recebe
    sua respectiva notícia e fonte mapeadas pelo LLM 2.
    """
    news = state.get("related_news", [])[:5]
    provider = state.get("provider", "openai")
    model = state.get("model") or ("gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash")

    # Garante execução dos 5 agentes: LLM 3, LLM 4, LLM 5, LLM 6, LLM 7 (IDs 3, 4, 5, 6, 7)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_worker_extract, news[i], i + 3, provider, model)
            for i in range(len(news))
        ]
        extracted = [f.result() for f in futures]

    return {
        "messages": [AIMessage(content=f"[LLM3-7/Workers] Extração concluída pelos 5 agentes (LLM 3, LLM 4, LLM 5, LLM 6 e LLM 7) com suas fontes.")],
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
        "Com base no tema e nas extrações das fontes abaixo (apuradas pelos agentes LLM 3 a LLM 7), "
        "redija um ARTIGO JORNALÍSTICO COMPLETO E SÓBRIO, em português do Brasil, "
        "com aproximadamente uma página A4 (~400 a 600 palavras).\n\n"
        "Estrutura recomendada:\n"
        "1. TÍTULO: informativo, impactante e sério (sem sensacionalismo).\n"
        "2. RESUMO: um resumo chamativo em 2 linhas logo abaixo do título.\n"
        "3. LEAD: abertura direta explicando o fato central (o que aconteceu, quem está envolvido, onde e quando).\n"
        "4. CORPO DA MATÉRIA: contexto aprofundado, dados relevantes das fontes cruzadas e desdobramentos.\n"
        "5. FECHAMENTO: impacto geral e perspectivas futuras.\n\n"
        "Diretrizes de estilo:\n"
        "- Linguagem formal, sóbria, clara e precisa, acessível ao público geral e leigo.\n"
        "- Total fidelidade aos fatos apurados; mencione com naturalidade as fontes das informações.\n\n"
        f"TEMA:\n{state.get('theme', '')}\n\n"
        f"RESUMO DA FONTE ORIGINAL:\n{state.get('source_summary', '')}\n\n"
        f"DADOS DAS FONTES COLETADAS (LLM 3 A LLM 7):\n{joined[:14000]}"
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages": [AIMessage(content="[LLM8/Synthesizer] Rascunho da matéria jornalística gerado a partir de todas as fontes.")],
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
    """
    Salva:
    1. O texto final sintetizado e revisado em outputs/texto_final_blog.txt.
    2. O arquivo com os textos crus e respectivas fontes de LLM 3 a LLM 7 em outputs/textos_fonte.txt.
    """
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Salva a matéria final
    final_path = os.path.join(out_dir, "texto_final_blog.txt")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write(state.get("final_text", ""))

    # 2. Reúne e salva os textos crus de LLM 3 a LLM 7 com as respectivas fontes
    raw_sources = compile_raw_sources_text(
        extracted_articles=state.get("extracted_articles", []),
        source_type=state.get("source_type", ""),
        source_value=state.get("source_value", ""),
        theme=state.get("theme", "")
    )
    sources_path = os.path.join(out_dir, "textos_fonte.txt")
    with open(sources_path, "w", encoding="utf-8") as f:
        f.write(raw_sources)

    return {
        "messages": [
            AIMessage(content=f"[Save] Matéria final salva em: {final_path}"),
            AIMessage(content=f"[Save] Compilação de fontes (LLM 3-7) salva em: {sources_path}"),
        ],
        "output_path": final_path,
        "sources_output_path": sources_path,
        "raw_sources_text": raw_sources,
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
        "raw_sources_text": "",
        "sources_output_path": "",
        "draft_text": "",
        "final_text": "",
        "output_path": "",
    }

    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        yield event