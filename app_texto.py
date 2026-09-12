"""
Frontend Streamlit do Agente de Blog (Orchestrator-Workers).
"""
import os
import time
import streamlit as st

from agentic_chatbot_texto_backend import stream_blog_generation

st.set_page_config(page_title="Agente de Blog", page_icon="📝", layout="wide")

st.title("📝 Agente de Blog — Orchestrator-Workers")
st.caption("Gere um texto jornalístico sóbrio a partir de um link de notícia, PDF ou texto direto.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Configuração")
    thread_id = st.text_input("ID da conversa (thread)", value="sessao_1")

    st.subheader("🤖 Provedor de IA")
    provider_display = st.selectbox("Escolha o Provedor", ["OpenAI", "Google Gemini"], index=0)
    provider = "openai" if provider_display == "OpenAI" else "gemini"

    if provider == "openai":
        model_options = ["gpt-4o-mini", "gpt-4o"]
    else:
        model_options = ["gemini-2.5-flash", "gemini-2.0-flash"]

    selected_model = st.selectbox("Modelo", model_options, index=0)

    # Verificação de Chaves
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    google_ok = bool(os.getenv("GOOGLE_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))

    st.markdown("---")
    st.markdown("**Status das Chaves (.env):**")
    st.markdown(f"- OpenAI: {'✅ Configurada' if openai_ok else '❌ Ausente'}")
    st.markdown(f"- Gemini: {'✅ Configurada' if google_ok else '❌ Ausente'}")
    st.markdown(f"- Tavily: {'✅ Configurada' if tavily_ok else '⚠️ Ausente (busca externa desativada)'}")

    st.markdown("---")
    st.markdown("**Fluxo:** LLM1 → LLM2 (Tavily) → LLM3-7 (Workers) → LLM8 (Sintetizador) → LLM9 (Revisão)")

    if st.button("🗑️ Limpar resultado e sessão"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# -------------------- Input --------------------
st.subheader("1. Fonte de conteúdo")
source_type = st.radio(
    "Tipo de fonte",
    ["URL de notícia", "PDF de artigo", "Texto direto"],
    horizontal=True
)

source_value = ""
stype = "url"

if source_type == "URL de notícia":
    stype = "url"
    source_value = st.text_input("Cole o link da notícia", placeholder="https://noticias.exemplo.com/materia")
elif source_type == "PDF de artigo":
    stype = "pdf"
    pdf_file = st.file_uploader("Envie o PDF do artigo", type=["pdf"])
    if pdf_file is not None:
        os.makedirs("uploads", exist_ok=True)
        source_value = os.path.join("uploads", pdf_file.name)
        with open(source_value, "wb") as f:
            f.write(pdf_file.read())
else:
    stype = "text"
    source_value = st.text_area("Cole o texto da notícia ou artigo", height=180, placeholder="Insira o texto completo aqui...")

run = st.button("🚀 Gerar texto para blog", type="primary")

# -------------------- Execução --------------------
if run:
    if not source_value.strip():
        st.warning("Forneça uma URL válida, envie um PDF ou digite o texto da notícia.")
        st.stop()

    # Validação da chave selecionada
    if provider == "openai" and not openai_ok:
        st.error("❌ A chave OPENAI_API_KEY não foi encontrada no arquivo .env.")
        st.stop()
    if provider == "gemini" and not google_ok:
        st.error("❌ A chave GOOGLE_API_KEY não foi encontrada no arquivo .env.")
        st.stop()

    st.subheader("2. Progresso do agente")

    status_box = st.empty()
    progress = st.progress(0)
    log_area = st.expander("🔍 Log detalhado dos nós", expanded=True)

    nodes_order = ["reader", "orchestrator", "workers", "synthesizer",
                   "fact_check", "proofread", "save_file"]
    node_labels = {
        "reader":       "LLM1 — Lendo e identificando o tema central",
        "orchestrator": "LLM2 — Buscando notícias relacionadas (Tavily)",
        "workers":      "LLM3-7 — Extraindo informações em paralelo",
        "synthesizer":  "LLM8 — Sintetizando artigo jornalístico sóbrio",
        "fact_check":   "LLM9a — Checagem de consistência factual",
        "proofread":    "LLM9b — Revisão ortográfica e gramatical",
        "save_file":    "💾 Salvando arquivo .txt",
    }

    final_state = {}
    logs_recorded = []

    try:
        with st.spinner("O agente está pesquisando e elaborando a matéria..."):
            for i, event in enumerate(stream_blog_generation(
                source_type=stype,
                source_value=source_value,
                thread_id=thread_id,
                provider=provider,
                model=selected_model
            )):
                for node_name, node_output in event.items():
                    final_state.update(node_output or {})
                    label = node_labels.get(node_name, node_name)
                    msg_list = (node_output or {}).get("messages", [])

                    logs_recorded.append((label, [m.content for m in msg_list]))

                    with log_area:
                        st.write(f"✅ **{label}**")
                        for msg in msg_list:
                            st.caption(msg.content)

                    progress.progress(min((i + 1) / len(nodes_order), 1.0))
                    status_box.info(f"Executando: {label}")

        status_box.success("✅ Matéria jornalística gerada e revisada com sucesso!")

        # Salva na sessão para persistir após cliques em download ou reloads
        st.session_state["resultado_blog"] = {
            "final_text": final_state.get("final_text", ""),
            "theme": final_state.get("theme", ""),
            "messages": [m.content for m in final_state.get("messages", [])],
            "logs": logs_recorded,
            "output_path": final_state.get("output_path", "")
        }

    except Exception as e:
        status_box.error(f"❌ Ocorreu um erro durante a execução: {e}")
        st.exception(e)
        st.stop()


# -------------------- Exibição do Resultado Persistido --------------------
if "resultado_blog" in st.session_state:
    res = st.session_state["resultado_blog"]
    final_text = res.get("final_text", "")

    st.markdown("---")
    st.subheader("3. Texto final")
    if res.get("theme"):
        st.info(f"**Tema apurado:** {res['theme']}")

    st.markdown(final_text)

    # Modo streaming visual (efeito máquina de escrever sob demanda)
    with st.expander("📜 Replay em modo máquina de escrever"):
        placeholder = st.empty()
        texto_acumulado = ""
        palavras = final_text.split()
        for i, palavra in enumerate(palavras):
            texto_acumulado += palavra + " "
            if i % 3 == 0 or i == len(palavras) - 1:
                placeholder.markdown(texto_acumulado + "▌")
                time.sleep(0.01)
        placeholder.markdown(final_text)

    # Download seguro que não perde o estado da página
    st.download_button(
        label="⬇️ Baixar texto final (.txt)",
        data=final_text.encode("utf-8"),
        file_name="texto_final_blog.txt",
        mime="text/plain",
        key="btn_download_texto"
    )

    if res.get("output_path"):
        st.caption(f"Arquivo também salvo localmente em: `{res['output_path']}`")

    # Histórico de mensagens do grafo
    with st.expander("📋 Histórico completo das mensagens do fluxo"):
        for msg_text in res.get("messages", []):
            st.write(f"- {msg_text}")