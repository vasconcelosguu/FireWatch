from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ai.chat_assistant import answer_firewatch_question
from ai.report_generator import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ReportGenerationError,
    prepare_report_data,
)


SUGGESTIONS = [
    ("📋", "Resuma o histórico", "Faça um resumo do histórico atual."),
    ("🔥", "Classe predominante", "Qual classe aparece mais: Fire ou Smoke?"),
    ("📈", "Evolução temporal", "Como estão distribuídas as detecções ao longo do período?"),
    ("🛡️", "Limitações", "Quais limitações devo considerar antes de interpretar esses registros?"),
]


def _render_chat_header(scope: dict, groq_installed: bool, configured: bool) -> None:
    status = "Pronto para conversar" if groq_installed and configured else (
        "Biblioteca Groq não instalada" if not groq_installed else "Chave da API não configurada"
    )
    status_color = "#22c55e" if groq_installed and configured else "#f59e0b"
    st.markdown(
        f"""
        <div class="fw-chat-hero">
          <div>
            <div class="fw-chat-eyebrow">FIREWATCH INTELLIGENCE</div>
            <div class="fw-chat-title">Assistente Inteligente</div>
            <div class="fw-chat-subtitle">Converse com seus dados de monitoramento em linguagem natural.</div>
            <div class="fw-chat-status"><span style="background:{status_color}"></span>{status}</div>
          </div>
          <div class="fw-chat-mark">✦</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Registros de detecção", f"{scope['total_detections']:,}".replace(",", "."))
    b.metric("Grupos de evidência", f"{scope['grouped_evidence_records']:,}".replace(",", "."))
    c.metric("Fontes analisadas", len(scope["sources"]))


def render_ai_chat(root: Path) -> None:
    load_dotenv(root / ".env", override=False)
    st.markdown(
        """
        <style>
        .fw-chat-hero{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:23px 26px;margin:4px 0 18px;border:1px solid rgba(248,113,113,.24);border-radius:19px;background:linear-gradient(115deg,rgba(127,29,29,.38),rgba(30,41,59,.72) 58%,rgba(15,23,42,.86));}
        .fw-chat-eyebrow{font-size:.66rem;letter-spacing:.16em;font-weight:800;color:#fca5a5;margin-bottom:7px;}
        .fw-chat-title{font-size:1.65rem;font-weight:850;letter-spacing:-.045em;line-height:1.15;}
        .fw-chat-subtitle{font-size:.88rem;opacity:.75;margin-top:7px;}
        .fw-chat-status{display:inline-flex;align-items:center;gap:7px;margin-top:13px;font-size:.73rem;font-weight:700;padding:5px 10px;border-radius:999px;background:rgba(15,23,42,.55);border:1px solid rgba(148,163,184,.2);}
        .fw-chat-status span{width:7px;height:7px;border-radius:50%;display:inline-block;}
        .fw-chat-mark{width:52px;height:52px;display:grid;place-items:center;border-radius:16px;font-size:1.8rem;color:#fed7aa;background:rgba(249,115,22,.13);border:1px solid rgba(249,115,22,.25);}
        .fw-chat-privacy{padding:11px 14px;border:1px solid rgba(56,189,248,.2);background:rgba(14,116,144,.09);border-radius:12px;font-size:.78rem;line-height:1.5;margin:10px 0 18px;}
        [data-testid="stChatMessage"]{border:1px solid rgba(148,163,184,.13);border-radius:14px;padding:12px 15px;background:rgba(30,41,59,.18);}
        [data-testid="stChatInput"]{border-color:rgba(248,113,113,.3);}
        </style>
        """,
        unsafe_allow_html=True,
    )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    configured = bool(api_key and api_key != "coloque_sua_chave_aqui")
    groq_installed = importlib.util.find_spec("groq") is not None

    df = st.session_state.logger.load()
    if df.empty:
        st.title("💬 Assistente Inteligente")
        st.info("Ainda não existem detecções no histórico. Processe um vídeo ou inicie a webcam para alimentar o assistente.")
        return

    records = df.to_dict(orient="records")
    try:
        context = prepare_report_data(records, group_id="HISTORICO_ATUAL", max_event_details=30)
    except ValueError as exc:
        st.info(str(exc))
        return

    scope = context["scope"]
    _render_chat_header(scope, groq_installed, configured)

    st.markdown(
        '<div class="fw-chat-privacy">🔒 <b>Privacidade:</b> o assistente utiliza estatísticas e metadados agregados do histórico. Imagens, vídeos, caminhos locais e coordenadas não são enviados à Groq.</div>',
        unsafe_allow_html=True,
    )

    if not groq_installed:
        st.error("A biblioteca da Groq não está instalada neste ambiente.")
        st.markdown("Abra o terminal do projeto, ative o ambiente virtual e execute:")
        st.code("python -m pip install -r requirements.txt", language="powershell")
        st.caption("Depois, reinicie o Streamlit. A dependência `groq` já está declarada no requirements.txt.")
    elif not configured:
        st.warning("A biblioteca está instalada, mas a chave GROQ_API_KEY não foi encontrada.")
        with st.expander("Como configurar a chave"):
            st.code("GROQ_API_KEY=sua_chave_real\nFIREWATCH_GROQ_MODEL=openai/gpt-oss-120b", language="bash")
            st.caption("Salve no arquivo .env na raiz do projeto. Nunca compartilhe sua chave.")

    model_default = os.getenv("FIREWATCH_GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    model_options = list(AVAILABLE_MODELS)
    if model_default not in model_options:
        model_options.insert(0, model_default)
    with st.expander("⚙️ Configurações do assistente", expanded=False):
        model = st.selectbox(
            "Modelo generativo",
            model_options,
            index=model_options.index(model_default),
            format_func=lambda value: AVAILABLE_MODELS.get(value, value),
            key="firewatch_chat_model",
        )

    st.markdown("### O que você gostaria de investigar?")
    st.caption("Escolha uma sugestão ou escreva sua própria pergunta.")
    cols = st.columns(2)
    for index, (icon, title, prompt) in enumerate(SUGGESTIONS):
        with cols[index % 2]:
            if st.button(f"{icon}  {title}", key=f"chat_suggestion_{index}", use_container_width=True):
                st.session_state.firewatch_chat_pending = prompt

    if "firewatch_chat_messages" not in st.session_state:
        st.session_state.firewatch_chat_messages = []

    if st.session_state.firewatch_chat_messages:
        st.markdown("### Conversa")
        for message in st.session_state.firewatch_chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        st.markdown(
            '<div class="fw-empty"><div style="font-size:1.7rem">✦</div><b>Sua análise começa aqui</b><br><span style="opacity:.72">Pergunte sobre classes detectadas, horários, confiança do modelo ou limitações dos dados.</span></div>',
            unsafe_allow_html=True,
        )

    question = st.chat_input(
        "Ex.: quais grupos merecem uma revisão humana?",
        disabled=not (configured and groq_installed),
    )
    pending = st.session_state.pop("firewatch_chat_pending", None)
    question = question or pending
    if question:
        st.session_state.firewatch_chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        history = st.session_state.firewatch_chat_messages[:-1]
        with st.chat_message("assistant"):
            with st.spinner("Analisando os registros do histórico..."):
                try:
                    answer = answer_firewatch_question(question, context, history=history, model=model)
                    st.markdown(answer)
                    st.session_state.firewatch_chat_messages.append({"role": "assistant", "content": answer})
                except (ValueError, ReportGenerationError) as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Não foi possível concluir a consulta. Verifique a conexão com a Groq e tente novamente. Detalhe: {exc}")

    if st.session_state.firewatch_chat_messages:
        left, right = st.columns([1, 4])
        with left:
            if st.button("🗑️ Limpar conversa", key="clear_firewatch_chat", use_container_width=True):
                st.session_state.firewatch_chat_messages = []
                st.rerun()
