from __future__ import annotations

from pathlib import Path

import streamlit as st
import pandas as pd


def _render_pipeline_diagram() -> None:
    st.markdown(
        """
        <section class="fw-pipeline" aria-label="Fluxo de funcionamento do FireWatch AI">
          <div class="fw-pipeline-step fw-step-source">
            <div class="fw-step-icon">🎥</div>
            <div class="fw-step-kicker">ENTRADA</div>
            <div class="fw-step-title">Vídeo / Webcam</div>
            <div class="fw-step-desc">Captura dos frames</div>
          </div>
          <div class="fw-pipeline-arrow">→</div>
          <div class="fw-pipeline-step">
            <div class="fw-step-icon">🖼️</div>
            <div class="fw-step-kicker">VISÃO COMPUTACIONAL</div>
            <div class="fw-step-title">OpenCV + YOLO</div>
            <div class="fw-step-desc">Pré-processamento e detecção</div>
          </div>
          <div class="fw-pipeline-arrow">→</div>
          <div class="fw-pipeline-step">
            <div class="fw-step-icon">📋</div>
            <div class="fw-step-kicker">DADOS</div>
            <div class="fw-step-title">Registros</div>
            <div class="fw-step-desc">Fire, Smoke, confiança e horário</div>
          </div>
          <div class="fw-pipeline-arrow">→</div>
          <div class="fw-pipeline-step fw-step-genai">
            <div class="fw-step-icon">✨</div>
            <div class="fw-step-kicker">IA GENERATIVA</div>
            <div class="fw-step-title">Groq + GPT-OSS</div>
            <div class="fw-step-desc">Interpretação dos dados</div>
          </div>
          <div class="fw-pipeline-arrow">→</div>
          <div class="fw-pipeline-outputs">
            <div class="fw-output-card"><span>📄</span><div><b>Relatórios</b><small>Análises técnicas</small></div></div>
            <div class="fw-output-card"><span>💬</span><div><b>Chatbot</b><small>Perguntas sobre o histórico</small></div></div>
            <div class="fw-output-card"><span>📝</span><div><b>Resumo</b><small>Análise após o vídeo</small></div></div>
          </div>
        </section>
        <div class="fw-diagram-note">O YOLO identifica possíveis ocorrências; a IA generativa interpreta os registros. A confirmação de um incêndio exige verificação humana.</div>
        """,
        unsafe_allow_html=True,
    )


def render_home(root: Path) -> None:
    detector = st.session_state.detector
    df = st.session_state.logger.load()
    total = len(df)
    fire_count = int((df["class_name"] == "Fire").sum()) if total else 0
    smoke_count = int((df["class_name"] == "Smoke").sum()) if total else 0
    mean_conf = float(df["confidence"].mean()) if total else 0.0

    st.markdown(
        """
        <div class="fw-hero">
          <div class="fw-hero-eyebrow">PLATAFORMA DE MONITORAMENTO INTELIGENTE</div>
          <h1>🔥 FireWatch AI</h1>
          <p>Detecção de fogo e fumaça com visão computacional e análise generativa.</p>
          <div class="fw-hero-tags"><span>YOLO + OpenCV</span><span>Groq API</span><span>GPT-OSS</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detecções registradas", f"{total:,}".replace(",", "."), "Histórico atual")
    c2.metric("Fire", f"{fire_count:,}".replace(",", "."), "Classe detectada")
    c3.metric("Smoke", f"{smoke_count:,}".replace(",", "."), "Classe detectada")
    c4.metric("Confiança média", f"{mean_conf:.1%}" if total else "—", "Pontuação YOLO")

    st.markdown("### Visão geral do sistema")
    status_col, model_col = st.columns([1.4, 1])
    with status_col:
        if detector.available:
            st.success("Modelo Fire/Smoke carregado e pronto para inferência.")
        else:
            st.warning("Interface disponível. O modelo personalizado ainda não foi carregado.")
    with model_col:
        st.info(f"Threshold atual: **{st.session_state.confidence_threshold:.0%}**")

    st.markdown("### Como o FireWatch funciona")
    st.caption("Da captura de vídeo à geração de conhecimento: fluxo integrado das duas abordagens de IA.")
    _render_pipeline_diagram()

    st.markdown("### Visão do histórico")
    if total:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("<div class='fw-section-kicker'>DISTRIBUIÇÃO POR CLASSE</div>", unsafe_allow_html=True)
            class_counts = df["class_name"].fillna("Não informada").value_counts().rename_axis("Classe").to_frame("Detecções")
            st.bar_chart(class_counts, use_container_width=True, height=230)
        with chart_right:
            st.markdown("<div class='fw-section-kicker'>ATIVIDADE AO LONGO DO TEMPO</div>", unsafe_allow_html=True)
            timeline = df.dropna(subset=["timestamp"]).copy()
            if not timeline.empty:
                timeline = timeline.set_index("timestamp").resample("1min").size().rename("Registros").to_frame()
                st.bar_chart(timeline, use_container_width=True, height=230)
            else:
                st.info("Os registros atuais não possuem horários válidos para montar a linha do tempo.")
        st.caption("Os gráficos representam registros do detector no histórico, não incidentes confirmados.")
    else:
        st.markdown(
            "<div class='fw-empty'><div style='font-size:2rem'>📊</div><b>Seu painel ainda não tem registros</b><br><span style='opacity:.72'>Processe um vídeo ou inicie a webcam para visualizar estatísticas e tendências.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("### O que você pode fazer")
    cards = st.columns(3)
    items = [
        ("🎥", "Monitorar vídeos", "Envie um vídeo e acompanhe as detecções frame a frame.", "Use a página Monitoramento."),
        ("💬", "Conversar com o assistente", "Faça perguntas em linguagem natural sobre os registros.", "Abra Assistente Inteligente."),
        ("📄", "Gerar relatórios", "Transforme estatísticas de detecção em análises técnicas exportáveis.", "Acesse Relatórios Inteligentes."),
    ]
    for col, (icon, title, description, hint) in zip(cards, items):
        with col:
            st.markdown(
                f'<div class="fw-feature-card"><div class="fw-feature-icon">{icon}</div>'
                f'<div class="fw-feature-title">{title}</div><div class="fw-feature-desc">{description}</div>'
                f'<div class="fw-feature-hint">{hint}</div></div>',
                unsafe_allow_html=True,
            )

    st.caption("Nota: confiança do YOLO não representa probabilidade de incêndio confirmado. Toda ocorrência deve ser validada por uma pessoa responsável.")
