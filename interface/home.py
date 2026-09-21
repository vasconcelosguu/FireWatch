from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_home(root: Path) -> None:
    detector = st.session_state.detector

    st.markdown(
        """
        <div class="fw-hero">
          <h1 style="margin:0">🔥 FireWatch AI</h1>
          <p style="font-size:1.15rem;margin:.4rem 0 0 0">Sistema Inteligente de Detecção de Incêndios e Fumaça</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classes", "2", "Fire / Smoke")
    c2.metric("Entrada", "2 modos", "Vídeo / Webcam")
    c3.metric("IA", "YOLO", "Detecção de objetos")
    c4.metric("Threshold", f"{st.session_state.confidence_threshold:.0%}", "Configurável")

    st.subheader("Status do sistema")
    if detector.available:
        st.success("🟢 Sistema operacional — modelo Fire/Smoke carregado.")
    else:
        st.info(
            "🟢 Interface operacional. ⚠️ Modelo personalizado ainda não treinado. "
            "Treine o dataset e gere models/fire_model.pt para ativar a detecção real."
        )

    st.subheader("Fluxo de funcionamento")
    st.code(
        "Vídeo/Webcam\n"
        "      ↓\n"
        "   OpenCV\n"
        "      ↓\n"
        "Pré-processamento\n"
        "      ↓\n"
        "     YOLO\n"
        "      ↓\n"
        " Fire / Smoke\n"
        "      ↓\n"
        "    Alerta\n"
        "      ↓\n"
        "   Registro",
        language="text",
    )

    st.subheader("O que o projeto demonstra")
    st.markdown(
        """
        - Processamento de vídeo frame a frame com **OpenCV**.
        - Pipeline de pré-processamento de imagens antes da inferência.
        - Detecção de objetos com **Ultralytics YOLO** em duas classes: Fire e Smoke.
        - Bounding boxes, confiança, FPS, alertas, screenshots e histórico em CSV.
        - Dashboard acadêmico para interpretar as ocorrências registradas.
        """
    )
