from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_settings(root: Path) -> None:
    st.title("⚙️ Configurações")
    st.caption("Ajuste o comportamento do pipeline sem editar o código.")

    st.subheader("Detecção")
    st.session_state.confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.confidence_threshold),
        step=0.05,
    )
    st.session_state.detector.confidence = st.session_state.confidence_threshold

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.alert_cooldown = st.number_input(
            "Cooldown do alerta sonoro (s)", 1.0, 120.0,
            float(st.session_state.alert_cooldown), 1.0
        )
        st.session_state.alert_manager.cooldown_seconds = st.session_state.alert_cooldown
    with col2:
        st.session_state.screenshot_cooldown = st.number_input(
            "Cooldown de screenshot (s)", 1.0, 300.0,
            float(st.session_state.screenshot_cooldown), 1.0
        )

    st.subheader("Pré-processamento")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.resize_enabled = st.checkbox("Resize", value=st.session_state.resize_enabled)
        st.session_state.resize_width = st.number_input(
            "Largura do resize", 320, 1920, int(st.session_state.resize_width), 32
        )
        st.session_state.gaussian_blur_enabled = st.checkbox(
            "Gaussian Blur", value=st.session_state.gaussian_blur_enabled
        )
        st.session_state.gaussian_kernel = st.selectbox(
            "Kernel Gaussian", [3, 5, 7, 9],
            index=[3, 5, 7, 9].index(int(st.session_state.gaussian_kernel))
            if int(st.session_state.gaussian_kernel) in [3, 5, 7, 9] else 0,
        )
    with col2:
        st.session_state.contrast_enabled = st.checkbox(
            "Contraste", value=st.session_state.contrast_enabled
        )
        st.session_state.contrast_alpha = st.slider(
            "Alpha do contraste", 0.5, 2.0, float(st.session_state.contrast_alpha), 0.05
        )
        st.session_state.denoise_enabled = st.checkbox(
            "Redução de ruído", value=st.session_state.denoise_enabled
        )
        st.session_state.denoise_strength = st.slider(
            "Força da redução de ruído", 1, 20, int(st.session_state.denoise_strength), 1
        )

    st.session_state.show_preprocess = st.checkbox(
        "Mostrar comparação ORIGINAL | PROCESSADA",
        value=st.session_state.show_preprocess,
    )

    st.subheader("Webcam")
    st.session_state.webcam_index = st.number_input(
        "Índice da câmera OpenCV", min_value=0, max_value=10,
        value=int(st.session_state.webcam_index), step=1
    )

    st.subheader("Modelo")
    model_path = root / "models" / "fire_model.pt"
    st.code(str(model_path.relative_to(root)), language="text")
    if model_path.exists():
        st.success("Arquivo de modelo encontrado.")
    else:
        st.warning("Modelo personalizado ainda não treinado.")

    if st.button("🔄 Recarregar modelo"):
        st.session_state.detector.load_model()
        if st.session_state.detector.available:
            st.success("Modelo recarregado com sucesso.")
        else:
            st.error(st.session_state.detector.error_message or "Não foi possível carregar o modelo.")
