from __future__ import annotations

from pathlib import Path

import streamlit as st

from detector.detector import FireSmokeDetector
from interface.history import render_history
from interface.home import render_home
from interface.monitor import render_monitor
from interface.settings import render_settings
from interface.webcam import render_webcam
from utils.alerts import AlertManager
from utils.logger import DetectionLogger


ROOT = Path(__file__).resolve().parent


def init_state() -> None:
    defaults = {
        "confidence_threshold": 0.50,
        "resize_enabled": True,
        "resize_width": 960,
        "gaussian_blur_enabled": False,
        "gaussian_kernel": 3,
        "contrast_enabled": True,
        "contrast_alpha": 1.08,
        "contrast_beta": 2,
        "denoise_enabled": False,
        "denoise_strength": 5,
        "show_preprocess": False,
        "alert_cooldown": 5.0,
        "screenshot_cooldown": 8.0,
        "webcam_index": 0,
        "webcam_running": False,
        "webcam_capture": None,
        "last_screenshots": {},
        "video_stats": {"processing_time": 0.0, "average_fps": 0.0},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    model_path = ROOT / "models" / "fire_model.pt"
    if "detector" not in st.session_state:
        st.session_state.detector = FireSmokeDetector(
            model_path=model_path,
            confidence=st.session_state.confidence_threshold,
        )
    else:
        st.session_state.detector.confidence = st.session_state.confidence_threshold

    if "logger" not in st.session_state:
        st.session_state.logger = DetectionLogger(ROOT / "logs" / "detections.csv")

    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AlertManager(st.session_state.alert_cooldown)
    st.session_state.alert_manager.cooldown_seconds = st.session_state.alert_cooldown


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px;}
        [data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.18);
            padding: 14px 16px;
            border-radius: 14px;
        }
        .fw-hero {
            padding: 22px 24px;
            border-radius: 18px;
            border: 1px solid rgba(239, 68, 68, .28);
            background: linear-gradient(135deg, rgba(127,29,29,.30), rgba(15,23,42,.70));
            margin-bottom: 18px;
        }
        .fw-status-ok {color:#22c55e; font-weight:700;}
        .fw-muted {opacity:.76;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="FireWatch AI",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    inject_css()

    st.sidebar.markdown("# 🔥 FireWatch AI")
    st.sidebar.caption("Central inteligente de monitoramento")

    pages = {
        "🏠 Início": render_home,
        "🎥 Monitoramento": render_monitor,
        "📷 Webcam": render_webcam,
        "📊 Histórico": render_history,
        "⚙️ Configurações": render_settings,
    }
    selected = st.sidebar.radio("Navegação", list(pages.keys()), label_visibility="collapsed")

    detector = st.session_state.detector
    st.sidebar.divider()
    if detector.available:
        st.sidebar.success("Modelo Fire/Smoke carregado")
    else:
        st.sidebar.warning("Modelo personalizado ainda não treinado.")
    st.sidebar.caption(f"Threshold: {st.session_state.confidence_threshold:.2f}")

    pages[selected](ROOT)


if __name__ == "__main__":
    main()
