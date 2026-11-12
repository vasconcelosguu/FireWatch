from __future__ import annotations

from pathlib import Path

import streamlit as st

from detector.detector import FireSmokeDetector
from interface.history import render_history
from interface.ai_reports import render_ai_reports
from interface.ai_chat import render_ai_chat
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
        :root { --fw-red:#ef4444; --fw-orange:#f97316; --fw-cyan:#38bdf8; --fw-border:rgba(148,163,184,.18); }
        .block-container {padding-top:1.25rem; padding-bottom:3rem; max-width:1480px;}
        h1,h2,h3 {letter-spacing:-.035em;}
        h2 {margin-top:1.25rem;}
        [data-testid="stCaptionContainer"] {line-height:1.45;}
        [data-testid="stSidebar"] [role="radiogroup"] label {padding:.42rem .55rem; border-radius:10px; transition:background .15s ease;}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(148,163,184,.10);}
        [data-testid="stFileUploaderDropzone"] {border:1px dashed rgba(248,113,113,.42); border-radius:14px; background:rgba(127,29,29,.07);}
        [data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stTextInput"] {border-radius:10px;}
        [data-testid="stAlert"] {border-radius:12px;}
        [data-testid="stProgress"] > div > div {background:linear-gradient(90deg,#dc2626,#f97316);}
        .fw-section-kicker {font-size:.68rem; text-transform:uppercase; letter-spacing:.13em; font-weight:800; color:#fb923c;}
        .fw-panel {border:1px solid var(--fw-border); border-radius:16px; padding:16px 18px; background:linear-gradient(145deg,rgba(30,41,59,.42),rgba(15,23,42,.18));}
        .fw-note {border-left:3px solid #f97316; padding:10px 14px; background:rgba(249,115,22,.08); border-radius:0 10px 10px 0; font-size:.83rem; line-height:1.5;}
        .fw-empty {text-align:center; padding:26px 18px; border:1px dashed var(--fw-border); border-radius:16px; background:rgba(30,41,59,.20);}
        [data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 12% 0%, rgba(127,29,29,.13), transparent 34%), var(--background-color);}
        [data-testid="stSidebar"] {border-right:1px solid var(--fw-border);}
        [data-testid="stSidebar"] > div:first-child {padding-top:1.4rem;}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {letter-spacing:-.04em;}
        [data-testid="stMetric"] {background:linear-gradient(145deg,rgba(30,41,59,.62),rgba(15,23,42,.28)); border:1px solid var(--fw-border); padding:15px 17px; border-radius:15px; box-shadow:0 8px 24px rgba(0,0,0,.08);}
        [data-testid="stMetricLabel"] {font-size:.82rem; opacity:.82;}
        [data-testid="stMetricValue"] {font-weight:750; letter-spacing:-.035em;}
        .fw-hero {position:relative; overflow:hidden; padding:28px 30px; border-radius:22px; border:1px solid rgba(248,113,113,.25); background:linear-gradient(115deg,rgba(127,29,29,.48),rgba(30,41,59,.78) 58%,rgba(15,23,42,.9)); margin:4px 0 22px; box-shadow:0 18px 45px rgba(0,0,0,.14);}
        .fw-hero:after {content:'🔥'; position:absolute; right:5%; top:3%; font-size:6.5rem; opacity:.12; filter:blur(.2px);}
        .fw-hero-eyebrow {font-size:.7rem; font-weight:800; letter-spacing:.16em; color:#fca5a5; margin-bottom:7px;}
        .fw-hero h1 {font-size:2.45rem; line-height:1.1; margin:0; letter-spacing:-.055em;}
        .fw-hero p {font-size:1.05rem; margin:.65rem 0 1rem; color:rgba(255,255,255,.82);}
        .fw-hero-tags {display:flex; flex-wrap:wrap; gap:8px;}
        .fw-hero-tags span {font-size:.72rem; font-weight:700; border:1px solid rgba(255,255,255,.2); background:rgba(15,23,42,.32); padding:5px 10px; border-radius:999px; color:#f8fafc;}
        .fw-pipeline {display:flex; align-items:stretch; gap:9px; overflow-x:auto; padding:14px 2px 8px;}
        .fw-pipeline-step {flex:1 0 145px; min-width:145px; padding:16px 13px; border:1px solid var(--fw-border); border-radius:16px; background:linear-gradient(160deg,rgba(51,65,85,.48),rgba(15,23,42,.32)); text-align:center; box-shadow:0 8px 20px rgba(0,0,0,.07);}
        .fw-step-source {border-color:rgba(56,189,248,.32);}
        .fw-step-genai {border-color:rgba(167,139,250,.42); background:linear-gradient(160deg,rgba(91,33,182,.22),rgba(15,23,42,.35));}
        .fw-step-icon {font-size:1.65rem; margin-bottom:7px;}
        .fw-step-kicker {font-size:.58rem; font-weight:800; letter-spacing:.11em; color:#94a3b8; min-height:15px;}
        .fw-step-title {font-size:.9rem; font-weight:800; margin:5px 0; color:var(--text-color);}
        .fw-step-desc {font-size:.7rem; line-height:1.35; color:var(--text-color); opacity:.68;}
        .fw-pipeline-arrow {align-self:center; color:#f87171; font-size:1.45rem; font-weight:800; flex:0 0 auto;}
        .fw-pipeline-outputs {display:flex; flex:1 0 170px; min-width:170px; flex-direction:column; justify-content:center; gap:7px;}
        .fw-output-card {display:flex; align-items:center; gap:10px; padding:9px 11px; border:1px solid var(--fw-border); border-radius:12px; background:rgba(30,41,59,.35);}
        .fw-output-card>span {font-size:1.2rem;}
        .fw-output-card b,.fw-output-card small {display:block;}
        .fw-output-card b {font-size:.78rem;}
        .fw-output-card small {font-size:.65rem; opacity:.68;}
        .fw-diagram-note {font-size:.73rem; color:var(--text-color); opacity:.66; margin:3px 0 18px; padding-left:3px;}
        .fw-feature-card {height:100%; min-height:180px; padding:18px; border:1px solid var(--fw-border); border-radius:16px; background:linear-gradient(145deg,rgba(30,41,59,.48),rgba(15,23,42,.2));}
        .fw-feature-icon {font-size:1.7rem; margin-bottom:10px;}
        .fw-feature-title {font-weight:800; margin-bottom:6px;}
        .fw-feature-desc {font-size:.82rem; line-height:1.5; opacity:.76; min-height:62px;}
        .fw-feature-hint {font-size:.7rem; color:#fca5a5; font-weight:700; margin-top:9px;}
        div.stButton > button[kind="primary"] {border:0; background:linear-gradient(100deg,#dc2626,#f97316); font-weight:750; box-shadow:0 5px 16px rgba(220,38,38,.18);}
        div.stButton > button {border-radius:10px; transition:transform .15s ease,border-color .15s ease;}
        div.stButton > button:hover {transform:translateY(-1px); border-color:rgba(248,113,113,.6);}
        [data-testid="stDataFrame"], [data-testid="stTable"] {border:1px solid var(--fw-border); border-radius:12px; overflow:hidden;}
        [data-testid="stExpander"] {border:1px solid var(--fw-border); border-radius:12px;}
        hr {border-color:var(--fw-border);}
        @media (max-width:760px) { .block-container{padding-left:1rem;padding-right:1rem;} .fw-hero{padding:22px;} .fw-hero h1{font-size:1.9rem;} .fw-hero:after{font-size:4.5rem;right:2%;} }
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
    st.sidebar.caption("CENTRAL DE MONITORAMENTO")
    st.sidebar.markdown("<div style='height:1px;background:rgba(148,163,184,.18);margin:12px 0 16px'></div>", unsafe_allow_html=True)

    pages = {
        "🏠 Início": render_home,
        "🎥 Monitoramento": render_monitor,
        "📷 Webcam": render_webcam,
        "📊 Histórico": render_history,
        "🧠 Relatórios Inteligentes": render_ai_reports,
        "💬 Assistente Inteligente": render_ai_chat,
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
