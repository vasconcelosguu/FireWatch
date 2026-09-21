from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from detector.preprocess import ImagePreprocessor, PreprocessConfig


def _preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(
        PreprocessConfig(
            resize_enabled=st.session_state.resize_enabled,
            resize_width=int(st.session_state.resize_width),
            gaussian_blur_enabled=st.session_state.gaussian_blur_enabled,
            gaussian_kernel=int(st.session_state.gaussian_kernel),
            contrast_enabled=st.session_state.contrast_enabled,
            contrast_alpha=float(st.session_state.contrast_alpha),
            contrast_beta=int(st.session_state.contrast_beta),
            denoise_enabled=st.session_state.denoise_enabled,
            denoise_strength=int(st.session_state.denoise_strength),
        )
    )


def _save_occurrence(root: Path, frame, detections) -> None:
    if not detections:
        return
    now_mono = time.monotonic()
    by_class: dict[str, list] = {}
    for det in detections:
        by_class.setdefault(det.class_name, []).append(det)

    for class_name, items in by_class.items():
        key = f"webcam:{class_name}"
        last = st.session_state.last_screenshots.get(key, -1e9)
        if now_mono - last < st.session_state.screenshot_cooldown:
            continue
        st.session_state.last_screenshots[key] = now_mono

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = root / "screenshots" / f"{class_name.lower()}_{stamp}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        rel = path.relative_to(root).as_posix()

        for det in items:
            st.session_state.logger.log(
                "webcam", det.class_name, det.confidence,
                det.x1, det.y1, det.x2, det.y2, rel
            )


def _release_camera() -> None:
    cap = st.session_state.get("webcam_capture")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.webcam_capture = None
    st.session_state.webcam_running = False


def render_webcam(root: Path) -> None:
    st.title("📷 Webcam")
    st.caption(
        "Modo local via OpenCV. Durante a apresentação, a câmera pode apontar para "
        "um vídeo de incêndio reproduzido em outro dispositivo."
    )

    if not st.session_state.detector.available:
        st.warning("Modelo personalizado ainda não treinado. A câmera abre, mas a IA não detectará Fire/Smoke.")

    c1, c2 = st.columns(2)
    start = c1.button("▶ Iniciar câmera", type="primary", use_container_width=True, disabled=st.session_state.webcam_running)
    stop = c2.button("⏹ Parar câmera", use_container_width=True, disabled=not st.session_state.webcam_running)

    if start:
        _release_camera()
        cap = cv2.VideoCapture(int(st.session_state.webcam_index), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(int(st.session_state.webcam_index))
        if not cap.isOpened():
            st.error("Não foi possível abrir a webcam. Verifique o índice e a permissão da câmera no Windows.")
            return
        st.session_state.webcam_capture = cap
        st.session_state.webcam_running = True
        st.rerun()

    if stop:
        _release_camera()
        st.success("Câmera encerrada.")
        return

    if not st.session_state.webcam_running:
        st.info("Clique em **Iniciar câmera** para começar.")
        return

    cap = st.session_state.webcam_capture
    if cap is None or not cap.isOpened():
        _release_camera()
        st.error("A captura da webcam foi perdida.")
        return

    tick = time.perf_counter()
    ok, original = cap.read()
    if not ok:
        _release_camera()
        st.error("Não foi possível ler um frame da webcam.")
        return

    processed, _ = _preprocessor().process(original)
    detector = st.session_state.detector
    detector.confidence = st.session_state.confidence_threshold
    detections = detector.detect(processed)
    annotated = detector.draw_detections(processed, detections)
    _save_occurrence(root, annotated, detections)

    fire = [d for d in detections if d.class_name == "Fire"]
    smoke = [d for d in detections if d.class_name == "Smoke"]
    best_conf = max((d.confidence for d in detections), default=0.0)
    class_names = {d.class_name for d in detections}

    if "Fire" in class_names:
        st.error("🔴 ALERTA — FOGO DETECTADO")
        st.session_state.alert_manager.play_sound("Fire")
    elif "Smoke" in class_names:
        st.warning("🟠 ATENÇÃO — FUMAÇA DETECTADA")
        st.session_state.alert_manager.play_sound("Smoke")
    else:
        st.success("🟢 Status normal")

    st.image(annotated, channels="BGR", use_container_width=True)
    if st.session_state.show_preprocess:
        st.markdown("#### ORIGINAL | PROCESSADA")
        a, b = st.columns(2)
        a.image(original, channels="BGR", caption="Original", use_container_width=True)
        b.image(processed, channels="BGR", caption="Processada", use_container_width=True)

    elapsed = time.perf_counter() - tick
    fps = 1.0 / max(elapsed, 1e-9)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("FPS", f"{fps:.1f}")
    m2.metric("Fire", len(fire))
    m3.metric("Smoke", len(smoke))
    m4.metric("Confiança", f"{best_conf:.1%}")
    m5.metric("Status", "ALERTA" if detections else "Normal")

    # Um frame por execução mantém os botões responsivos no modelo de rerun do Streamlit.
    time.sleep(0.04)
    st.rerun()
