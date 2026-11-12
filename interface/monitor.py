from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
from dotenv import load_dotenv

from ai.report_generator import (
    DEFAULT_MODEL, ReportGenerationError, generate_report, payload_fingerprint,
    prepare_report_data, render_report_markdown, render_report_pdf,
)

from detector.preprocess import ImagePreprocessor, PreprocessConfig
from utils.video import read_video_metadata, save_uploaded_video


def _preprocessor_from_state() -> ImagePreprocessor:
    cfg = PreprocessConfig(
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
    return ImagePreprocessor(cfg)


def _save_and_log(root: Path, frame, detections, source: str) -> list[str]:
    if not detections:
        return []

    now_mono = time.monotonic()
    saved: list[str] = []
    by_class: dict[str, list] = {}
    for det in detections:
        by_class.setdefault(det.class_name, []).append(det)

    for class_name, class_detections in by_class.items():
        key = f"{source}:{class_name}"
        last = st.session_state.last_screenshots.get(key, -1e9)
        if now_mono - last < st.session_state.screenshot_cooldown:
            continue

        st.session_state.last_screenshots[key] = now_mono
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{class_name.lower()}_{stamp}.jpg"
        path = root / "screenshots" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
        rel = path.relative_to(root).as_posix()
        saved.append(rel)

        for det in class_detections:
            st.session_state.logger.log(
                source=source,
                class_name=det.class_name,
                confidence=det.confidence,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                image_path=rel,
            )
    return saved


def render_monitor(root: Path) -> None:
    st.title("🎥 Monitoramento de vídeo")
    st.caption("Upload MP4, AVI ou MOV e processamento frame a frame com OpenCV + YOLO.")

    load_dotenv(root / ".env", override=False)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_configured = bool(api_key and api_key != "coloque_sua_chave_aqui")
    auto_summary = st.checkbox(
        "✨ Gerar resumo automático com IA ao terminar o vídeo",
        value=True,
        help="Ao concluir o processamento, a Groq receberá somente as estatísticas e os metadados das novas detecções. Nenhuma imagem ou vídeo será enviado.",
    )
    if auto_summary and not groq_configured:
        st.caption("Configure a chave Groq no .env para habilitar o resumo automático. O vídeo continuará sendo processado normalmente.")

    uploaded = st.file_uploader("Selecione um vídeo", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Envie um vídeo para iniciar o monitoramento.")
        return

    temp_path = save_uploaded_video(uploaded)
    try:
        metadata = read_video_metadata(temp_path)
    except Exception as exc:
        st.error(f"Falha ao abrir o vídeo: {exc}")
        temp_path.unlink(missing_ok=True)
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Largura", f"{metadata.width}px")
    m2.metric("Altura", f"{metadata.height}px")
    m3.metric("FPS original", f"{metadata.fps:.2f}")
    m4.metric("Frames", f"{metadata.frame_count}")

    if not st.session_state.detector.available:
        st.warning(
            "Modelo personalizado ainda não treinado. O vídeo e o pré-processamento funcionam, "
            "mas não haverá detecções até existir models/fire_model.pt."
        )

    if not st.button("▶ Processar vídeo", type="primary", use_container_width=True):
        temp_path.unlink(missing_ok=True)
        return

    detector = st.session_state.detector
    detector.confidence = st.session_state.confidence_threshold
    preprocessor = _preprocessor_from_state()

    initial_log_count = len(st.session_state.logger.load())
    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        st.error("OpenCV não conseguiu iniciar a leitura do vídeo.")
        temp_path.unlink(missing_ok=True)
        return

    progress = st.progress(0.0, text="Preparando processamento...")
    frame_slot = st.empty()
    stats_slot = st.empty()
    alert_slot = st.empty()
    compare_title = st.empty()
    original_slot = processed_slot = None
    if st.session_state.show_preprocess:
        compare_title.markdown("#### ORIGINAL | PROCESSADA")
        c1, c2 = st.columns(2)
        original_slot = c1.empty()
        processed_slot = c2.empty()

    frame_index = 0
    processed_count = 0
    sum_instant_fps = 0.0
    start_total = time.perf_counter()
    last_tick = start_total

    try:
        while True:
            ok, original = cap.read()
            if not ok:
                break

            frame_index += 1
            try:
                processed, _views = preprocessor.process(original)
            except Exception as exc:
                st.error(f"Erro no pré-processamento do frame {frame_index}: {exc}")
                break

            detections = detector.detect(processed)
            annotated = detector.draw_detections(processed, detections)

            class_names = {d.class_name for d in detections}
            if "Fire" in class_names:
                alert_slot.error("🔴 ALERTA — FOGO DETECTADO")
                st.session_state.alert_manager.play_sound("Fire")
            elif "Smoke" in class_names:
                alert_slot.warning("🟠 ATENÇÃO — FUMAÇA DETECTADA")
                st.session_state.alert_manager.play_sound("Smoke")
            else:
                alert_slot.success("🟢 Sem detecção acima do threshold")

            _save_and_log(root, annotated, detections, "video")
            frame_slot.image(annotated, channels="BGR", use_container_width=True)

            if original_slot is not None and processed_slot is not None:
                original_slot.image(original, channels="BGR", caption="Original", use_container_width=True)
                processed_slot.image(processed, channels="BGR", caption="Processada", use_container_width=True)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_tick, 1e-9)
            last_tick = now
            processed_count += 1
            sum_instant_fps += instant_fps
            stats_slot.markdown(
                f"**Frame:** {frame_index}/{metadata.frame_count or '?'} &nbsp;&nbsp; "
                f"**FPS atual:** {instant_fps:.2f} &nbsp;&nbsp; "
                f"**Detecções:** {len(detections)}"
            )

            if metadata.frame_count > 0:
                ratio = min(frame_index / metadata.frame_count, 1.0)
                progress.progress(ratio, text=f"Processando... {ratio * 100:.1f}%")

        elapsed = time.perf_counter() - start_total
        average_fps = processed_count / elapsed if elapsed > 0 else 0.0
        st.session_state.video_stats = {
            "processing_time": elapsed,
            "average_fps": average_fps,
        }
        progress.progress(1.0, text="Processamento concluído")
        st.success(
            f"Vídeo concluído: {processed_count} frames em {elapsed:.2f}s "
            f"(FPS médio de processamento: {average_fps:.2f})."
        )
        if auto_summary and groq_configured:
            try:
                new_records = st.session_state.logger.load().iloc[initial_log_count:].to_dict(orient="records")
                if new_records:
                    summary_payload = prepare_report_data(new_records, group_id="RESUMO-VIDEO")
                    model = os.getenv("FIREWATCH_GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
                    with st.spinner("Gerando resumo automático do vídeo com a IA generativa..."):
                        sections = generate_report(summary_payload, model=model)
                        markdown = render_report_markdown(summary_payload, sections)
                    st.session_state.video_ai_summary = {
                        "markdown": markdown,
                        "model": model,
                        "fingerprint": payload_fingerprint(summary_payload),
                    }
                    st.success("Resumo inteligente do vídeo gerado.")
                else:
                    st.info("O vídeo terminou sem novas detecções registradas. Não há dados para gerar um resumo.")
            except (ValueError, ReportGenerationError) as exc:
                st.warning(f"O vídeo foi processado, mas não foi possível gerar o resumo pela Groq: {exc}")
            except Exception:
                st.warning("O vídeo foi processado, mas ocorreu um erro inesperado ao gerar o resumo. O histórico foi preservado.")
        elif auto_summary and not groq_configured:
            st.info("Processamento concluído. Configure GROQ_API_KEY para gerar o resumo automático.")
    finally:
        cap.release()
        temp_path.unlink(missing_ok=True)
    summary = st.session_state.get("video_ai_summary")
    if summary:
        st.divider()
        st.subheader("📝 Resumo automático do último vídeo")
        st.caption(f"Modelo generativo: {summary['model']}")
        st.markdown(summary["markdown"])
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Baixar resumo (.txt)",
            data=summary["markdown"].encode("utf-8"),
            file_name="firewatch_resumo_video.txt",
            mime="text/plain; charset=utf-8",
            use_container_width=True,
        )
        try:
            pdf = render_report_pdf(summary["markdown"])
            d2.download_button(
                "⬇️ Baixar resumo (.pdf)", data=pdf,
                file_name="firewatch_resumo_video.pdf", mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            d2.caption("Instale reportlab para exportar PDF.")

