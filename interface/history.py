from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils.logger import basic_statistics


def render_history(root: Path) -> None:
    st.title("📊 Histórico e dashboard")
    df = st.session_state.logger.load()

    if df.empty:
        st.info("Nenhuma ocorrência registrada ainda.")
        _show_runtime_stats()
        return

    f1, f2 = st.columns(2)
    classes = f1.multiselect("Classe", ["Fire", "Smoke"], default=["Fire", "Smoke"])
    sources = f2.multiselect("Origem", ["video", "webcam"], default=["video", "webcam"])

    filtered = df[df["class_name"].isin(classes) & df["source"].isin(sources)].copy()
    stats = basic_statistics(filtered)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", stats["total"])
    c2.metric("Fire", stats["fire"])
    c3.metric("Smoke", stats["smoke"])
    c4.metric("Conf. média", f"{stats['mean_confidence']:.1%}")
    c5.metric("Maior conf.", f"{stats['max_confidence']:.1%}")

    _show_runtime_stats()

    if filtered.empty:
        st.warning("Os filtros atuais não retornaram ocorrências.")
        return

    st.subheader("Detecções por classe")
    counts = filtered["class_name"].value_counts().reindex(["Fire", "Smoke"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Quantidade")
    ax.set_xlabel("Classe")
    ax.set_title("Ocorrências registradas")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Registros")
    table = filtered.copy()
    table["Data"] = table["timestamp"].dt.strftime("%d/%m/%Y")
    table["Hora"] = table["timestamp"].dt.strftime("%H:%M:%S")
    table["Confiança"] = table["confidence"].map(lambda x: f"{x:.1%}")
    table["Origem"] = table["source"].replace({"video": "Vídeo", "webcam": "Webcam"})
    table["Classe"] = table["class_name"]
    st.dataframe(
        table[["Data", "Hora", "Origem", "Classe", "Confiança", "image_path"]],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Baixar histórico filtrado (CSV)",
        data=csv_bytes,
        file_name="firewatch_historico.csv",
        mime="text/csv",
    )


def _show_runtime_stats() -> None:
    runtime = st.session_state.get("video_stats", {})
    c1, c2 = st.columns(2)
    c1.metric("Tempo último vídeo", f"{float(runtime.get('processing_time', 0.0)):.2f}s")
    c2.metric("FPS médio último vídeo", f"{float(runtime.get('average_fps', 0.0)):.2f}")
