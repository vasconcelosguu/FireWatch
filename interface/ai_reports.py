from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ai.prompts import PROMPT_VERSION
from ai.report_generator import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    ReportGenerationError,
    generate_report,
    payload_fingerprint,
    prepare_report_data,
    render_report_markdown,
    render_report_pdf,
    verify_groq_connection,
)


def _load_dotenv(root: Path) -> None:
    load_dotenv(root / ".env", override=False)


def _group_detections(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Agrupa linhas pela evidência salva; paths são usados somente dentro da aplicação."""
    groups: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        timestamp = row.get("timestamp")
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(timestamp) else "horário-não-informado"
        source_value = row.get("source")
        source = str(source_value).strip() if pd.notna(source_value) and str(source_value).strip() else "origem-não-informada"
        image_value = row.get("image_path")
        image_path = str(image_value).strip() if pd.notna(image_value) else ""
        evidence_key = image_path if image_path and image_path.lower() != "nan" else timestamp_text
        group_key = f"{source}|{evidence_key}"
        groups.setdefault(group_key, []).append(row.to_dict())
    return {key: pd.DataFrame(rows) for key, rows in groups.items()}


def _make_selected_records(groups: dict[str, pd.DataFrame], selected: list[str]) -> list[dict]:
    """Adiciona um código interno de agrupamento sem exportar o caminho da evidência."""
    records: list[dict] = []
    for number, key in enumerate(sorted(selected), start=1):
        group = groups[key]
        for row in group.to_dict(orient="records"):
            row["_report_group"] = f"G{number:03d}"
            records.append(row)
    return records


def _render_class_metrics(payload: dict) -> None:
    stats = payload.get("class_statistics", {})
    if not stats:
        st.info("Não há classes reconhecidas nos registros selecionados.")
        return
    columns = st.columns(max(1, len(stats)))
    for column, (class_name, details) in zip(columns, stats.items()):
        score = details["confidence_score"]
        mean = score["mean"]
        column.metric(class_name, details["detections"], f"Confiança média: {mean:.1%}" if mean is not None else "Confiança indisponível")


def _render_report_visual_summary(payload: dict) -> None:
    """Apresenta indicadores e gráficos locais sem alterar os dados enviados à Groq."""
    scope = payload.get("scope", {})
    overall = payload.get("overall_confidence_score", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detecções", f"{scope.get('total_detections', 0):,}".replace(",", "."))
    c2.metric("Grupos de evidência", f"{scope.get('grouped_evidence_records', 0):,}".replace(",", "."))
    mean = overall.get("mean")
    c3.metric("Confiança média YOLO", f"{mean:.1%}" if mean is not None else "—")
    c4.metric("Origens", len(scope.get("sources", [])))

    left, right = st.columns(2)
    with left:
        st.markdown("**Distribuição por classe**")
        stats = payload.get("class_statistics", {})
        if stats:
            chart = pd.DataFrame({"Detecções": {name: values.get("detections", 0) for name, values in stats.items()}})
            st.bar_chart(chart, height=220, use_container_width=True)
        else:
            st.info("Sem dados de classe para exibir.")
    with right:
        st.markdown("**Confiança média por grupo**")
        points = []
        for event in payload.get("event_details", []):
            weighted_sum = 0.0
            weighted_count = 0
            for details in event.get("classes", {}).values():
                score = details.get("confidence_score", {}).get("mean")
                count = details.get("detections", 0)
                if score is not None and count:
                    weighted_sum += float(score) * int(count)
                    weighted_count += int(count)
            if weighted_count:
                points.append({
                    "Grupo": event.get("group_code", "Grupo"),
                    "Confiança média": weighted_sum / weighted_count,
                })
        if points:
            trend = pd.DataFrame(points).set_index("Grupo")
            st.line_chart(trend, height=220, use_container_width=True)
        else:
            st.info("Não há confiança válida para montar o gráfico.")
    st.caption("A confiança é uma pontuação interna do YOLO; não equivale à probabilidade de incêndio real ou à gravidade do evento.")


def render_ai_reports(root: Path) -> None:
    _load_dotenv(root)
    st.title("🧠 FireWatch AI — Relatórios Inteligentes")
    st.markdown(
        "A camada generativa utiliza um LLM da Groq para interpretar estatísticas e registros "
        "produzidos pelo YOLO e redigir um relatório técnico. **A detecção continua sendo feita "
        "pelo YOLO; a IA generativa não analisa a mídia nem confirma incêndios.**"
    )

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    configured = bool(api_key and api_key != "coloque_sua_chave_aqui")
    if configured:
        st.success("Chave Groq encontrada no ambiente. Nenhuma chamada de geração é feita automaticamente.")
    else:
        st.warning("Configure GROQ_API_KEY no arquivo .env para habilitar os relatórios generativos.")
        with st.expander("Como configurar a chave Groq"):
            st.code("# Na raiz do projeto, copie .env.example para .env\nGROQ_API_KEY=sua_chave_real\nFIREWATCH_GROQ_MODEL=openai/gpt-oss-120b", language="bash")
            st.caption("Não compartilhe o arquivo .env nem envie a chave ao GitHub. A chave não deve ser colada na interface ou no código-fonte.")

    configured_model = os.getenv("FIREWATCH_GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    model_options = list(AVAILABLE_MODELS)
    if configured_model not in model_options:
        model_options.insert(0, configured_model)
    left, right = st.columns([2, 1])
    with left:
        model = st.selectbox(
            "Modelo generativo",
            options=model_options,
            index=model_options.index(configured_model) if configured_model in model_options else 0,
            format_func=lambda value: AVAILABLE_MODELS.get(value, f"{value} — modelo informado no .env"),
            help="O GPT-OSS 120B é o padrão para priorizar qualidade de análise. O 20B é uma alternativa mais leve.",
        )
    with right:
        st.write("")
        st.write("")
        if st.button("🔌 Testar conexão", disabled=not configured, use_container_width=True):
            try:
                with st.spinner("Consultando a API Groq..."):
                    ok, message = verify_groq_connection(model=model)
                (st.success if ok else st.warning)(message)
            except ReportGenerationError as exc:
                st.error(str(exc))

    st.divider()
    df = st.session_state.logger.load()
    if df.empty:
        st.info("Ainda não existem detecções no histórico. Execute um vídeo ou a webcam para registrar dados.")
        return

    groups = _group_detections(df)
    labels: dict[str, str] = {}
    ordered_keys = sorted(
        groups,
        key=lambda key: str(groups[key].iloc[0].get("timestamp", "")),
        reverse=True,
    )
    for key in ordered_keys:
        group = groups[key]
        first = group.iloc[0]
        ts = first.get("timestamp")
        ts_label = ts.strftime("%d/%m/%Y %H:%M:%S") if pd.notna(ts) else "Data não informada"
        source = str(first.get("source", "Origem não informada"))
        counts = group["class_name"].fillna("Não informada").value_counts().to_dict()
        classes_label = ", ".join(f"{name}: {count}" for name, count in counts.items())
        labels[key] = f"{ts_label} | {source} | {classes_label}"

    # Filtros locais para facilitar a seleção quando o histórico contém muitos grupos.
    st.markdown("### Seleção de evidências")
    st.caption(f"{len(ordered_keys)} grupos disponíveis no histórico. Selecione os grupos que deseja incluir no relatório.")

    filter_col, class_col = st.columns([2, 1])
    with filter_col:
        search_text = st.text_input(
            "Buscar por data, horário ou origem",
            placeholder="Ex.: 09/11/2026, 21:59 ou vídeo",
            key="fw_report_group_search",
        )
    all_classes = sorted({
        str(class_name)
        for group in groups.values()
        for class_name in group["class_name"].dropna().unique()
    })
    with class_col:
        class_filter = st.selectbox(
            "Filtrar classe",
            ["Todas as classes", *all_classes],
            key="fw_report_class_filter",
        )

    filtered_keys = []
    for key in ordered_keys:
        group = groups[key]
        searchable = labels[key].casefold()
        if search_text.strip() and search_text.strip().casefold() not in searchable:
            continue
        if class_filter != "Todas as classes" and class_filter not in set(group["class_name"].dropna().astype(str)):
            continue
        filtered_keys.append(key)

    selected_key = "fw_report_selected_groups"
    if selected_key not in st.session_state:
        st.session_state[selected_key] = []
    # Remove seleções que deixaram de existir no histórico.
    st.session_state[selected_key] = [
        key for key in st.session_state[selected_key] if key in ordered_keys
    ]

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
    with action_col1:
        if st.button("Selecionar filtrados", use_container_width=True, disabled=not filtered_keys):
            st.session_state[selected_key] = list(dict.fromkeys(st.session_state[selected_key] + filtered_keys))
            st.rerun()
    with action_col2:
        if st.button("Limpar seleção", use_container_width=True, disabled=not st.session_state[selected_key]):
            st.session_state[selected_key] = []
            st.rerun()
    with action_col3:
        st.caption(f"Exibindo {len(filtered_keys)} de {len(ordered_keys)} grupos")

    # Mantém visíveis os grupos já selecionados mesmo quando um filtro os oculta.
    widget_options = list(dict.fromkeys(filtered_keys + st.session_state[selected_key]))
    selected = st.multiselect(
        "Grupos selecionados",
        options=widget_options,
        format_func=lambda value: labels.get(value, "Grupo de detecções"),
        key=selected_key,
        placeholder="Pesquise e selecione grupos...",
        help="Cada opção representa um agrupamento técnico de registros associados à mesma evidência. Não representa um incêndio confirmado.",
    )
    st.caption(f"{len(selected)} grupo(s) selecionado(s) • {sum(len(groups[key]) for key in selected)} registro(s) de detecção")

    if not selected:
        st.info("Selecione pelo menos um grupo para visualizar as estatísticas e gerar o relatório.")
        return

    records = _make_selected_records(groups, selected)
    selection_hash = payload_fingerprint({"selected_groups": sorted(selected)})
    report_id = "FW-" + selection_hash[:10].upper()
    payload = prepare_report_data(records, group_id=report_id)
    fingerprint = payload_fingerprint({
        "payload": payload,
        "model": model,
        "prompt_version": PROMPT_VERSION,
    })

    st.markdown("### Resumo da seleção")
    _render_report_visual_summary(payload)
    _render_class_metrics(payload)

    with st.expander("Conferir dados que serão usados pela IA", expanded=False):
        preview = pd.DataFrame([
            {
                "Data/hora": row["timestamp"],
                "Origem": row["source"],
                "Classe": row["class_name"],
                "Confiança YOLO": row["confidence"],
            }
            for row in records
        ])
        st.dataframe(preview, use_container_width=True, hide_index=True)
        st.caption("O envio à Groq contém apenas estatísticas e metadados necessários. Imagens, caminhos locais, coordenadas e dados pessoais não são enviados.")

    if st.button(
        "✨ Gerar relatório técnico com IA",
        type="primary",
        use_container_width=True,
        disabled=not configured,
        key="generate_firewatch_ai_report",
    ):
        try:
            cache = st.session_state.setdefault("ai_report_cache", {})
            if fingerprint in cache:
                st.session_state["ai_current_report"] = cache[fingerprint]
                st.info("Relatório reutilizado do cache da sessão; nenhuma nova chamada foi feita à API.")
            else:
                with st.spinner("A IA está analisando os dados e redigindo o relatório técnico..."):
                    sections = generate_report(payload, model=model)
                    markdown_report = render_report_markdown(payload, sections)
                generated = {
                    "fingerprint": fingerprint,
                    "markdown": markdown_report,
                    "payload": payload,
                    "sections": sections,
                    "model": model,
                }
                cache[fingerprint] = generated
                st.session_state["ai_current_report"] = generated
        except ValueError as exc:
            st.error(str(exc))
        except ReportGenerationError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Ocorreu um erro inesperado ao gerar o relatório. O histórico original foi preservado.")

    current = st.session_state.get("ai_current_report")
    if current and current.get("fingerprint") == fingerprint:
        st.divider()
        st.subheader("Relatório gerado")
        st.caption(f"Modelo: {current['model']} • Prompt: {PROMPT_VERSION}")
        st.markdown(
            "<div class='fw-note'><b>Leitura responsável:</b> o relatório interpreta registros do YOLO. "
            "Ele não confirma a existência de incêndio e deve ser revisado por uma pessoa.</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Visualizar conteúdo textual completo", expanded=True):
            st.markdown(current["markdown"])
        filename = report_id.lower()
        col_txt, col_pdf = st.columns(2)
        with col_txt:
            st.download_button(
                "⬇️ Baixar relatório (.txt)",
                data=current["markdown"].encode("utf-8"),
                file_name=f"{filename}.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
            )
        with col_pdf:
            try:
                pdf_data = render_report_pdf(current["markdown"])
                st.download_button(
                    "⬇️ Baixar relatório (.pdf)",
                    data=pdf_data,
                    file_name=f"{filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.caption("Para habilitar o PDF, instale reportlab com: python -m pip install reportlab")
    elif current:
        st.info("Existe um relatório anterior na sessão. Gere novamente para a seleção e o modelo atuais.")

    with st.expander("Como a IA funciona e quais são seus limites"):
        st.markdown("\n".join([
            "- **YOLO:** executa a detecção visual de Fire e Smoke e produz classe, confiança e caixa delimitadora.",
            "- **LLM da Groq:** recebe estatísticas e metadados dos registros e redige resumo, análise, pontos de atenção, recomendações e conclusão.",
            "- **Sem análise de mídia:** nenhuma imagem ou vídeo é enviado à Groq. A IA generativa não consegue verificar visualmente a cena nesta integração.",
            "- **Sem confirmação automática:** uma detecção Fire/Smoke é um resultado do modelo e precisa ser verificada por uma pessoa.",
            "- **Privacidade:** caminhos locais, coordenadas e dados pessoais não são enviados.",
            "- **Custo/limite:** a API só é chamada quando o botão de geração é acionado; relatórios iguais são reutilizados no cache da sessão.",
        ]))
