from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any

from ai.prompts import PROMPT_VERSION, REPORT_SCHEMA, SYSTEM_PROMPT, build_user_prompt

# O 120B é o modelo de maior capacidade da família GPT-OSS atualmente listado na Groq.
# O 20B fica disponível como alternativa mais leve/econômica.
DEFAULT_MODEL = "openai/gpt-oss-120b"
AVAILABLE_MODELS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B — maior capacidade de análise (recomendado)",
    "openai/gpt-oss-20b": "GPT-OSS 20B — alternativa mais leve e rápida",
}


class ReportGenerationError(RuntimeError):
    """Erro seguro e apresentável ao usuário durante a geração do relatório."""


def _clean(value: Any, fallback: str = "Não informado") -> str:
    if value is None:
        return fallback
    try:
        if value != value:  # NaN / NaT
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat", "null"}:
        return fallback
    return text


def _safe_class(value: Any) -> str:
    normalized = _clean(value, "").strip().lower()
    if normalized == "fire":
        return "Fire"
    if normalized == "smoke":
        return "Smoke"
    return "Classe não reconhecida"


def _safe_source(value: Any) -> str:
    normalized = _clean(value, "").strip().lower()
    allowed = {"video": "Vídeo", "webcam": "Webcam", "image": "Imagem"}
    return allowed.get(normalized, "Origem não reconhecida")


def _timestamp_text(value: Any) -> str:
    raw = _clean(value)
    if raw == "Não informado":
        return raw
    if hasattr(value, "to_pydatetime"):
        try:
            value = value.to_pydatetime()
        except Exception:
            pass
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    # Normalize only valid ISO-like timestamps; reject arbitrary CSV text.
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return "Não informado"


def _confidence_values(records: list[dict]) -> list[float]:
    values: list[float] = []
    for row in records:
        try:
            value = float(row.get("confidence"))
            if 0 <= value <= 1:
                values.append(value)
        except (TypeError, ValueError):
            continue
    return values


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"valid_values": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "valid_values": len(values),
        "mean": round(sum(values) / len(values), 4),
        "minimum": round(min(values), 4),
        "maximum": round(max(values), 4),
    }


def _event_key(row: dict, index: int) -> str:
    explicit = _clean(row.get("_report_group"), "")
    if explicit:
        return explicit
    # Legacy records: the evidence path is used only internally as a grouping key.
    # It is never included in the payload sent to Groq.
    evidence = _clean(row.get("image_path"), "")
    if evidence:
        return f"evidence:{evidence}"
    timestamp = _timestamp_text(row.get("timestamp"))
    source = _safe_source(row.get("source"))
    if timestamp != "Não informado":
        return f"legacy:{source}:{timestamp}"
    return f"legacy:record:{index}"


def prepare_report_data(
    records: list[dict],
    group_id: str = "Não informado",
    max_event_details: int = 60,
) -> dict:
    """Calcula estatísticas e cria payload seguro. Imagens, paths e coordenadas ficam fora."""
    if not records:
        raise ValueError("Não há detecções selecionadas para gerar o relatório.")

    class_confidences: dict[str, list[float]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    sources: set[str] = set()
    timestamps: list[str] = []
    grouped: dict[str, list[dict]] = defaultdict(list)

    for index, row in enumerate(records):
        class_name = _safe_class(row.get("class_name"))
        source = _safe_source(row.get("source"))
        timestamp = _timestamp_text(row.get("timestamp"))
        confidence = _confidence_values([row])
        class_counts[class_name] += 1
        class_confidences[class_name].extend(confidence)
        sources.add(source)
        if timestamp != "Não informado":
            timestamps.append(timestamp)

        grouped[_event_key(row, index)].append({
            "class_name": class_name,
            "source": source,
            "timestamp": timestamp,
            "confidence": confidence[0] if confidence else None,
        })

    valid_times = sorted(timestamps)
    class_statistics = {}
    for class_name in sorted(class_counts):
        class_statistics[class_name] = {
            "detections": int(class_counts[class_name]),
            "confidence_score": _stats(class_confidences[class_name]),
        }

    event_summaries: list[dict] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: min(
            (record["timestamp"] for record in item[1] if record["timestamp"] != "Não informado"),
            default="9999",
        ),
    )
    for number, (_, rows) in enumerate(ordered_groups, start=1):
        event_classes: dict[str, list[float]] = defaultdict(list)
        event_counts: Counter[str] = Counter()
        event_times = [r["timestamp"] for r in rows if r["timestamp"] != "Não informado"]
        event_sources = sorted({r["source"] for r in rows})
        for row in rows:
            event_counts[row["class_name"]] += 1
            if row["confidence"] is not None:
                event_classes[row["class_name"]].append(row["confidence"])
        event_summaries.append({
            "group_code": f"G{number:03d}",
            "timestamp_start": min(event_times) if event_times else "Não informado",
            "timestamp_end": max(event_times) if event_times else "Não informado",
            "sources": event_sources,
            "detections": len(rows),
            "classes": {
                name: {
                    "detections": count,
                    "confidence_score": _stats(event_classes[name]),
                }
                for name, count in sorted(event_counts.items())
            },
        })

    if max_event_details <= 0:
        details = []
    elif len(event_summaries) <= max_event_details:
        details = event_summaries
    elif max_event_details == 1:
        details = [event_summaries[0]]
    else:
        # Amostra distribuída ao longo de todo o período, não apenas os eventos mais antigos.
        sample_indexes = sorted({
            round(index * (len(event_summaries) - 1) / (max_event_details - 1))
            for index in range(max_event_details)
        })
        details = [event_summaries[index] for index in sample_indexes]
    period_start = valid_times[0] if valid_times else "Não informado"
    period_end = valid_times[-1] if valid_times else "Não informado"

    return {
        "report_id": _clean(group_id),
        "scope": {
            "total_detections": len(records),
            "grouped_evidence_records": len(event_summaries),
            "period_start": period_start,
            "period_end": period_end,
            "sources": sorted(sources),
        },
        "class_statistics": class_statistics,
        "overall_confidence_score": {
            **_stats(_confidence_values(records)),
            "meaning": "Pontuação de confiança interna do detector YOLO; não é probabilidade calibrada de incêndio real nem indicador isolado de gravidade.",
        },
        "event_details": details,
        "event_details_included": len(details),
        "event_details_omitted": max(0, len(event_summaries) - len(details)),
        "event_details_sampling": "Amostra distribuída cronologicamente" if len(event_summaries) > len(details) else "Todos os grupos incluídos",
        "data_limitations": [
            "Os registros representam classificações produzidas pelo YOLO e não confirmação visual de um evento real.",
            "Localização geográfica e identificação do ambiente não foram registradas nos dados fornecidos.",
            "Nenhuma imagem, vídeo, coordenada ou caminho de arquivo é enviado à IA generativa.",
            "Os agrupamentos representam evidências registradas no histórico e não necessariamente incidentes distintos.",
            "As estatísticas abrangem todos os registros selecionados; quando há muitos grupos, os detalhes individuais enviados à IA são amostrados ao longo do período.",
        ],
    }


def payload_fingerprint(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_ai_sections(data: Any) -> dict:
    required = {
        "executive_summary", "technical_analysis", "temporal_analysis",
        "attention_points", "recommended_actions", "limitations", "conclusion",
    }
    if not isinstance(data, dict) or not required.issubset(data):
        raise ReportGenerationError("A IA retornou uma estrutura de relatório incompleta. Gere novamente.")

    def text_field(name: str) -> str:
        value = data.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ReportGenerationError("A IA retornou uma seção vazia ou inválida. Gere novamente.")
        return value.strip()[:5000]

    def list_field(name: str) -> list[str]:
        value = data.get(name)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ReportGenerationError("A IA retornou uma lista de relatório inválida. Gere novamente.")
        return [item.strip()[:1000] for item in value[:8]]

    return {
        "executive_summary": text_field("executive_summary"),
        "technical_analysis": text_field("technical_analysis"),
        "temporal_analysis": text_field("temporal_analysis"),
        "attention_points": list_field("attention_points"),
        "recommended_actions": list_field("recommended_actions"),
        "limitations": list_field("limitations"),
        "conclusion": text_field("conclusion"),
    }


def _create_client(api_key: str):
    try:
        from groq import Groq
    except ImportError as exc:
        raise ReportGenerationError(
            "Biblioteca Groq ausente. Execute: python -m pip install -r requirements.txt"
        ) from exc
    return Groq(api_key=api_key, timeout=60.0, max_retries=1)


def generate_report(payload: dict, client=None, model: str | None = None) -> dict:
    """Gera as seções do relatório com Groq Structured Outputs (JSON Schema estrito)."""
    if not payload or int(payload.get("scope", {}).get("total_detections", 0)) <= 0:
        raise ValueError("Não há detecções válidas para gerar o relatório.")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if client is None:
        if not api_key or api_key == "coloque_sua_chave_aqui":
            raise ReportGenerationError(
                "Chave GROQ_API_KEY não configurada. Crie um arquivo .env a partir do .env.example e informe sua chave."
            )
        client = _create_client(api_key)

    selected_model = (model or os.getenv("FIREWATCH_GROQ_MODEL") or DEFAULT_MODEL).strip()
    if not selected_model:
        raise ReportGenerationError("Informe um identificador de modelo válido da Groq.")

    try:
        request_options = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(payload)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "firewatch_technical_report",
                    "strict": True,
                    "schema": REPORT_SCHEMA,
                },
            },
            "max_completion_tokens": 3200,
        }
        if "gpt-oss" in selected_model:
            request_options["reasoning_effort"] = "medium"
        response = client.chat.completions.create(**request_options)
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ReportGenerationError("A API retornou um relatório vazio. Tente novamente.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ReportGenerationError("A IA retornou um formato inesperado. Gere o relatório novamente.") from exc
        return _validate_ai_sections(parsed)
    except ReportGenerationError:
        raise
    except Exception as exc:
        name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", None)
        if "authentication" in name or status_code == 401:
            message = "Falha de autenticação na Groq. Confira se a chave GROQ_API_KEY está correta e ativa."
        elif "permission" in name or status_code == 403:
            message = "A chave foi reconhecida, mas não tem permissão para usar este recurso/modelo na Groq."
        elif "ratelimit" in name or "rate_limit" in name or status_code == 429:
            message = "Limite de requisições da Groq atingido. Aguarde a renovação do limite e tente novamente."
        elif "connection" in name or "timeout" in name:
            message = "Não foi possível conectar à Groq. Verifique a conexão com a internet e tente novamente."
        elif "notfound" in name or status_code == 404:
            message = "O modelo configurado não foi encontrado ou não está disponível para sua conta. Confira o ID do modelo."
        elif "badrequest" in name or status_code == 400:
            message = "A Groq rejeitou o formato da solicitação. Verifique o modelo selecionado e tente novamente."
        else:
            message = "Não foi possível gerar o relatório pela Groq. Tente novamente mais tarde."
        raise ReportGenerationError(message) from exc



def verify_groq_connection(client=None, model: str | None = None) -> tuple[bool, str]:
    """Verifica a chave e a disponibilidade do modelo sem gerar texto nem consumir tokens."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if client is None:
        if not api_key or api_key == "coloque_sua_chave_aqui":
            raise ReportGenerationError("Chave GROQ_API_KEY não configurada no arquivo .env.")
        client = _create_client(api_key)
    selected_model = (model or os.getenv("FIREWATCH_GROQ_MODEL") or DEFAULT_MODEL).strip()
    try:
        response = client.models.list()
        available = {getattr(item, "id", "") for item in getattr(response, "data", [])}
        if selected_model and available and selected_model not in available:
            return False, f"A chave funcionou, mas o modelo {selected_model} não apareceu na lista de modelos disponíveis para sua conta."
        return True, f"Conexão com a Groq validada. Modelo selecionado: {selected_model}."
    except Exception as exc:
        name = type(exc).__name__.lower()
        status_code = getattr(exc, "status_code", None)
        if "authentication" in name or status_code == 401:
            message = "A Groq recusou a chave. Confira se ela foi copiada corretamente e continua ativa."
        elif "ratelimit" in name or status_code == 429:
            message = "Limite de consultas da API atingido. Tente novamente mais tarde."
        elif "connection" in name or "timeout" in name:
            message = "Não foi possível acessar a Groq. Verifique a conexão com a internet."
        else:
            message = "Não foi possível validar a chave ou consultar os modelos disponíveis."
        raise ReportGenerationError(message) from exc


def render_report_markdown(payload: dict, ai_sections: dict) -> str:
    """Monta o relatório final com estatísticas calculadas localmente + análise generativa."""
    scope = payload["scope"]
    report_id = payload.get("report_id", "Não informado")
    sources = ", ".join(scope.get("sources") or []) or "Não informado"
    lines = [
        "# FireWatch AI — Relatório Técnico Inteligente",
        "",
        f"**Identificador do relatório:** {report_id}",
        "**Período observado:** " + (
            scope.get("period_start", "Não informado")
            if scope.get("period_start") == scope.get("period_end")
            else f"{scope.get('period_start', 'Não informado')} até {scope.get('period_end', 'Não informado')}"
        ),
        f"**Origem(ns):** {sources}",
        f"**Grupos de evidências selecionados:** {scope.get('grouped_evidence_records', 0)}",
        f"**Total de registros de detecção:** {scope.get('total_detections', 0)}",
        "",
        "## 1. Resumo executivo",
        ai_sections["executive_summary"],
        "",
        "## 2. Resultado quantitativo por classe",
        "| Classe | Registros | Confiança média | Mínima | Máxima |",
        "|---|---:|---:|---:|---:|",
    ]

    for class_name, stats in payload.get("class_statistics", {}).items():
        score = stats["confidence_score"]
        def fmt(value):
            return f"{value * 100:.2f}%" if value is not None else "Não informado"
        lines.append(
            f"| {class_name} | {stats['detections']} | {fmt(score['mean'])} | "
            f"{fmt(score['minimum'])} | {fmt(score['maximum'])} |"
        )

    overall = payload.get("overall_confidence_score", {})
    overall_text = (
        f"Média: {overall['mean'] * 100:.2f}%; mínima: {overall['minimum'] * 100:.2f}%; "
        f"máxima: {overall['maximum'] * 100:.2f}%"
        if overall.get("mean") is not None else "Não informado"
    )
    lines += [
        "",
        f"**Pontuação geral do YOLO:** {overall_text}.",
        "> As pontuações de confiança são valores internos do detector e não representam probabilidade calibrada de incêndio real ou gravidade.",
        "",
        "## 3. Grupos de evidência selecionados",
        "> Os grupos abaixo representam agrupamentos do histórico, não incidentes confirmados.",
        "| Grupo | Data/hora | Origem(ns) | Registros | Classes e pontuação média |",
        "|---|---|---|---:|---|",
    ]
    for event in payload.get("event_details", []):
        start_time = event.get("timestamp_start", "Não informado")
        end_time = event.get("timestamp_end", "Não informado")
        timestamp_label = start_time if start_time == end_time else f"{start_time} até {end_time}"
        class_labels = []
        for class_name, class_data in event.get("classes", {}).items():
            score = class_data.get("confidence_score", {}).get("mean")
            score_label = f"média {score * 100:.2f}%" if score is not None else "confiança não informada"
            class_labels.append(f"{class_name}: {class_data.get('detections', 0)} ({score_label})")
        lines.append(
            f"| {event.get('group_code', 'Não informado')} | {timestamp_label} | "
            f"{', '.join(event.get('sources', [])) or 'Não informado'} | "
            f"{event.get('detections', 0)} | {'; '.join(class_labels) or 'Não informado'} |"
        )
    if payload.get("event_details_omitted", 0):
        lines.append(
            f"*Foram exibidos {payload.get('event_details_included', len(payload.get('event_details', [])))} grupos em amostra cronológica; "
            f"{payload['event_details_omitted']} grupos adicionais foram considerados nas estatísticas agregadas.*"
        )
    lines += [
        "",
        "## 4. Análise técnica",
        ai_sections["technical_analysis"],
        "",
        "## 5. Análise temporal",
        ai_sections["temporal_analysis"],
        "",
        "## 6. Pontos de atenção",
    ]
    lines.extend([f"- {item}" for item in ai_sections["attention_points"]] or ["- Não foram informados pontos adicionais."])
    lines += ["", "## 7. Recomendações de verificação humana"]
    lines.extend([f"{index}. {item}" for index, item in enumerate(ai_sections["recommended_actions"], start=1)] or ["- Não foram geradas recomendações adicionais."])
    lines += ["", "## 8. Limitações e qualidade dos dados"]
    combined_limitations = list(dict.fromkeys(payload.get("data_limitations", []) + ai_sections["limitations"]))
    lines.extend([f"- {item}" for item in combined_limitations] or ["- Não informado."])
    lines += [
        "",
        "## 9. Conclusão",
        ai_sections["conclusion"],
        "",
        "---",
        "**Aviso:** este documento foi redigido por IA generativa com base em metadados e estatísticas produzidos pelo FireWatch AI. As detecções precisam ser verificadas por uma pessoa responsável na mídia original. Este relatório não confirma incêndio e não substitui procedimentos de segurança ou serviços de emergência.",
    ]
    return "\n".join(lines)


def render_report_pdf(report: str) -> bytes:
    """Gera PDF A4 com identidade visual, indicadores, tabelas e paginação legível."""
    import re
    from pathlib import Path
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle, Flowable,
    )

    # Use a Unicode-capable system font when available (Arial on Windows,
    # DejaVu on Linux). This avoids missing Portuguese punctuation/glyphs.
    font_regular = None
    font_bold = None
    font_candidates = [
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for regular_path, bold_path in font_candidates:
        if Path(regular_path).exists() and Path(bold_path).exists():
            try:
                if "FWUnicode" not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont("FWUnicode", regular_path))
                    pdfmetrics.registerFont(TTFont("FWUnicode-Bold", bold_path))
                font_regular, font_bold = "FWUnicode", "FWUnicode-Bold"
                break
            except Exception:
                continue
    if not font_regular:
        font_regular, font_bold = "Helvetica", "Helvetica-Bold"

    navy = colors.HexColor("#142033")
    navy_soft = colors.HexColor("#243449")
    orange = colors.HexColor("#E87532")
    orange_pale = colors.HexColor("#FFF2E9")
    text = colors.HexColor("#263445")
    muted = colors.HexColor("#65758B")
    line = colors.HexColor("#DCE3EA")
    pale = colors.HexColor("#F4F7FA")
    white = colors.white

    buffer = BytesIO()
    page_w, page_h = A4
    left_margin = 17 * mm
    right_margin = 17 * mm
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=right_margin, leftMargin=left_margin,
        topMargin=19 * mm, bottomMargin=19 * mm,
        title="FireWatch AI - Relatório Técnico Inteligente",
        author="FireWatch AI",
    )
    width = page_w - left_margin - right_margin
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("FWTitle", parent=base["Title"], fontName=font_bold,
            fontSize=20, leading=24, alignment=TA_LEFT, textColor=navy, spaceAfter=3),
        "subtitle": ParagraphStyle("FWSubtitle", parent=base["Normal"], fontName=font_regular,
            fontSize=8, leading=11, textColor=muted, spaceAfter=0),
        "h2": ParagraphStyle("FWH2", parent=base["Heading2"], fontName=font_bold,
            fontSize=11.5, leading=15, textColor=navy, spaceBefore=13, spaceAfter=6, keepWithNext=True),
        "body": ParagraphStyle("FWBody", parent=base["BodyText"], fontName=font_regular,
            fontSize=8.6, leading=13.2, textColor=text, spaceAfter=6),
        "small": ParagraphStyle("FWSmall", parent=base["BodyText"], fontName=font_regular,
            fontSize=7.2, leading=9.4, textColor=muted, spaceAfter=3),
        "table": ParagraphStyle("FWTable", parent=base["BodyText"], fontName=font_regular,
            fontSize=6.8, leading=8.3, textColor=text, spaceAfter=0),
        "table_head": ParagraphStyle("FWTableHead", parent=base["BodyText"], fontName=font_bold,
            fontSize=6.8, leading=8.3, textColor=white, spaceAfter=0),
        "callout": ParagraphStyle("FWCallout", parent=base["BodyText"], fontName=font_regular,
            fontSize=8, leading=11.5, textColor=navy, spaceAfter=0),
        "bullet": ParagraphStyle("FWBullet", parent=base["BodyText"], fontName=font_regular,
            fontSize=8.2, leading=12, leftIndent=10, firstLineIndent=-7,
            textColor=text, spaceAfter=3),
    }

    class ConfidenceTrend(Flowable):
        """Compact chart of the detector's mean score by evidence group."""
        def __init__(self, points, chart_width):
            super().__init__()
            self.points = points
            self.width = chart_width
            self.height = 43 * mm

        def wrap(self, available_width, available_height):
            self.width = min(self.width, available_width)
            return self.width, self.height

        def draw(self):
            c = self.canv
            c.saveState()
            c.setFillColor(navy)
            c.setFont(font_bold, 8)
            c.drawString(0, self.height - 9, "Confiança média por grupo")
            c.setFont(font_regular, 6.5)
            c.setFillColor(muted)
            c.drawRightString(self.width, self.height - 9, "Pontuação YOLO (não é probabilidade de incêndio)")
            x0, x1 = 25, self.width - 8
            y0, y1 = 12, self.height - 20
            values = [max(0.0, min(1.0, float(v))) for _, v in self.points]
            for tick in (0.5, 0.6, 0.7, 0.8, 0.9):
                y = y0 + (tick - 0.45) / 0.5 * (y1 - y0)
                c.setStrokeColor(line)
                c.setLineWidth(.45)
                c.line(x0, y, x1, y)
                c.setFillColor(muted)
                c.setFont(font_regular, 6)
                c.drawRightString(x0 - 4, y - 2, f"{tick:.0%}")
            if len(values) == 1:
                xs = [(x0 + x1) / 2]
            else:
                xs = [x0 + i * (x1 - x0) / (len(values) - 1) for i in range(len(values))]
            ys = [y0 + (v - 0.45) / 0.5 * (y1 - y0) for v in values]
            c.setStrokeColor(orange)
            c.setLineWidth(1.7)
            for idx in range(len(xs) - 1):
                c.line(xs[idx], ys[idx], xs[idx + 1], ys[idx + 1])
            for x, y in zip(xs, ys):
                c.setFillColor(orange)
                c.circle(x, y, 2, stroke=0, fill=1)
            for index in sorted(set([0, len(xs) // 2, len(xs) - 1])):
                c.setFillColor(muted)
                c.setFont(font_regular, 6)
                c.drawCentredString(xs[index], 2, self.points[index][0])
            c.restoreState()

    def add_page(canvas, document):
        canvas.saveState()
        canvas.setFillColor(orange)
        canvas.rect(0, page_h - 4 * mm, page_w, 4 * mm, stroke=0, fill=1)
        canvas.setStrokeColor(line)
        canvas.setLineWidth(0.6)
        canvas.line(left_margin, 13.5 * mm, page_w - right_margin, 13.5 * mm)
        canvas.setFont(font_regular, 7)
        canvas.setFillColor(muted)
        canvas.drawString(left_margin, 9 * mm, "FIREWATCH AI  |  RELATÓRIO TÉCNICO  |  REQUER VALIDAÇÃO HUMANA")
        canvas.drawRightString(page_w - right_margin, 9 * mm, f"Página {document.page}")
        canvas.restoreState()

    def inline(text_value: str) -> str:
        escaped = escape(text_value)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
        escaped = re.sub(r"\*(.+?)\*", r"<i>\1</i>", escaped)
        return escaped

    lines = report.splitlines()
    meta = {}
    body_start = 0
    for i, raw in enumerate(lines[:18]):
        match = re.match(r"\*\*(.+?):\*\*\s*(.*)", raw.strip())
        if match:
            meta[match.group(1).strip()] = match.group(2).strip()
            body_start = i + 1
        elif raw.startswith("## "):
            body_start = i
            break

    def value_for(label: str, default: str = "-") -> str:
        return meta.get(label, default) or default

    # The overall score is also shown as an explicitly labeled model score.
    confidence_text = "Não informado"
    for raw in lines:
        if "Pontuação geral do YOLO" in raw:
            match = re.search(r"Média:\s*([0-9]+(?:\.[0-9]+)?%)", raw)
            if match:
                confidence_text = match.group(1)
            break

    story = []
    brand = Table([[Paragraph("FIREWATCH <font color='#E87532'>AI</font>", ParagraphStyle(
        "FWBrand", fontName=font_bold, fontSize=17, leading=20, textColor=white)),
        Paragraph("MONITORAMENTO<br/><b>INTELIGENTE</b>", ParagraphStyle(
        "FWBrandRight", fontName=font_regular, fontSize=7, leading=9, alignment=2, textColor=colors.HexColor("#D7E0EA"))) ]],
        colWidths=[width * .63, width * .37])
    brand.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), navy), ("BOX", (0,0), (-1,-1), 0, navy),
        ("LEFTPADDING", (0,0), (-1,-1), 12), ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 11), ("BOTTOMPADDING", (0,0), (-1,-1), 11),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [brand, Spacer(1, 11)]
    story.append(Paragraph("Relatório técnico inteligente", styles["title"]))
    story.append(Paragraph("Análise de registros de detecção por visão computacional e IA generativa", styles["subtitle"]))
    story.append(Spacer(1, 9))

    meta_rows = [
        [Paragraph("<b>IDENTIFICADOR</b><br/>" + inline(value_for("Identificador do relatório")), styles["small"]),
         Paragraph("<b>PERÍODO OBSERVADO</b><br/>" + inline(value_for("Período observado")), styles["small"])],
        [Paragraph("<b>ORIGEM</b><br/>" + inline(value_for("Origem(ns)")), styles["small"]),
         Paragraph("<b>MODELO DE DETECÇÃO</b><br/>YOLO", styles["small"])],
    ]
    meta_table = Table(meta_rows, colWidths=[width * .37, width * .63])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), pale), ("BOX", (0,0), (-1,-1), .6, line),
        ("INNERGRID", (0,0), (-1,-1), .4, line), ("LEFTPADDING", (0,0), (-1,-1), 9),
        ("RIGHTPADDING", (0,0), (-1,-1), 9), ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story += [meta_table, Spacer(1, 10)]

    kpi_values = [
        ("DETECÇÕES", value_for("Total de registros de detecção")),
        ("GRUPOS DE EVIDÊNCIA", value_for("Grupos de evidências selecionados")),
        ("CONFIANÇA MÉDIA YOLO", confidence_text),
    ]
    kpi_cells = []
    for label, value in kpi_values:
        kpi_cells.append(Paragraph(
            f"<font color='#65758B' size='6.5'><b>{escape(label)}</b></font><br/><font color='#142033' size='16'><b>{escape(value)}</b></font>",
            ParagraphStyle("FWKPI", fontName=font_regular, fontSize=8, leading=19, alignment=TA_LEFT)))
    kpi = Table([kpi_cells], colWidths=[width / 3] * 3)
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), orange_pale), ("BOX", (0,0), (-1,-1), .6, colors.HexColor("#F2C5A8")),
        ("INNERGRID", (0,0), (-1,-1), .5, colors.HexColor("#F2C5A8")),
        ("LEFTPADDING", (0,0), (-1,-1), 9), ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [kpi, Spacer(1, 7)]

    if body_start >= len(lines):
        body_start = 0
    content_lines = lines[body_start:]
    i = 0
    current_heading = ""
    while i < len(content_lines):
        stripped = content_lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("# "):
            # Brand/title already rendered above.
            i += 1
            continue
        if stripped.startswith("## "):
            current_heading = stripped[3:].strip()
            story.append(Paragraph(inline(current_heading), styles["h2"]))
            story.append(HRFlowable(width="100%", thickness=.7, color=line, spaceAfter=5))
            i += 1
            continue
        if stripped.startswith("|" ):
            raw_table = []
            while i < len(content_lines) and content_lines[i].strip().startswith("|"):
                row_line = content_lines[i].strip()
                if not re.fullmatch(r"[|\s:\-]+", row_line):
                    raw_table.append([cell.strip() for cell in row_line.strip("|").split("|")])
                i += 1
            if raw_table:
                max_cols = max(len(row) for row in raw_table)
                table_data = []
                for row_index, row in enumerate(raw_table):
                    row = row + [""] * (max_cols - len(row))
                    style = styles["table_head"] if row_index == 0 else styles["table"]
                    table_data.append([Paragraph(inline(cell), style) for cell in row])
                # Group evidence table gets more width for its description column.
                if max_cols == 5:
                    col_widths = [18*mm, 31*mm, 20*mm, 17*mm, width - 86*mm]
                elif max_cols == 4:
                    col_widths = [width / 4] * 4
                else:
                    col_widths = [width / max_cols] * max_cols
                tbl = Table(table_data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), navy_soft), ("TEXTCOLOR", (0,0), (-1,0), white),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, pale]),
                    ("GRID", (0,0), (-1,-1), .35, line), ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
                    ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ]))
                story += [tbl, Spacer(1, 7)]
                if max_cols == 5 and "grupos de evidência" in current_heading.lower():
                    trend_points = []
                    for row in raw_table[1:]:
                        label = row[0] if row else ""
                        cell_text = " ".join(row[4:]) if len(row) > 4 else ""
                        match = re.search(r"média\s+([0-9]+(?:\.[0-9]+)?)%", cell_text, re.IGNORECASE)
                        if match:
                            trend_points.append((label, float(match.group(1)) / 100))
                    if trend_points:
                        story.append(KeepTogether([
                            Paragraph("Evolução das pontuações registradas", styles["small"]),
                            ConfidenceTrend(trend_points, width),
                            Spacer(1, 4),
                        ]))
            continue
        if stripped.startswith("> "):
            quote_text = inline(stripped[2:])
            callout = Table([[Paragraph(quote_text, styles["callout"])]], colWidths=[width])
            callout.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), orange_pale), ("LINEBEFORE", (0,0), (0,0), 3, orange),
                ("LEFTPADDING", (0,0), (-1,-1), 10), ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]))
            story += [callout, Spacer(1, 6)]
            i += 1
            continue
        if stripped.startswith("- "):
            story.append(Paragraph("•  " + inline(stripped[2:]), styles["bullet"]))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            story.append(Paragraph(inline(stripped), styles["bullet"]))
            i += 1
            continue
        if stripped == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=.8, color=line, spaceBefore=3, spaceAfter=7))
            i += 1
            continue
        # Merge wrapped lines in the source markdown into a readable paragraph.
        paragraph_lines = [stripped]
        i += 1
        while i < len(content_lines):
            nxt = content_lines[i].strip()
            if not nxt or nxt.startswith(("#", "|", "> ", "- ", "---")) or re.match(r"^\d+\.\s+", nxt):
                break
            paragraph_lines.append(nxt)
            i += 1
        paragraph_text = " ".join(paragraph_lines)
        story.append(Paragraph(inline(paragraph_text), styles["body"]))

    doc.build(story, onFirstPage=add_page, onLaterPages=add_page)
    return buffer.getvalue()

