"""Evidence-led PDF report generator for MedIntel-AI.

Supported report fields:
  question, title, summary, answer, sections, key_points, tables,
  charts, visuals, flowcharts, sources, warnings, limitations.

Charts/flowcharts must carry source_ids (or citations) referencing the
report's source IDs, unless explicitly marked illustrative=True.
Visuals may use a project-local image `path`, or `source_pdf` + `page`.
"""

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    KeepTogether, HRFlowable
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "reports"
ASSET_DIR = OUTPUT_DIR / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASSET_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#153B55")
BLUE = colors.HexColor("#247BA0")
TEAL = colors.HexColor("#168C91")
PALE_BLUE = colors.HexColor("#EAF4F8")
PALE_ORANGE = colors.HexColor("#FFF2E5")
INK = colors.HexColor("#263746")
MUTED = colors.HexColor("#64788A")
BORDER = colors.HexColor("#D9E4EA")


def _safe(value) -> str:
    """Escape untrusted text for ReportLab Paragraph XML."""
    if value is None:
        return ""
    return escape(str(value)).replace("\n", "<br/>")


def _paragraph_text(value) -> str:
    """Small safe Markdown subset: **bold** and newlines."""
    text = _safe(value)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return text


def _style_set():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MI_Brand", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=23, leading=27, textColor=NAVY, alignment=TA_LEFT,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="MI_Kicker", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7.5, leading=10, textColor=TEAL, alignment=TA_LEFT,
        tracking=1.1, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="MI_Section", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=13, leading=16, textColor=NAVY, spaceBefore=13,
        spaceAfter=6, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="MI_Subsection", parent=styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=10, leading=13, textColor=BLUE, spaceBefore=8,
        spaceAfter=4, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="MI_Body", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.2, leading=14, textColor=INK, spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="MI_Small", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7.6, leading=10, textColor=MUTED, spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="MI_Table", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7.5, leading=10, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        name="MI_TableHeader", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=7.5, leading=10, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        name="MI_Callout", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8.6, leading=12.5, textColor=INK, spaceAfter=2,
    ))
    return styles


def _safe_project_path(raw_path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def extract_pdf_page_image(pdf_path, page_number, dpi=150) -> Path | None:
    """Render a 1-based PDF page to PNG using PyMuPDF (fitz), if installed."""
    source = _safe_project_path(pdf_path)
    if source is None:
        return None
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError:
        return None

    try:
        page_number = int(page_number)
        if page_number < 1:
            return None
        with fitz.open(source) as pdf:
            if page_number > pdf.page_count:
                return None
            page = pdf.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            target = ASSET_DIR / f"source_page_{source.stem}_{page_number}.png"
            pix.save(target)
            return target
    except Exception:
        return None


def _number_list(values):
    if not isinstance(values, list) or not values:
        return None
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def _chart_has_provenance(item, known_source_ids):
    if item.get("illustrative") is True:
        return True
    cited = item.get("source_ids") or item.get("citations") or []
    return bool(cited) and all(source_id in known_source_ids for source_id in cited)


def create_chart(chart_info):
    """Create a bar, line or pie chart from supplied numeric evidence."""
    if not isinstance(chart_info, dict):
        return None
    chart_type = str(chart_info.get("type", "bar")).lower()
    data = chart_info.get("data") or {}
    labels = data.get("labels") or []
    if not isinstance(labels, list) or not labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.1), dpi=150)
    try:
        if chart_type in {"bar", "bar_chart", "bargraph"}:
            values = _number_list(data.get("values"))
            if values is None or len(values) != len(labels):
                return None
            ax.bar(labels, values)
            ax.set_ylabel(str(data.get("unit", "Value")))
            ax.grid(axis="y", alpha=0.22)
            ax.tick_params(axis="x", rotation=20)

        elif chart_type in {"line", "line_chart"}:
            series = data.get("series") or []
            if not series:
                values = _number_list(data.get("values"))
                series = [{"name": data.get("series_name", "Value"), "values": values}]
            plotted = 0
            for item in series:
                values = _number_list(item.get("values"))
                if values is None or len(values) != len(labels):
                    continue
                ax.plot(labels, values, marker="o", linewidth=2, label=str(item.get("name", "Series")))
                plotted += 1
            if not plotted:
                return None
            if plotted > 1:
                ax.legend(frameon=False)
            ax.set_ylabel(str(data.get("unit", "Value")))
            ax.grid(alpha=0.22)
            ax.tick_params(axis="x", rotation=20)

        elif chart_type in {"pie", "pie_chart", "donut"}:
            values = _number_list(data.get("values"))
            if values is None or len(values) != len(labels) or any(v < 0 for v in values) or sum(values) <= 0:
                return None
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")

        else:
            return None

        ax.set_title(str(chart_info.get("title", "Evidence visualization")), fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        return buffer
    finally:
        plt.close(fig)


def create_flowchart(flowchart_info):
    """Create a simple source-grounded flowchart.

    nodes: [{id, label, x?, y?, kind?}], edges: [{from, to, label?}]
    x/y are normalized 0..1 coordinates; absent positions use a vertical layout.
    """
    if not isinstance(flowchart_info, dict):
        return None
    nodes = flowchart_info.get("nodes") or []
    edges = flowchart_info.get("edges") or []
    if not nodes:
        return None

    positions = {}
    count = len(nodes)
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or not node.get("id") or not node.get("label"):
            return None
        x = node.get("x", 0.5)
        y = node.get("y", 0.92 - index * (0.82 / max(count - 1, 1)))
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        positions[node["id"]] = (x, y)

    fig, ax = plt.subplots(figsize=(8, max(4.5, min(8, count * 0.85))), dpi=150)
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        node_map = {node["id"]: node for node in nodes}
        for edge in edges:
            start_id, end_id = edge.get("from"), edge.get("to")
            if start_id not in positions or end_id not in positions:
                continue
            x1, y1 = positions[start_id]
            x2, y2 = positions[end_id]
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops={"arrowstyle": "->", "color": "#54758A", "lw": 1.5,
                                    "shrinkA": 24, "shrinkB": 24})
            if edge.get("label"):
                ax.text((x1+x2)/2 + 0.015, (y1+y2)/2, str(edge["label"]), fontsize=7,
                        color="#526575", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        for node_id, node in node_map.items():
            x, y = positions[node_id]
            kind = str(node.get("kind", "step")).lower()
            face = "#E8F5F8" if kind != "decision" else "#FFF0DF"
            ax.text(x, y, str(node["label"]), ha="center", va="center", fontsize=8,
                    color="#173B55", wrap=True,
                    bbox={"boxstyle": "round,pad=0.65", "facecolor": face,
                          "edgecolor": "#7EA5B8", "linewidth": 1.2})
        ax.set_title(str(flowchart_info.get("title", "Clinical pathway")), fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        buffer.seek(0)
        return buffer
    finally:
        plt.close(fig)


def _fit_image(image_source, max_width=165 * mm, max_height=175 * mm):
    reader = ImageReader(image_source)
    width, height = reader.getSize()
    if not width or not height:
        return None
    scale = min(max_width / width, max_height / height, 1.0)
    return Image(image_source, width=width * scale, height=height * scale)


def _append_callout(story, text, styles, background=PALE_BLUE):
    cell = Paragraph(_paragraph_text(text), styles["MI_Callout"])
    box = Table([[cell]], colWidths=[170 * mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL if background == PALE_BLUE else colors.HexColor("#D68A39")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(box)
    story.append(Spacer(1, 7))


def generate_pdf(report, filename=None):
    """Generate a professional, evidence-led A4 PDF from structured report data."""
    if not isinstance(report, dict):
        raise ValueError("Report must be a dictionary.")

    question = str(report.get("question", "")).strip()
    answer = str(report.get("answer", "")).strip()
    title = str(report.get("title") or "Medical Research Report").strip()
    summary = str(report.get("summary", "")).strip()
    if not question or not (answer or summary or report.get("sections")):
        raise ValueError("Question and at least one answer/summary/section are required.")

    if not filename:
        filename = "MedIntel_Report_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".pdf"
    filename = Path(filename).name
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    output_path = OUTPUT_DIR / filename

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=19 * mm, bottomMargin=22 * mm,
        title=f"MedIntel-AI | {title}", author="MedIntel-AI",
        subject=question,
    )
    styles = _style_set()
    story = []

    # Report identity header
    header = Table([
        [Paragraph("MedIntel-AI", styles["MI_Brand"]),
         Paragraph("MEDICAL INTELLIGENCE<br/>SOURCE-BASED REPORT", styles["MI_Kicker"])],
        [Paragraph(_safe(title), styles["MI_Section"]),
         Paragraph(_safe(datetime.now().strftime("%d %B %Y")), styles["MI_Small"])],
    ], colWidths=[112 * mm, 58 * mm])
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, TEAL),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.extend([header, Spacer(1, 12)])

    story.append(Paragraph("Research question", styles["MI_Section"]))
    _append_callout(story, question, styles)

    if summary:
        story.append(Paragraph("Executive summary", styles["MI_Section"]))
        _append_callout(story, summary, styles)

    if answer:
        story.append(Paragraph("Answer", styles["MI_Section"]))
        for paragraph in re.split(r"\n\s*\n", answer):
            paragraph = paragraph.strip()
            if paragraph:
                story.append(Paragraph(_paragraph_text(paragraph), styles["MI_Body"]))

    sections = report.get("sections") or []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = section.get("heading") or section.get("title")
        content = section.get("content", "")
        if heading:
            story.append(Paragraph(_safe(heading), styles["MI_Section"]))
        if content:
            for paragraph in re.split(r"\n\s*\n", str(content)):
                if paragraph.strip():
                    story.append(Paragraph(_paragraph_text(paragraph.strip()), styles["MI_Body"]))
        for bullet in section.get("bullets", []) or []:
            story.append(Paragraph("• " + _paragraph_text(bullet), styles["MI_Body"]))
        citation_labels = section.get("citations") or []
        if citation_labels:
            story.append(Paragraph("Evidence: " + _safe(", ".join(map(str, citation_labels))), styles["MI_Small"]))

    key_points = report.get("key_points") or []
    if key_points:
        story.append(Paragraph("Key findings", styles["MI_Section"]))
        for point in key_points:
            story.append(Paragraph("• " + _paragraph_text(point), styles["MI_Body"]))

    known_source_ids = {
        source.get("id") for source in (report.get("sources") or [])
        if isinstance(source, dict) and source.get("id")
    }

    # Tables
    for table_info in report.get("tables") or []:
        if not isinstance(table_info, dict):
            continue
        headers = table_info.get("headers") or []
        rows = table_info.get("rows") or []
        if not headers or not rows:
            continue
        story.append(Paragraph(_safe(table_info.get("title", "Evidence table")), styles["MI_Section"]))
        data = [[Paragraph(_safe(cell), styles["MI_TableHeader"]) for cell in headers]]
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            normalized = list(row[:len(headers)]) + [""] * max(0, len(headers) - len(row))
            data.append([Paragraph(_safe(cell), styles["MI_Table"]) for cell in normalized])
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([table, Spacer(1, 7)])
        cites = table_info.get("source_ids") or table_info.get("citations") or []
        if cites:
            story.append(Paragraph("Evidence: " + _safe(", ".join(map(str, cites))), styles["MI_Small"]))

    # Charts — require citations to known sources unless explicitly illustrative.
    for chart in report.get("charts") or []:
        if not _chart_has_provenance(chart, known_source_ids):
            continue
        image_buffer = create_chart(chart)
        if image_buffer is None:
            continue
        story.append(Paragraph(_safe(chart.get("title", "Data visualization")), styles["MI_Section"]))
        image = _fit_image(image_buffer, max_width=165 * mm, max_height=95 * mm)
        if image:
            story.append(image)
        note = "Illustrative data — not clinical evidence." if chart.get("illustrative") is True else "Chart generated from cited source data."
        story.append(Paragraph(_safe(note), styles["MI_Small"]))
        story.append(Spacer(1, 7))

    # Flowcharts — only show when provenance is provided or illustrative is explicit.
    for flowchart in report.get("flowcharts") or []:
        if not _chart_has_provenance(flowchart, known_source_ids):
            continue
        image_buffer = create_flowchart(flowchart)
        if image_buffer is None:
            continue
        story.append(Paragraph(_safe(flowchart.get("title", "Clinical pathway")), styles["MI_Section"]))
        image = _fit_image(image_buffer, max_width=165 * mm, max_height=150 * mm)
        if image:
            story.append(image)
        story.append(Paragraph("Illustrative pathway." if flowchart.get("illustrative") else "Pathway constructed from cited source material; verify the source before clinical use.", styles["MI_Small"]))
        story.append(Spacer(1, 7))

    # Visuals: existing local images or selected pages from source PDFs.
    for visual in report.get("visuals") or []:
        if not isinstance(visual, dict):
            continue
        image_path = _safe_project_path(visual.get("path"))
        if image_path is None and visual.get("source_pdf") and visual.get("page"):
            image_path = extract_pdf_page_image(visual["source_pdf"], visual["page"])
        if image_path is None:
            continue
        story.append(Paragraph(_safe(visual.get("title", "Source visual")), styles["MI_Section"]))
        image = _fit_image(str(image_path), max_width=165 * mm, max_height=175 * mm)
        if image:
            story.append(image)
        caption = visual.get("caption")
        if caption:
            story.append(Paragraph(_safe(caption), styles["MI_Small"]))
        if visual.get("source_document") or visual.get("page"):
            label = f"Source: {visual.get('source_document', visual.get('source_pdf', ''))} | Page {visual.get('page', 'N/A')}"
            story.append(Paragraph(_safe(label), styles["MI_Small"]))
        story.append(Spacer(1, 8))

    # References
    sources = report.get("sources") or []
    if sources:
        story.append(Paragraph("Evidence and references", styles["MI_Section"]))
        for index, source in enumerate(sources, 1):
            if not isinstance(source, dict):
                continue
            source_id = source.get("id") or f"S{index}"
            document = source.get("document") or source.get("document_name") or "Unknown document"
            page = source.get("page", source.get("page_number", "N/A"))
            extraction = source.get("extraction_method")
            line = f"[{source_id}] {_safe(document)} | Page {_safe(page)}"
            if extraction:
                line += f" | Extraction: {_safe(extraction)}"
            story.append(Paragraph(line, styles["MI_Body"]))

    # Warnings and limitations
    warnings = list(report.get("warnings") or [])
    limitations = list(report.get("limitations") or [])
    if warnings or limitations:
        story.append(Paragraph("Important limitations and cautions", styles["MI_Section"]))
        for item in warnings + limitations:
            _append_callout(story, item, styles, background=PALE_ORANGE)
    else:
        story.append(Paragraph("Educational information only. This report is not a diagnosis or medical advice.", styles["MI_Small"]))

    def add_footer(canvas, document_obj):
        canvas.saveState()
        page_width, _ = A4
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, page_width, 7 * mm, fill=1, stroke=0)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.white)
        canvas.drawString(20 * mm, 3.3 * mm, "MedIntel-AI | Educational and research purposes only")
        canvas.drawRightString(page_width - 20 * mm, 3.3 * mm, f"Page {document_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return output_path
