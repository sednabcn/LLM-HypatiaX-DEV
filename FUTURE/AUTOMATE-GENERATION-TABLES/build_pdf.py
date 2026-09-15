# -*- coding: utf-8 -*-
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

from table_data import ROWS

PAGE_SIZE = landscape(A3)
OUT = "/mnt/user-data/outputs/table_of_tables.pdf"

styles = getSampleStyleSheet()

cell_style = ParagraphStyle(
    "cell", parent=styles["Normal"], fontSize=7.2, leading=9, alignment=TA_LEFT,
)
label_style = ParagraphStyle(
    "label", parent=cell_style, fontName="Courier", fontSize=7,
)
header_style = ParagraphStyle(
    "header", parent=styles["Normal"], fontSize=8.5, leading=10,
    textColor=colors.white, fontName="Helvetica-Bold",
)
title_style = ParagraphStyle(
    "title", parent=styles["Title"], fontSize=16, spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "subtitle", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#444444"),
    spaceAfter=10,
)

# Status -> background color for the status cell
STATUS_COLORS = {
    "Generated (JSON-backed)": colors.HexColor("#dff0d8"),
    "Generated": colors.HexColor("#dff0d8"),
    "Static / stable": colors.HexColor("#d9edf7"),
    "Mechanical": colors.HexColor("#d9edf7"),
    "Generated (caveated)": colors.HexColor("#fcf8e3"),
    "Generated (unconfirmed schema)": colors.HexColor("#fcf8e3"),
    "Generated (path guessed)": colors.HexColor("#fcf8e3"),
    "Needs human decision": colors.HexColor("#f2dede"),
    "Blocked": colors.HexColor("#f2dede"),
    "Intentionally not automated": colors.HexColor("#e6e6e6"),
}

def make_pdf():
    doc = SimpleDocTemplate(
        OUT, pagesize=PAGE_SIZE,
        leftMargin=14*mm, rightMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm,
        title="List of Tables — Generation Status",
    )

    story = []
    story.append(Paragraph("List of Tables — Paper / Generation / Experiment Source", title_style))
    story.append(Paragraph(
        "All 56 \\label{tab:...} entries across jmlr_paper_main, supp_benchmark_report, "
        "and supp_routing_improvements, cross-referenced against generate_tables.py. "
        "None are currently wired via \\input{} — the paper still hard-types every table.",
        subtitle_style,
    ))

    header = ["#", "Label (tab:)", "Caption", "Generated file", "Status", "Experiment / source"]
    col_widths = [8*mm, 34*mm, 62*mm, 40*mm, 32*mm, 90*mm]

    data = [[Paragraph(h, header_style) for h in header]]
    row_bg_spans = []  # (row_index, color) for status column background

    for i, (num, label, caption, gen_file, status, source) in enumerate(ROWS, start=1):
        data.append([
            Paragraph(str(num), cell_style),
            Paragraph(label, label_style),
            Paragraph(caption, cell_style),
            Paragraph(gen_file, label_style),
            Paragraph(status, cell_style),
            Paragraph(source, cell_style),
        ])
        row_bg_spans.append((i, STATUS_COLORS.get(status, colors.white)))

    table = Table(data, colWidths=col_widths, repeatRows=1)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    # Color the status column per-row based on status
    for row_idx, color in row_bg_spans:
        style_cmds.append(("BACKGROUND", (4, row_idx), (4, row_idx), color))

    table.setStyle(TableStyle(style_cmds))
    story.append(table)

    story.append(Spacer(1, 8))
    legend_style = ParagraphStyle("legend", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#444444"))
    story.append(Paragraph(
        "<b>Legend:</b> "
        "<font backColor='#dff0d8'>&nbsp;&nbsp;</font> generated from real data &nbsp;&nbsp;"
        "<font backColor='#d9edf7'>&nbsp;&nbsp;</font> static/stable or mechanical &nbsp;&nbsp;"
        "<font backColor='#fcf8e3'>&nbsp;&nbsp;</font> generated but flagged (guessed path / unconfirmed schema / caveated) &nbsp;&nbsp;"
        "<font backColor='#f2dede'>&nbsp;&nbsp;</font> blocked or needs a human decision &nbsp;&nbsp;"
        "<font backColor='#e6e6e6'>&nbsp;&nbsp;</font> intentionally never automated",
        legend_style,
    ))

    doc.build(story)
    print("wrote", OUT)

if __name__ == "__main__":
    make_pdf()
