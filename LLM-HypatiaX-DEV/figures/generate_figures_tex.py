#!/usr/bin/env python3
r"""
generate_figures_tex.py

Scans a figures/ directory and generates a standalone LaTeX document
(figures.tex) that \includegraphics's every image found, one per page,
with a caption and label — ready to compile on its own:

    python3 generate_figures_tex.py
    pdflatex figures.tex

Usage
-----
    python3 generate_figures_tex.py \
        [--figures-dir figures] \
        [--output figures.tex] \
        [--source-tex jmlr_paper_extended.tex] \
        [--recursive] \
        [--title "HypatiaX — Figure Index"]

--source-tex is optional. If given, the script tries to pull the *real*
\\caption{...} text out of that paper for any figure it can match by
filename (via \\includegraphics{...<name>...} inside the same
\\begin{figure}...\\end{figure} block), so the index shows the actual
paper captions instead of a filename-derived guess. Any figure it can't
match falls back to a humanized version of the filename.

Only stdlib is used — no dependencies to install.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMAGE_EXTS = (".pdf", ".png", ".jpg", ".jpeg", ".eps")

LATEX_SPECIAL = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text: str) -> str:
    return "".join(LATEX_SPECIAL.get(ch, ch) for ch in text)


def humanize_filename(name: str) -> str:
    """fig07_scatter_train_vs_extrap -> 'Fig07 Scatter Train Vs Extrap'"""
    stem = Path(name).stem
    words = re.split(r"[_\-]+", stem)
    return escape_latex(" ".join(w for w in words if w))


def natural_key(s: str):
    """Natural sort so fig2 comes before fig10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def find_figures(figures_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in figures_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(files, key=lambda p: natural_key(p.name))


def _flatten_caption(raw: str) -> str:
    """
    Join a multi-line \\caption{...} body into one line safely.

    A bare '%' anywhere in LaTeX starts a comment to end of line — commonly
    used at the end of a line (e.g. '\\caption{%') purely to suppress the
    newline's spurious space. That's harmless in the original multi-line
    source, but if we naively join lines with spaces, an unescaped '%'
    silently swallows everything after it on the same output line,
    including (if it was the last line) the caption's closing brace. So we
    strip each line down to its first *unescaped* '%' before joining.
    """
    cleaned_lines = []
    for line in raw.splitlines():
        out, i, n = [], 0, len(line)
        while i < n:
            if line[i] == "\\" and i + 1 < n:
                out.append(line[i:i + 2])
                i += 2
                continue
            if line[i] == "%":
                break
            out.append(line[i])
            i += 1
        cleaned_lines.append("".join(out))
    return " ".join(" ".join(cleaned_lines).split())


BASE_PACKAGES = [
    ("graphicx",  r"\usepackage{graphicx}"),
    ("amsmath",   r"\usepackage{amsmath}"),
    ("natbib",    r"\usepackage{natbib}"),   # needed for \citep/\citet used in many captions
    ("hyperref",  r"\usepackage{hyperref}"),
    ("caption",   r"\usepackage{caption}"),
    ("geometry",  r"\usepackage[margin=1in]{geometry}"),
    ("longtable", r"\usepackage{longtable}"),
]


DEFAULT_SKIP_PACKAGE_SUBSTRINGS = ("jmlr",)


def extract_preamble_extras(source_tex: Path, skip_substrings=DEFAULT_SKIP_PACKAGE_SUBSTRINGS) -> tuple[list[str], list[str]]:
    """
    Best-effort scrape of the source paper's preamble (everything before
    \\begin{document}) for:
      - \\usepackage{...} lines (verbatim, so e.g. \\usepackage{bm} or
        \\usepackage{xcolor} carry over — captions often use \\bm{},
        \\textcolor{}, custom math operators, etc.)
      - \\newcommand / \\renewcommand / \\providecommand /
        \\DeclareMathOperator definitions (e.g. \\newcommand{\\Rsq}{R^{2}}),
        since captions frequently reuse the paper's own notation macros.

    Single-line definitions only (the common case) — a definition that
    spans multiple lines won't be picked up.
    """
    text = source_tex.read_text(encoding="utf-8", errors="ignore")
    preamble = text.split(r"\begin{document}", 1)[0]

    pkg_lines = re.findall(r"^\s*\\usepackage(?:\[[^\]]*\])?\{[^}]*\}.*$", preamble, re.M)
    if skip_substrings:
        pkg_lines = [
            l for l in pkg_lines
            if not any(s.lower() in l.lower() for s in skip_substrings)
        ]
    macro_lines = re.findall(
        r"^\s*\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator)\*?"
        r"\{?\\[a-zA-Z]+\}?(?:\[[0-9]+\])?(?:\[[^\]]*\])?\{.*\}\s*$",
        preamble, re.M,
    )
    # Drop macros that redefine LaTeX/class internals we don't want to touch.
    macro_lines = [m for m in macro_lines if r"\maketitle" not in m and r"\title" not in m]

    return [l.strip() for l in pkg_lines], [l.strip() for l in macro_lines]


def extract_captions_from_source(source_tex: Path) -> dict[str, str]:
    """
    Best-effort scrape of \\begin{figure}...\\end{figure} blocks in an
    existing paper: maps the basename referenced by \\includegraphics to
    the \\caption{...} text found in the same block. Not a full LaTeX
    parser — handles the common single-line \\caption{...} case, which
    covers the vast majority of real papers.
    """
    text = source_tex.read_text(encoding="utf-8", errors="ignore")
    captions: dict[str, str] = {}

    for block in re.findall(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", text, re.S):
        img_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", block)
        cap_match = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", block, re.S)
        if not img_match or not cap_match:
            continue
        basename = Path(img_match.group(1)).name
        caption_text = _flatten_caption(cap_match.group(1))
        if caption_text:
            captions[basename] = caption_text

    return captions


def make_label(name: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_\-]", "", Path(name).stem)
    return f"fig:auto:{stem}"


def build_document(figures: list[Path], figures_dir: str, captions: dict[str, str], title: str,
                    extra_packages: list[str] | None = None, extra_macros: list[str] | None = None) -> str:
    extra_packages = extra_packages or []
    extra_macros = extra_macros or []
    extra_pkg_text = "\n".join(extra_packages)

    lines = []
    lines.append(r"% Auto-generated by generate_figures_tex.py — do not hand-edit.")
    lines.append(r"% Regenerate with: python3 generate_figures_tex.py")
    lines.append(r"\documentclass[11pt]{article}")
    for name, default_line in BASE_PACKAGES:
        # Skip our default if the source preamble already loads this package
        # (preserves its original options instead of clashing with ours).
        if re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*\b" + re.escape(name) + r"\b[^}]*\}", extra_pkg_text):
            continue
        lines.append(default_line)
    if extra_packages:
        lines.append(r"% --- packages scraped from source paper preamble ---")
        lines.extend(extra_packages)
    if extra_macros:
        lines.append(r"% --- notation macros scraped from source paper preamble ---")
        lines.extend(extra_macros)
    lines.append(f"\\graphicspath{{{{{figures_dir}/}}}}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append("")
    lines.append(f"\\title{{{escape_latex(title)}}}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append("")
    lines.append(r"\tableofcontents")
    lines.append(r"\clearpage")
    lines.append("")

    if not figures:
        lines.append(r"\textit{No figures found.}")
    else:
        for fig in figures:
            basename = fig.name
            caption = captions.get(basename)
            source_note = ""
            if caption is None:
                caption = humanize_filename(basename)
            else:
                # Already valid LaTeX (contains \textbf, $...$, \ref, etc.)
                # from the source paper — do NOT re-escape it.
                source_note = r"\footnotesize\textit{(caption pulled from source .tex)}\\"

            rel_path = fig.as_posix()
            label = make_label(basename)

            lines.append(r"\begin{figure}[p]")
            lines.append(r"  \centering")
            lines.append(f"  \\includegraphics[width=0.9\\textwidth,height=0.75\\textheight,keepaspectratio]{{{rel_path}}}")
            lines.append(f"  \\caption{{{caption}}}")
            if source_note:
                lines.append(f"  {source_note}")
            lines.append(f"  \\label{{{label}}}")
            lines.append(r"  \par\smallskip")
            lines.append(f"  \\footnotesize\\texttt{{{escape_latex(basename)}}}")
            lines.append(r"\end{figure}")
            lines.append(r"\clearpage")
            lines.append("")

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures-dir", default="figures", help="Directory to scan (default: figures)")
    ap.add_argument("--output", default="figures.tex", help="Output .tex path (default: figures.tex)")
    ap.add_argument("--source-tex", default=None, help="Optional paper .tex to scrape real captions from")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories of figures-dir")
    ap.add_argument("--title", default="Figure Index", help="Title of the generated document")
    ap.add_argument("--include-venue-class", action="store_true",
                     help="Also pull in venue/class-specific packages (e.g. jmlr2e) from the "
                          "source preamble. Off by default so figures.tex only needs a stock "
                          "TeX Live install, not the paper's custom class file.")
    args = ap.parse_args()

    figures_dir = Path(args.figures_dir)
    if not figures_dir.is_dir():
        print(f"ERROR: figures directory not found: {figures_dir}", file=sys.stderr)
        return 1

    figures = find_figures(figures_dir, args.recursive)
    print(f"Found {len(figures)} figure(s) in {figures_dir}/ "
          f"({'recursive' if args.recursive else 'top-level only'})")
    for f in figures:
        print(f"  - {f}")

    captions: dict[str, str] = {}
    extra_packages: list[str] = []
    extra_macros: list[str] = []
    if args.source_tex:
        source_path = Path(args.source_tex)
        if source_path.is_file():
            captions = extract_captions_from_source(source_path)
            print(f"\nScraped {len(captions)} caption(s) from {source_path}")
            skip = () if args.include_venue_class else DEFAULT_SKIP_PACKAGE_SUBSTRINGS
            extra_packages, extra_macros = extract_preamble_extras(source_path, skip)
            print(f"Scraped {len(extra_packages)} package(s) and {len(extra_macros)} "
                  f"notation macro(s) from its preamble (so captions using them still compile)")
        else:
            print(f"WARNING: --source-tex given but file not found: {source_path}", file=sys.stderr)

    matched = sum(1 for f in figures if f.name in captions)
    print(f"\n{matched}/{len(figures)} figure(s) matched to a real caption; "
          f"{len(figures) - matched} will use a filename-derived caption.")

    doc = build_document(figures, args.figures_dir, captions, args.title, extra_packages, extra_macros)
    out_path = Path(args.output)
    out_path.write_text(doc, encoding="utf-8")
    print(f"\n✔ Wrote {out_path} ({len(figures)} figure(s))")
    print(f"  Compile with:  pdflatex {out_path.name}   (run twice for the ToC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
