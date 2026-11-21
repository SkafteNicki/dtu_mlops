#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf>=1.24"]
# ///
"""Scaffold a first-draft Slidev deck (.md) from a Gamma-exported slide PDF.

Gamma decks are born-digital (embedded font subsets), but styled glyphs are
baked into a broken Type 3 subset: em dashes extract as literal "4"
("orchestration4each"), ampersands as "&" ("is&hard"), bullets as "¦".
No text-layer-only tool can recover those characters, so this script extracts
everything that *is* reliable in the text layer and marks the rest as explicit
TODOs for a render + vision-model review pass.

Pipeline position:
    1. this script          -> per-slide text, layout hints, figure inventory
    2. renders (--render)   -> pages/pNN.png + figures/pNN_figK.png
    3. vision-model review  -> fill TODOs (figure descriptions, glyph fixes)

Usage:
    uv run slides/research/extract_gamma_pdf.py slides/day7_cloud.pdf --render
    uv run slides/research/extract_gamma_pdf.py slides/day7_cloud.pdf -o /tmp/deck

Outputs (in -o OUTDIR, default: next to the PDF):
    gammaExtract_<stem>/<stem>.slidev.md     one Slidev slide per PDF page
    gammaExtract_<stem>/<stem>.figures.json  figure inventory (page, bbox, px)
    gammaExtract_<stem>/figures/pNN_figK.png cropped figures (unique content)
    gammaExtract_<stem>/pages/pNN.png        full-page renders (with --render)

How it reads the page:
  * Text is extracted span-by-span and reassembled into visual lines, merging
    fragments on the same baseline whose x-gap is small (< 25pt). Gamma splits
    list markers ("1.", bullet glyphs) into separate spans from the item text;
    merging them restores list lines, while wide gaps (side-by-side cards) stay
    separate blocks, so card grids do not interleave like pdftotext output.
  * Lines keep PDF content order, which Gamma writes in visual order; each
    line is tagged with its bbox as an HTML comment for layout verification.
  * Slide title = line(s) with the largest font size (Gamma slide titles are
    always the biggest text on the page).
  * Deck decorations (Gamma background art, course logo) repeat on every page
    but with a NEW xref each time; images are deduplicated by content hash, and
    any hash appearing on >= half the pages is dropped from the figure
    inventory.
  * Bullets: the broken "¦" glyph is recognized and rendered as "- ".
  * Known glyph corruptions ("...use4HTTP", "is&hard") are left inline and
    flagged in the header for the vision pass - they cannot be fixed from the
    text layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf

DECORATIVE_MIN_SHARE = 0.5  # image content on >= half of the pages => decoration
LINE_Y_TOL = 4.0            # pt of vertical tolerance for same baseline
LINE_GAP_MAX = 25.0          # pt of horizontal gap still merged into one line
BULLET_CHARS = {"¦", "•", "\uf0b7"}


def bullet_markers(page: pymupdf.Page) -> list[float]:
    """Y-centres of Gamma's list markers.

    Gamma draws bullets as zero-width Type 3 glyphs that get_text() drops
    entirely (poppler emits them as "¦"). texttrace() still sees them; a text
    line whose vertical range contains such a marker is a bullet item.
    """
    ys = []
    for span in page.get_texttrace():
        text = "".join(chr(c[0]) for c in span["chars"])
        bbox = span["bbox"]
        if "Type3" in span["font"] or text.strip() in BULLET_CHARS:
            if bbox[2] - bbox[0] < 1.0:  # zero-advance marker glyph
                ys.append((bbox[1] + bbox[3]) / 2)
    return ys


def find_decorative_digests(doc: pymupdf.Document) -> set[str]:
    """Content hashes of images that repeat on many pages.

    Gamma exports re-embed the same logo with a NEW xref on every page, so
    xref counting is not enough; content hashes are the reliable key.
    """
    per_digest: Counter[str] = Counter()
    for page in doc:
        for info in page.get_image_info(xrefs=True, hashes=True):
            digest = info.get("digest")
            if digest:
                per_digest[digest] += 1
    threshold = max(3, int(DECORATIVE_MIN_SHARE * doc.page_count))
    return {d for d, n in per_digest.items() if n >= threshold}


def page_pieces(page: pymupdf.Page) -> list[dict]:
    """Span-level text pieces in PDF content order (Gamma writes visual order)."""
    pieces: list[dict] = []
    order = 0
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                pieces.append({
                    "text": span["text"], "size": span["size"], "idx": order,
                    "x0": span["bbox"][0], "x1": span["bbox"][2],
                    "y0": span["bbox"][1], "y1": span["bbox"][3],
                })
                order += 1
    return pieces


def _x_gap(p: dict, q: dict) -> float:
    return max(p["x0"] - q["x1"], q["x0"] - p["x1"])


def assemble_lines(pieces: list[dict], markers: list[float]) -> list[dict]:
    """Merge same-baseline fragments (list markers, styled spans) into lines."""
    lines: list[dict] = []
    for p in pieces:  # content order in
        host = None
        for L in lines:
            if abs((p["y0"] + p["y1"]) / 2 - L["cy"]) > LINE_Y_TOL:
                continue
            for q in L["pieces"]:
                if _x_gap(p, q) <= LINE_GAP_MAX:
                    host = L
                    break
            if host:
                break
        if host:
            host["pieces"].append(p)
        else:
            lines.append({"cy": (p["y0"] + p["y1"]) / 2, "pieces": [p]})

    ordered = sorted(lines, key=lambda L: min(p["idx"] for p in L["pieces"]))
    result = []
    for L in ordered:
        parts = sorted(L["pieces"], key=lambda p: (p["x0"], p["idx"]))
        text = parts[0]["text"]
        for a, b in zip(parts, parts[1:]):
            text += b["text"] if _x_gap(a, b) <= 1.0 else " " + b["text"]
        y0 = min(p["y0"] for p in parts)
        x0 = min(p["x0"] for p in parts)
        x1 = max(p["x1"] for p in parts)
        norm = " ".join(text.split())
        is_bullet = norm.startswith(tuple(BULLET_CHARS))
        if is_bullet:
            norm = "- " + norm.lstrip("".join(BULLET_CHARS)).strip()
        elif any(y0 - 2 <= m <= max(p["y1"] for p in parts) + 2 for m in markers):
            is_bullet = True
            norm = "- " + norm
        result.append({
            "text": norm,
            "size": max(p["size"] for p in parts),
            "idx": min(p["idx"] for p in parts),
            "bbox": (x0, y0, x1, max(p["y1"] for p in parts)),
            "bullet": is_bullet,
        })
    return result


def split_title(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """Title line(s) = largest font size; wrapped titles share the size."""
    if not lines:
        return [], []
    top = max(L["size"] for L in lines)
    return [L for L in lines if L["size"] == top], [L for L in lines if L["size"] < top]


def figure_notes(page: pymupdf.Page, page_no: int, fig_dir: Path,
                 decorative: set[str], dpi: int, figures: list[dict]) -> list[str]:
    notes: list[str] = []
    k = 0
    for info in page.get_image_info(xrefs=True, hashes=True):
        if info.get("digest") in decorative:
            continue
        k += 1
        bbox = pymupdf.Rect(info["bbox"])
        pix = page.get_pixmap(dpi=dpi, clip=bbox)
        name = fig_dir / f"p{page_no:02d}_fig{k}.png"
        pix.save(name)
        figures.append({
            "page": page_no, "file": str(name),
            "bbox": [round(v, 1) for v in bbox],
            "native_px": [info["width"], info["height"]],
            "crop_px": [pix.width, pix.height],
        })
        notes.append(
            f"<!-- TODO figure: figures/p{page_no:02d}_fig{k}.png "
            f"(bbox ({round(bbox.x0)},{round(bbox.y0)})-({round(bbox.x1)},{round(bbox.y1)}), "
            f"native {info['width']}x{info['height']}px, on-slide "
            f"{round(bbox.width)}x{round(bbox.height)}pt). Describe it with a "
            f"vision model and note what a redrawn Slidev version needs. -->")
    return notes


def convert(pdf_path: Path, outdir: Path, render: bool, dpi: int) -> None:
    doc = pymupdf.open(pdf_path)
    stem = pdf_path.stem
    work = outdir / f"gammaExtract_{stem}"
    fig_dir, page_dir = work / "figures", work / "pages"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if render:
        page_dir.mkdir(exist_ok=True)

    decorative = find_decorative_digests(doc)
    figures: list[dict] = []
    md = [
        "<!--",
        f"Slidev scaffold generated from {pdf_path.name} ({doc.page_count} pages).",
        "Before first use, review with a vision model against pages/pNN.png:",
        "  1. TODO figure comments below -> describe + recreate images",
        "  2. broken Type 3 glyphs: em dashes read as '4' (use4HTTP -> use — HTTP),",
        "     '&' -> 'is hard' emphasis mark, '¦' -> bullet (converted to '- ' here)",
        "  3. reading order: every line is tagged with its bbox; re-column if off",
        "-->",
    ]
    for i, page in enumerate(doc, start=1):
        lines = assemble_lines(page_pieces(page), bullet_markers(page))
        titles, body = split_title(lines)
        md.append(f"<!--  slide {i}/{doc.page_count}  -->")
        md.append("---")
        md.append("")
        if titles:
            md.append("# " + " ".join(t["text"] for t in titles))
            md.append("")
        for L in body:
            x0, y0, x1, y1 = (round(v) for v in L["bbox"])
            md.append(f"<!-- line ({x0},{y0})-({x1},{y1}) -->")
            md.append(L["text"])
        figs = figure_notes(page, i, fig_dir, decorative, dpi, figures)
        if figs:
            md.append("")
            md.extend(figs)
        md.append("")
        if render:
            page.get_pixmap(dpi=dpi).save(page_dir / f"p{i:02d}.png")

    (work / f"{stem}.slidev.md").write_text("\n".join(md), encoding="utf-8")
    (work / f"{stem}.figures.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    print(f"wrote {work / f'{stem}.slidev.md'}")
    print(f"wrote {work / f'{stem}.figures.json'}: {len(figures)} figures, "
          f"{len(decorative)} decorative image digests skipped")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold Slidev markdown from a Gamma slide PDF")
    ap.add_argument("pdf", type=Path, help="Gamma-exported slide PDF")
    ap.add_argument("-o", "--outdir", type=Path, default=None,
                    help="output dir (default: next to the PDF)")
    ap.add_argument("--render", action="store_true",
                    help="also render full-page PNGs for a vision review pass")
    ap.add_argument("--dpi", type=int, default=200, help="render dpi (default 200)")
    args = ap.parse_args()
    if not args.pdf.exists():
        sys.exit(f"no such file: {args.pdf}")
    convert(args.pdf, args.outdir if args.outdir else args.pdf.parent, args.render, args.dpi)


if __name__ == "__main__":
    main()
