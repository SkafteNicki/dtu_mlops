# Research: extracting Gamma slide PDFs into first-draft Slidev markdown

Ticket: issue #703. Scope: the `slides/*.pdf` decks (Gamma exports, producer
`pdf-lib` + `GPL Ghostscript 9.56.1`, no tagged structure) are the only
source; the goal is one Slidev `.md` per deck, with images recreated rather
than reused. Question: what is the most reliable path from PDF to per-slide
text + layout hints + figure descriptions?

Everything below was measured on `slides/day7_cloud.pdf` (18 pages, 3.7 MB,
900x507 pt pages) on this machine (poppler 26.07.0, PyMuPDF 2.1.0,
pymupdf4llm 2.1.0, docling 2.151.1, CPython 3.13, CPU; a CUDA driver is
present but docling's model stack cannot execute on it - see below).

## Recommendation

**A three-stage pipeline: (1) a deterministic PyMuPDF scaffolder for text,
reading-order hints, and a deduplicated figure inventory; (2) a render+vision
pass for figure descriptions and glyph forensics; (3) docling standard as an
optional cross-check for missed text - never as the primary extractor.**

Stage 1 is committed as [extract_gamma_pdf.py](extract_gamma_pdf.py) (inline
script metadata, runs with plain `uv run`, dependency is `pymupdf` only):

```bash
# scaffold + figure crops + per-page renders (vision-pass inputs)
uv run slides/research/extract_gamma_pdf.py slides/day7_cloud.pdf -o /tmp/deck --render
```

This writes `gammaExtract_day7_cloud/` containing:

| File | Content |
|------|---------|
| `day7_cloud.slidev.md` | one `---`-separated Slidev slide per PDF page; title chosen by largest font size; every line tagged with its bbox in an HTML comment; explicit `TODO figure` comments where images sit |
| `day7_cloud.figures.json` | figure inventory: page, on-page bbox, native pixel size |
| `figures/pNN_figK.png` | cropped figure images (decorations removed) |
| `pages/pNN.png` | full-page renders (only with `--render`) |

Stage 2 (the part no text-layer tool can do - see "The glyph problem" below):
feed each `pages/pNN.png` to a vision-capable model/agent and ask it to (a)
write the figure descriptions for the `TODO figure` comments, (b) confirm or
repair the known glyph corruptions, (c) confirm the column order against the
bbox comments. Rendering 18 pages at 150 dpi takes ~2.7 s; at the script's
default 200 dpi the whole run (scaffold + 19 crops + 18 renders) is ~20 s.

Stage 3, optional: `uv run --with docling docling convert slides/day7_cloud.pdf
--device cpu --no-ocr --to md --image-export-mode placeholder` (see the
docling section for flags; ~60 s) and diff its markdown against the scaffold
to catch any text PyMuPDF missed. Do not feed docling's `![Image](...)` lines
into the Slidev file - half of them are the repeated Gamma background and
course logo.

## Sample output (stage 1, day7_cloud pages 4, 5, 16, 12)

The three layout corner cases - a 2x2 card grid, a numbered list whose markers
Gamma splits into separate spans, and bulleted list where bullets are
zero-width Type 3 glyphs - all come out reading correctly:

```markdown
<!--  slide 4/18  -->
---
# Why use cloud?
<!-- line (72,154)-(161,181) -->
Reliability
<!-- line (72,192)-(394,212) -->
Hardware failures, network outages, and system
<!-- line (475,154)-(569,181) -->
Scalability
<!-- ...cards stay grouped; left column then right, no interleaving ... -->

<!--  slide 5/18  -->
---
# Why cloud exists?
<!-- line (452,129)-(837,145) -->
1. All teams will henceforth expose their data and functionality through
<!-- line (452,173)-(833,189) -->
2. Teams must communicate with each other through these interfaces
<!-- line (452,261)-(824,277) -->
4. It doesn't matter what technology they use4HTTP, Corba, Pubsub,

<!--  slide 16/18  -->
---
# A word of warning
<!-- line (54,171)-(251,191) -->
Working in the cloud is&hard.
<!-- line (73,213)-(497,233) -->
- Everything taking longer because extra layer of communication
<!-- line (73,258)-(292,278) -->
- Syntax can be hard to remember

<!--  slide 12/18  -->
---
# Important services for ML
<!-- line (70,339)-(117,359) -->
Engine
<!-- line (197,339)-(244,359) -->
Bucket
<!-- TODO figure: figures/p12_fig1.png (bbox (70,219)-(143,292), native 225x225px, ... -->
<!-- TODO figure: figures/p12_fig2.png (bbox (197,219)-(271,292), native 225x225px, ... -->
```

## What is inside these PDFs (drives every result below)

Measured with `pdffonts` and `pdfimages -list` on `day7_cloud.pdf`:

- Body text uses embedded `Barlow-*` CID TrueType subsets (Identity-H); there is
  also a **Type 3 font** (object 323) that carries Gamma's styled glyphs.
- **No real em dash (U+2014) exists anywhere in the text layer.** Where the
  slide shows an em dash, both poppler and PyMuPDF extract a literal `4`
  (`use4HTTP`, `orchestration4each`: char `U+0034`, verified at span level in
  `rawdict`). An ampersand-like styled glyph extracts as `&` (`is&hard`).
- Bullets are **zero-width Type 3 glyphs** at the left margin: poppler emits
  them as `¦` words with `xMin == xMax`; PyMuPDF's `get_text()` drops them
  completely (they are only visible via the low-level `get_texttrace()`).
- Every page embeds the same 2 Gamma background images plus a 575x137 course
  logo, but **each page gets new xrefs** for the logo (pdf-lib re-embeds by
  page), so decoration dedup must use content hashes, not object numbers.
- The PDF is untagged (no structure tree), so "reading order" is only
  recoverable from geometry or content-stream order.

Consequences: any tool that only reads the text layer inherits the `4`/`&`
corruption; any tool that trusts object refs for image dedup drowns in
decorations; the glyph damage can only be confirmed/repaired by looking at
rendered pixels.

## Tool-by-tool results (all on day7_cloud.pdf, 18 pages)

### poppler - `pdftotext` / `pdfimages`

- **Speed**: `pdftotext` default and `-layout` each < 0.05 s. `pdfimages -png
  -p` took 22 s.
- **Reading order**: default mode interleaves the columns of the page-4 card
  grid line by line (`Reliability / Scalability / Hardware failures... /
  Predicting capacity needs...`) and drops card structure. `-layout` keeps
  visual columns by whitespace padding, which preserves nothing semantic and
  still splits wrapped lines (`capabilities` and `wasted resources` end up in
  another visual row). `-bbox-layout` fixes this partially: it emits
  word-level bounding boxes grouped into `<flow>/<block>` XML - a genuinely
  useful machine-readable layout source (poppler's pdftotext(1) documents
  this mode), but you would have to write the assembler script yourself.
- **Figures**: `pdfimages -list` lists per-page images with object ids, but
  `pdfimages -png` dumps **93 files**, of which 72 are the 4 decoration
  images repeated on all 18 pages, with **no placement (bbox) information**
  in the filenames. Unusable as a figure inventory without post-processing.
- **Glyphs**: `use4HTTP`, `is&hard`, and bullets as stray `¦` lines - all the
  Type 3 corruption described above. poppler's own man page admits the limit:
  "Some PDF files contain fonts whose encodings have been mangled beyond
  recognition. There is no way (short of OCR) to extract text from these files."
  (pdftotext(1), BUGS) - which is exactly the situation the Gamma Type 3
  subset creates, and why stage 2 renders pages.

Verdict: good for a quick eyeball (default), good raw coordinates
(`-bbox-layout`); bad as a primary pipeline.

### PyMuPDF 2.1.0 (the stage-1 engine)

- **Speed**: `get_text("dict")` over all 18 pages in 0.16 s; rendering all
  pages at 150 dpi 2.7 s; full stage-1 run (text + 19 figure crops + 18 page
  renders at 200 dpi) ~20 s, or ~8 s without page renders.
- **Reading order**: PyMuPDF returns blocks in content order, and Gamma writes
  text in visual order - the page-4 card grid comes out as four intact cards
  in row-major order (verified block-by-block). No layout model, but for
  these decks geometry heuristics suffice: the script re-assembles span
  fragments into lines by baseline ( Gamma splits list markers like `1.` into
  separate spans from the item text ) while refusing to merge across wide
  x-gaps, so side-by-side cards never interleave.
- **Titles**: slide titles are the largest font on the page - the scaffold
  picks them by `span size` (e.g. 34.4 pt vs 15.5 pt body on page 16).
- **Figures**: `Page.get_image_info(xrefs=True, hashes=True)` returns each
  placed image with bbox, xref, and a **content digest**; counting digests
  across pages flags the 2 backgrounds + logo as decorations (they appear on
  all 18 pages), leaving 19 true figures with bboxes to crop. This is the
  cleanest figure inventory of any candidate.
- **Failure modes**: same `4`/`&` corruption (unfixable in the text layer);
  zero-width bullet glyphs silently dropped by `get_text()` and only
  recoverable via `get_texttrace()` (the script does this); wrapped
  continuation lines are left as separate lines on purpose (the bbox comments
  make joining them trivial for the review pass).

Verdict: the best fit for a deterministic scaffold. Dependency-light
(`pymupdf` only), fast, and every heuristic is scoped to what Gamma decks
actually contain.

### pymupdf4llm 2.1.0

- **Speed**: 24 s (it OCRs every embedded image even on born-digital decks).
- **Output**: slide titles become arbitrary heading levels (`####`), the page-4
  card grid collapses into one interleaved paragraph (`Why use cloud?
  Reliability Scalability Hardware failures...`), and its "picture text"
  feature hallucinates on decorative art: `Made with GRMMA`, plus garbled
  diagram labels. It is designed for article-shaped text (README of the
  pymupdf4llm repo documents the header/picture-text behavior), not slide
  decks.

Verdict: not suitable.

### docling 2.151.1 (standard pipeline)

Run via `uv run --with docling docling convert ...` (no pyproject change).

- **Speed / stability**: `--device auto` picked the CUDA path and crashed
  after 93 s with `torch.AcceleratorError: CUDA error: no kernel image is
  available for execution on the device` (driver/kernel mismatch).
  Pinning `--device cpu` works: ~60 s for 18 pages with
  `--no-ocr --pdf-backend pypdfium2 --table-mode fast`.
  (`DOCLING_INFERENCE_COMPILE_TORCH_MODELS=false` avoids a pathological
  torch.compile warmup documented on slower machines; see docling CLI docs.)
- **Text quality**: best structural output of the automated tools - real
  bullet lists (page 6 "Pros and Cons" is perfectly reconstructed), card
  groupings, and headlined sections. But: page-level structure is *missing* -
  the `.md` has no page separators and slide titles and card headings both
  export as `##` (page numbers exist only in the JSON `prov` metadata);
  numbered lists get mangled differently (`- All teams ... service
  interfaces 1.`, `Anyone who doesn't do this will be fired6.`); bullet
  glyphs leak in as literal `¦` lines; and the same `use4HTTP` corruption.
- **Figures**: `--image-export-mode referenced` extracted **52 artifact
  files**, roughly half of them the repeated backgrounds/logo (docling names
  files by content hash, so duplicates are at least visible), and the `.md`
  inlines an `![Image](...)` line for every occurrence of every decoration.

Verdict: excellent as a **cross-check** (its layout model catches anything a
geometry heuristic misses), poor as the primary pipeline for this conversion
(no slide/page semantics, decoration spam, device fragility).

### docling VLM pipeline (SmolDocling) - automated "render + vision model"

`--pipeline vlm --vlm-model smoldocling` renders each page and runs a local
document-vision model. **Not measured.** On this machine (CPU only, the
GPU is unusable for docling's torch build, see above) the run on 18 pages had
not finished after ~4 minutes and was stopped. Given stage 1 already
recovers text and figure positions, a small local document VLM adds little
over stage 2's general vision model; not pursued further.

### Per-page render + general vision model (stage 2)

Rendering is cheap (2.7 s for 18 pages at 150 dpi with PyMuPDF), and it is
the **only** representation that contains the missing information: what the
`4`/`&`/bullet glyphs actually look like, and what each figure depicts.
The vision step itself is one model call per page with a fixed prompt
(text already extracted, figure crops attached). The original researcher's model
could not ingest images, so this step was spot-checked afterwards by the
orchestrating agent (Claude Opus, multimodal) on `pages/p05.png` from the
stage-1 `--render` output: it read the true em dash (`use—HTTP`), saw that
item 5 of the Bezos list is **red-highlighted** (emphasis that the text layer
does not carry, so the scaffold loses it), and described the left figure
(blue "spikes in demand vs. max compute over time" area chart, a
recreate-as-chart candidate). It also shows the deck's visual identity (dark
background, Barlow, red accent) useful for the theme ticket. One page is a
spot check, not an accuracy measurement. Note the model does not
need to *read* much (the scaffold already carries the text); it needs to
describe figures and adjudicate the flagged glyphs only.

## Summary table

| Tool | Reading order (card grid, list markers) | Figure inventory | Glyph corruption (`4`, `&`, bullets) | Time (18 pg) | Role |
|------|----------------------------------------|------------------|--------------------------------------|--------------|------|
| `pdftotext` default | interleaves cards, detaches markers | none | inherited | < 0.05 s | quick eyeball only |
| `pdftotext -layout` | visual, whitespace-only | none | inherited | < 0.05 s | none |
| `pdftotext -bbox-layout` | word coords + flows (raw but machine-readable) | none | inherited | ~ 0.1 s | DIY alternative to stage 1 |
| `pdfimages` | n/a | 93 files, 72 decorative, **no placement** | n/a | 22 s | none |
| PyMuPDF `get_text` | content order = visual order; needs span-level line assembly | none | inherited (+ silently drops zero-width bullets) | 0.16 s | engine inside stage 1 |
| PyMuPDF + `get_texttrace`/`get_image_info(hashes)` | as above, bullets recoverable | **bbox + content-hash dedup** | bullets recoverable; `4`/`&` not | ~ 8-20 s incl. crops/renders | **stage 1 (recommended)** |
| pymupdf4llm | collapses cards, arbitrary heading levels | OCR'd "picture text" (hallucinating) | inherited + OCR noise | 24 s | not suitable |
| docling standard (CPU) | best structure; slide titles and card headings flattened to the same `##` level; `- All ... 1.` list bugs | 52 files, ~half decorations, hash-named | inherited + literal `¦` | 60 s (+ 93 s for a CUDA auto-detect crash) | cross-check |
| render + vision model | n/a (n/a) | describes figures | **reads true glyphs** | 2.7 s render + 1 call/page | **stage 2 (required)** |

## Sources

Tool documentation and claims about intended behavior:

- poppler utils man pages (`pdftotext`, incl. `-bbox-layout`; `pdfimages`):
  <https://manpages.debian.org/testing/poppler-utils/pdftotext.1.en.html>,
  <https://manpages.debian.org/testing/poppler-utils/pdfimages.1.en.html>,
  project page <https://poppler.freedesktop.org/>
- PyMuPDF API: `Page.get_text`, `Page.get_texttrace`, `Page.get_image_info`:
  <https://pymupdf.readthedocs.io/en/latest/page.html>
- pymupdf4llm (README documents heading/picture-text behavior):
  <https://github.com/pymupdf/RAG>
- docling: CLI reference (device/pipeline flags):
  <https://docling-project.github.io/docling/reference/cli/> and repo:
  <https://github.com/docling-project/docling>
  SmolDocling vision model: <https://huggingface.co/ds4sd/smoldocling-256-preview>
- pdf-lib (the Gamma export producer): <https://github.com/Hopding/pdf-lib>
- Inline script metadata (uv): <https://docs.astral.sh/uv/guides/scripts/>
- Slidev syntax guide (slide separators, frontmatter):
  <https://sli.dev/guide/syntax>

All empirical claims above were produced by running the named tools on
`slides/day7_cloud.pdf` in this repo worktree; intermediate outputs live in
`/tmp/opencode/gamma/` and are not committed.
