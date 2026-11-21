# Day 7 Cloud: pilot Slidev draft (provisional)

Rough Slidev draft of `slides/day7_cloud.pdf`, made for the wayfinder ticket
[Pilot: extract Day 7 Cloud into a rough Slidev draft](https://github.com/SkafteNicki/dtu_mlops/issues/707).
The location (`slides/pilot/`) is temporary until the slides project layout is decided. The deck uses the default
Slidev theme and has no review changes: the content is as in the PDF.

Preview it without adding anything to the repo:

```bash
cd slides/pilot/day7_cloud
npx @slidev/cli slides.md
```

## How it was made

1. Scaffold: `uv run extract_gamma_pdf.py slides/day7_cloud.pdf --render`, using the script from the
   `research/gamma-pdf-extraction` branch. The output and page renders are not committed (they contain the Gamma images).
2. Vision pass over the page renders: em dashes repaired (`orchestration4each` → `orchestration—each`), the `&` glyph
   on page 16 is an ellipsis (`is…hard`), the "bullets" on page 16 are ⚠️ emojis, colour emphasis restored (red Bezos
   rule 5, red quote bar, coloured IAM headings), bold lead-ins restored on the pros/cons slide.
3. Images, following the image-recreation research: cheap ones redrawn inline (Sculley et al. Fig. 1, the demand-spike
   chart, the "want to do vs have to do" pyramid, the IAM policy table). The rest are `<Placeholder>` boxes
   (`components/Placeholder.vue`) that describe the source image and name its route.

No Gamma click builds (duplicate pages) and no "Error uploading image" placeholders were found in this deck.
