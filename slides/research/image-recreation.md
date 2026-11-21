# Research: recreating the Gamma deck images for Slidev (#705)

> Research ticket for the "replace Gamma PDFs with Slidev" plan (sibling of
> [slidev-hosting.md](./slidev-hosting.md)). Web sources checked on 2026-11-21. Deck evidence was
> collected with `pdftoppm` (50 DPI page renders), `pdfimages -list` / `pdfimages -all` and OCR
> (`tesseract`) over every embedded image in all 11 exported decks. This agent cannot view images
> directly, so image classes were assigned from OCR text + page context + image metadata; counts
> below are therefore accurate to about ±10 per class, with every quoted example individually
> verified by its OCR/`pdftotext` output.
>
> Nothing here is legal advice; it is a reading of the providers' own published terms.

## Recommendation

**Recreate, never copy from the PDFs — every class of image has a better Slidev-native form, and
the licensing story supports it.** The exported PDFs carry no per-image provenance (Gamma does not
embed source metadata), so it is impossible to know image-by-image which of Gamma's sources it came
from — AI model, Unsplash/Pexels stock, Giphy, Pictographic, or a plain web grab that Gamma itself
flags as "licensing ... unknown". A blanket recreate-the-images rule dissolves that whole question,
and it upgrades the deck at the same time (code-as-screenshot becomes real highlighted code,
diagrams become editable text, logos become components).

Rule of thumb per image type:

| Image type (≈ count / 317 content visuals) | How to recreate in Slidev | Licensing notes |
| --- | --- | --- |
| **Concept diagrams, flowcharts, architecture** (~80; incl. the recurring serving/logging/metrics pipeline figures in day11, MLflow registry in day8, GCP stack + IAM in day7, MLOps lifecycle figures in day2/day5) | **Mermaid or PlantUML code block in the deck.** Slidev supports both natively ([Syntax guide → Diagrams](https://sli.dev/guide/syntax): "Slidev supports Mermaid and PlantUML for creating diagrams from text"; PlantUML renders through `plantUmlServer`, default `https://www.plantuml.com/plantuml`). Use Mermaid `flowchart`/`sequenceDiagram` for pipelines; PlantUML for UML-flavored ones. For deliberately hand-drawn looks, draw in Excalidraw → export SVG (the decks already use Excalidraw screenshots, e.g. day8 "Exercise" pages). | Text you write; no third-party rights. |
| **Code, config, terminals, stack traces pasted as images** (~35; pyproject/requirements/Dockerfile screenshots in day3, pytest/pre-commit/workflow YAML in day6, fastapi code in day12, tracebacks in day4) | **Never re-image: retype (or `--8<--`-import) as real code blocks.** Slidev/Shiki highlighting, line numbers and `twoslash` already cover this — strictly better than the originals. | Own content. |
| **Tool/UI screenshots** (~45; GitHub repo/PR/Actions views, VS Code debugger, LearnIt, GCP console, PyTorch/HF/Kaggle pages, wandb/profiler UIs in day12) | Re-take **fresh, minimal screenshots** of the same tool state and store as per-deck `public/` assets; crop to the teaching point. Alternatively describe + link, or use an embedded live example. | Screenshots of third-party UIs shown for identification/teaching (nominative use) are ordinary practice; don't imply endorsement, don't use them decoratively. |
| **Tool logos & product icons** (~35; GCP service icons on day7 "Important services for ML", ONNX/Glow/TensorRT on day8, fast.ai, wandb, Excalidraw) | **Iconify logos as components:** Slidev docs — after `npm i @iconify-json/logos` you can write `<logos-docker />`, `<logos-google-cloud />`, `<logos-onnx />` etc. directly in the markdown ([sli.dev/features/icons](https://sli.dev/features/icons)). The `logos` collection is [gilbarbara/logos](https://github.com/gilbarbara/logos), **CC0-1.0** for the SVG artwork, with "All logos appearing on the site are the property of their respective owners." | Artwork CC0, free to use; each mark is still its owner's trademark — fine on educational slides to identify the tool, unmodified, without implying endorsement. This is also the standard way the tools expect their marks to appear. |
| **Charts/plots/benchmarks** (~30; cloud market-share, Triton/latency benchmarks in day8, distributed-training plots in day12, student growth in day1) | **Re-plot where the data is ours** (student numbers, course survey); otherwise generate small data-file + chart component (e.g. ECharts via `vue-chart` add-on or simple SVG) or import a re-plotted figure. For external benchmark numbers (Artificial Analysis, papers), cite the source on the slide. | Charts of facts are low-risk, but the specific raster may be copyrighted; re-plotting from cited numbers is the clean equivalent. |
| **Figures from papers/reports** (~12; Sculley et al. "Hidden Technical Debt" Fig. 1 appears verbatim or redrawn on 4 decks in 6 instances, Nature reproducibility-crisis survey bar chart, "reproducibility spectrum" Fig. 1 in day3, ML Canvas in day2/day5, WEF jobs chart in day2) | **Redraw simplified** (Mermaid/Excalidraw) **with citation**, or re-plot cited numbers. The Sculley figure is a block diagram: a Mermaid redraw is faithful and legible at slide scale, unlike the tiny 970×320 px rasters currently used. Keep the existing `[1] Sculley et al. 2015` citations that day8 already prints. | Copyrighted figures: republication of the raster needs permission/fair-use reasoning; a redraw-plus-citation avoids it. |
| **Memes/comics** (~25; xkcd "Machine Learning" (day2 p20), "with great power" (day1), Owlturd technical-debt comic (day2 p44), assorted work memes) | **Drop most; replace with a dry one-liner or nothing.** For the xkcd that must stay: xkcd is CC BY-NC 2.5 — the [license page](https://xkcd.com/license.html) explicitly allows reuse & hotlinking "with some kind of link" in "books, blogs, newsletters, and presentations" for non-commercial use, i.e. embed/hotlink with a visible credit link. Memes of unknown provenance (reposted social-media screenshots) should not be re-published. | xkcd: CC BY-NC 2.5, attribution mandatory, non-commercial. Random web memes/social screenshots: unknown rights-holders — do not re-host. |
| **Decorative AI art / theme backgrounds** (~50; every page carries 2 theme-background rasters, plus hero/accent art) | **Replace with the Slidev theme.** Use a clean theme + CSS gradients/`layout: cover`; if a slide genuinely needs art, generate new images with a model whose terms grant the course output rights, or use course-owned graphics. Do **not** lift the backgrounds out of the PDFs (see ownership note below). | New art = no third-party claims. Gamma's own theme artwork stays Gamma property ("themes, ... artwork, animations" are Services, ToS §4.1). |
| **Photos of people/products** (~5; case-study vials in day2, real-world ad example day8) | Keep as course-owned/re-shot photos, or pick equivalents from **Unsplash/Pexels** (both allow free use incl. commercial, no required attribution; credit anyway). Case-study photos made for the course are already ours. | Unsplash/Pexels licenses quoted below. |
| **"Made with Gamma" badge** (on every page of every deck) | **Drop it.** It disappears automatically with the recreation. | Gamma trademark (ToS §4.2; badge behavior per [help center](https://help.gamma.app/en/articles/11047226-how-do-collaboration-and-sharing-settings-work-in-gamma)); removes only on paid plans — every PDF here was exported from a **free-plan workspace**, confirming we have no badge-free versions to reuse anyway. |

Net effort: ~80 of the 317 visuals become **diagram code**, ~35 become **real code blocks**, ~35
become **Iconify components**, ~45 are **re-takeable screenshots** — only the charts/paper figures
(~42) and surviving memes (~2-3) need per-item judgment, and the ~50 decorative pieces are simply
subsumed by the new theme.

## What Gamma's terms and help docs say

### Ownership of exports and AI images

From the [Gamma Terms of Use](https://gamma.app/terms) (effective 2025-09-09):

- **§4.3 "Your Content"**: "Gamma does not claim ownership of Your Content. However, when you as a
  Registered User post or publish Your Content on or in Services, you represent that you own and/or
  have sufficient rights to Your Content to grant the license set forth in Section 4.4."
- **§4.1 "Services"**: "Except with respect to Your Content and User Content, you agree that Gamma
  and its suppliers own all rights, title and interest in Services (including but not limited to,
  any computer code, themes, objects, concepts, artwork, animations, sounds, audiovisual effects,
  methods of operation, moral rights, documentation, and Gamma software)."
- **§1.4 "Use of AI"**: AI "Results" are provided "as is" and "with all faults", and "Gamma makes
  no representations or warranties of any kind or nature with respect to any Results" — the terms
  never affirmatively assign ownership of AI outputs to you, they only decline to claim them.
- **§1.7 "Certain Restrictions"**: you shall not "reproduce, distribute, host or otherwise
  commercially exploit the Services or any portion thereof" — this restricts the *platform*, not
  your deck, but it is a second reason not to copy Gamma-owned theme artwork out of the exports.
- **§4.5 "AI Model Training"**: on plans without privacy controls, content (incl. published decks)
  is licensed to Gamma to train its models.

Gamma's pricing FAQ answers "What happens to my content if I downgrade or cancel your
subscription?" with "Your content remains yours, regardless of your subscription status"
([gamma.app/pricing](https://gamma.app/pricing)). No help-center article assigns copyright in
AI-generated images; a [community question](https://community.gamma.app/x/questions/msg_leArUfBey8YZ/understanding-gammas-property-rights-on-created-im)
("Does Gamma exert property rights to images it creates?") still has no official answer.

The [Acceptable Use Policy](https://gamma.app/acceptable-use-policy) adds an **attribution
requirement** for AI-assisted content: "you may not represent such AI-generated content as being
wholly generated by you or wholly generated by an AI, and you must take ultimate responsibility for
the content you publish. The role of AI in formulating your content must be disclosed in a manner
no reader could miss or not understand." A recreated-by-humans Slidev deck has no such disclosure
burden beyond honesty about any new AI art we choose to generate.

**Free-plan export badge.** Every page of all 11 PDFs contains a shared 575×137 px raster that OCRs
as "Made with GAMMA". Help center: "Free plan: the badge shows on shared and published views and on
exports"; "Paid plans ... the badge is removed automatically" and "Files you already exported keep
the badge" ([How do collaboration and sharing settings work in Gamma?](https://help.gamma.app/en/articles/11047226-how-do-collaboration-and-sharing-settings-work-in-gamma)).
So the decks were exported from a free-plan workspace, and the badge is Gamma's mark (ToS §4.2) —
publishing or keeping it in the Slidev decks is neither required nor wanted.

### Where Gamma's images come from, and their licenses

Gamma's help center lists the image sources in the editor's visuals menu: "AI images (default).
Unsplash, Giphy, web images, pictographic, or your own uploads" ([How do I use the visuals menu in
Gamma?](https://help.gamma.app/en/articles/11856101-how-do-i-use-the-visuals-menu-in-gamma)); AI
models offered include "DALL·E 3, GPT Image, Flux Kontext Max". The insert menu adds "Web image
search or Stock photos", "GIFs from GIPHY", "Icons, Pictographic illustrations, and Accent images"
([How do I add and edit images in Gamma?](https://help.gamma.app/en/articles/11028379-how-do-i-add-and-edit-images-in-gamma)).
The Generate API exposes `pexels`, `pictographic`, `giphy`, `aiGenerated` and web options, and
describes `webAllImages` as: *"Pulls the most relevant images from the web, even if licensing is
unknown."* ([Generate API parameters](https://developers.gamma.app/guides/generate-api-parameters-explained)).

Provider licenses (their own pages):

- **Unsplash** ([license](https://unsplash.com/license)): "Unsplash grants you an irrevocable,
  nonexclusive, worldwide copyright license to download, copy, modify, distribute, perform, and
  use images from Unsplash for free, including for commercial purposes, without permission from or
  attributing the photographer or Unsplash." Excluded: selling unaltered copies, and compiling
  images "to replicate a similar or competing service". Trademarks/logos visible *in* a photo are
  not covered.
- **Pexels** ([license](https://www.pexels.com/license/)): "All photos and videos on Pexels are free
  to use. Attribution is not required." Not allowed: selling unaltered copies, implying endorsement
  by people/brands shown, redistributing on other stock platforms, or using photos as your
  trademark.
- **Giphy** ([User Terms of Service](https://support.giphy.com/hc/en-us/articles/360020027752-GIPHY-User-Terms-of-Service)
  §5): except for public-domain content or content you have permission to use, "you shall not copy,
  modify, publish, transmit, distribute, perform, or display any content, nor shall you sell,
  license, rent, or otherwise use or exploit any content for commercial use" — their plain-language
  summary: you're only allowed to use found content "in connection with your use of the Services
  and solely for personal and non-commercial purposes". **GIFs from Giphy should not be re-published
  in the decks.**
- **Pictographic** ([Figma plugin page](https://www.figma.com/community/plugin/1379710174937620830)): free use requires
  "a link back to pictographic.io, crediting us in the final work"; commercial/no-attribution use needs a
  paid license. (No GIFs/stills in our decks were identified as Pictographic, but Gamma inserts them.)
- **Web images**: licensing explicitly unknown per Gamma's own API docs — this is the risky bucket,
  and OCR shows the decks do contain likely web-grabbed panels (a hype-cycle-style ML-infrastructure
  graphic, GCP "sketchnote" infographic, role-bubble charts, "Machine Learning Canvas").

**Practical conclusion:** publishing the original images is defensible for some classes (AI art,
Unsplash/Pexels), ambiguous for AI art generated while on a plan whose model terms we never
controlled (presented "as is", no warranty of non-infringement given third-party model training
data), and clearly problematic for Giphy content and unknown-license web images. Because the PDF
exports carry no per-source metadata, we cannot sort one from the other after the fact — which is
the decisive argument for the author's "redraw or regenerate" decision.

## Inventory of the images in the decks

Method: `pdfimages -list -all` on all 11 decks, deduplicated by MD5, then OCR (tesseract) of every
unique image plus `pdftotext` of its host page; page renders at 50 DPI (`pdftoppm`) checked the
layout. All OCR/metadata artifacts stayed in `/tmp` — nothing was committed.

**Global structure.** ~700 embedded raster objects deduplicate to **4 theme assets + 317 content
visuals** (51 of the 368 unique files are `smask` transparency channels of an image counted with
it). The 4 theme assets repeat on *every* page of *every* deck: two 2400×1352 / 2400×1600 JPEG
backgrounds (Gamma theme art), and the 575×137 "Made with GAMMA" badge + its alpha mask (free-plan
export, see above). None of this should survive into Slidev.

**Rough counts per type over 317 content visuals** (borders are fuzzy, e.g. a social-media
screenshot used as a meme counts once, as meme):

| Type | ~Count | Typical examples (deck, page, OCR evidence) |
| --- | --- | --- |
| Concept diagrams / flowcharts / architecture | 80 | day11 p21–27: six progressively annotated "Data → DB → Gather input → ML Prediction → Application" pipelines (ideal Mermaid sequence); day8 p5 MLflow registry diagram; day7 p13 GCP technology stack; day2 p19 MLOps/DevOps/ML venn |
| Tool/UI screenshots (incl. code-as-image) | 80 | day1 p3/p6/p11 GitHub profile/repo/docs; day6 p12 pre-commit config + p25 pytest output; day5 p13 HuggingFace datasets; day7 p10/p17 GCP console; day12 p10 wandb/system monitors |
| — of which code/config/terminal images | ~35 | day3 p14 `hparams` class vs CLI vs YAML config trio; day3 p16 pyproject.toml; day4 p9 traceback; day12 p20/p21 fastapi snippets |
| Decorative AI art (backgrounds, heroes, accents) | 50 | day2 p24 slide literally titled "Prompt: `<Machine Learning operations engineer>`"; full-page 2400×1356 art on title/case-study pages |
| Tool logos / product icons | 35 | day7 p12 eight GCP service icons (Cloud Engine, Bucket, Build, Registry, Vertex, Functions, Run — 225–400 px squares, several with alpha masks); day8 p7 ONNX/Glow/TensorRT marks; day8 p20/p31 Excalidraw logo cards |
| Charts / plots / benchmark graphs | 30 | day7 p9 "Cloud Provider Market Share Trend"; day8 p8 NVIDIA Triton "Total Requests per Second" benchmark; day12 p9 dataloader `num_workers` scaling; day1 p20 "Year-over-Year Increase in Number of Students" |
| Memes / comics | 25 | day2 p20 xkcd 1838 ("THIS IS YOUR MACHINE LEARNING SYSTEM? ... JUST STIR THE PILE"); day2 p44 Owlturd ("Hey technical debt, why do you always wear that mask? ... OWLTURD.COM"); day4 p13 xkcd 221 RNG; day1 p16 "OF COURSE YOU CAN PASS YOUR EXAM WITHOUT STUDYING" |
| Paper figures / report graphics | 12 | Sculley et al. ML-system Fig. 1: day1 p2/p9 (with printed "Figure 1: Only a small fraction of real-world ML systems is composed of the ML code"), day2 p9/p32 (2006 "Java impl." variant), day7 p3, day8 p2 (cited as `[1] Sculley, D., et al. (2015)`); day3 p8 Nature "IS THERE A REPRODUCIBILITY CRISIS?" survey; day3 p12 "Fig. 1. The spectrum of reproducibility." |
| Photos (people/products/places) | 5 | day2 p39 case-study vials "(a) Good vial — No Defect (b) Particle Defect (c) Chips & Cracks"; day8 p18 Danish car-insurance telematics ad shown as real-world example |
| Misc / unclassified low-text fragments | 30 | Small collage fragments, cropped halves, residual accents |

Per-deck flavor: day1 (intro) is screenshot+meme heavy; day2 (MLOps intro) is diagram/chart/AI-art
heavy; day3 (reproducibility) mixes paper figures with code screenshots; day4 (debugging) has
wandb UI screenshots and loss plots; day5 (project) is nearly all tool screenshots (HF, Kaggle,
MONAI, kornia); day6 (CI) is GitHub-workflow screenshots and YAML-as-image; day7 (cloud) is GCP
screenshots + service icons + IAM/stack diagrams; day8 (deployment) is logos + benchmark charts +
MLflow/Triton diagrams; day11 (monitoring) is the drift/pipeline diagram sequence; day12
(scaling) is FSDP/distributed-training diagrams + profiler screenshots + PyTorch-Lightning code
snippets; DeepLearningSoftware (guest deck) is DL history/timeline charts + TF/PyTorch memes +
precision/TensorRT figures.

Two notes for converters: the day11 p21–27 sequence is *the* showcase for Mermaid (one
`sequenceDiagram` extended slide by slide with clicks or fragments), and the ~35 code-as-image
panels simply become the markdown code blocks the decks should have had all along — slide-by-slide
conversion will naturally absorb them into text.

## Sources

- Gamma Terms of Use: <https://gamma.app/terms>
- Gamma Acceptable Use Policy: <https://gamma.app/acceptable-use-policy>
- Gamma pricing FAQ ("Your content remains yours"): <https://gamma.app/pricing>
- Gamma help center:
  [How do I add and edit images in Gamma?](https://help.gamma.app/en/articles/11028379-how-do-i-add-and-edit-images-in-gamma),
  [How do I use the visuals menu in Gamma?](https://help.gamma.app/en/articles/11856101-how-do-i-use-the-visuals-menu-in-gamma),
  [Collaboration and sharing settings / "Made with Gamma" badge](https://help.gamma.app/en/articles/11047226-how-do-collaboration-and-sharing-settings-work-in-gamma),
  [How do I generate images with AI in Gamma?](https://help.gamma.app/en/articles/11047176-how-do-i-generate-images-with-ai-in-gamma)
- Gamma Generate API (image source options incl. `webAllImages` "even if licensing is unknown"): <https://developers.gamma.app/guides/generate-api-parameters-explained>
- Gamma community (unanswered) question on image rights: <https://community.gamma.app/x/questions/msg_leArUfBey8YZ/understanding-gammas-property-rights-on-created-im>
- Unsplash license: <https://unsplash.com/license>
- Pexels license: <https://www.pexels.com/license/>
- GIPHY User Terms of Service (§5): <https://support.giphy.com/hc/en-us/articles/360020027752-GIPHY-User-Terms-of-Service>
- Pictographic (Figma plugin listing, attribution requirement): <https://www.figma.com/community/plugin/1379710174937620830/pictographic-free-illustrations-vector-and-png-format>
- Slidev Syntax guide (Mermaid + PlantUML in-deck): <https://sli.dev/guide/syntax>
- Slidev PlantUML feature (server rendering, `plantUmlServer`): <https://sli.dev/features/plantuml>
- Slidev Icons (Iconify components, `<logos-vue />`): <https://sli.dev/features/icons>
- gilbarbara/logos (Iconify `logos` collection, CC0-1.0; trademark note): <https://github.com/gilbarbara/logos>
- xkcd license (CC BY-NC 2.5): <https://xkcd.com/license.html>
- Sculley et al. 2015, "Hidden Technical Debt in Machine Learning Systems" (figure referenced and
  cited in the decks themselves), NIPS 2015:
  <https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems>
