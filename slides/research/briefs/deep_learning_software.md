# Review brief: deep_learning_software (`slides/DeepLearningSoftware.pdf`)

- **Modules compared:** `s1_development_environment/` (all modules; mainly `deep_learning_software.md`) @ `main`
    d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b, plus `s4_debugging_and_logging/boilerplate.md` because it folds into this
    module. Overlaps checked by text search in `slides/day1_introduction_to_the_course.pdf`,
    `slides/day2_introduction_to_mlops.pdf`, `slides/day12_scaling_applications.pdf` and in `pages/timeplan.md`.
- **Other inputs:** `literature/` (`deep_learning.pdf`, Raff 2019, Sculley et al. 2015, Shankar et al. 2022); web sources
    (accessed 2026-11-24)
- **Pages:** 19 PDF pages → 18 slides after merging click builds (pages 6=7). Pages 10–13 zoom in on one circle at a time
    and stay separate. The file is a PowerPoint export dated 2 January 2024, not a Gamma export, so there are no Gamma
    placeholders, theme art or badge. Text was extracted and every page render was checked by eye.

## Deck outline (narrative order to keep)

1. Title: page 1
2. Deep learning then and now (old field, 2012 revolution, what followed, industry): pages 2–5
3. The drivers of the revolution, with software as the one not talked about: pages 6–7
4. Why frameworks exist (hand-written layer, barrier to enter): pages 8–9
5. How to make a modern framework (tensors, hardware acceleration, autodiff): pages 10–13
6. The current landscape and why PyTorch: pages 14–16
7. High-level frameworks: page 17
8. Trends: page 18
9. Meme of the day: page 19

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "Deep learning is old" (p. 2)
    - "In 2012 the Alexnet paper crushed all competition on in the imagenet benchmark" (p. 3).
        DEEP_LEARNING_SOFTWARE-23 proposes a language fix.
    - "Beating humans at their own games" (p. 4)
    - "Not talked about that much" (p. 7)
    - "Deep learning is just a log of simple math" (p. 8). DEEP_LEARNING_SOFTWARE-23 proposes a language fix.
    - "We do not want to deal with this ourself⚠️" (p. 8). DEEP_LEARNING_SOFTWARE-23 proposes a language fix.
    - "Low barrier to enter mean it accessible to more people = more people driving the technology forward" (p. 9).
        DEEP_LEARNING_SOFTWARE-23 proposes a language fix.
    - "If you have the time, learn the basics of them all 🙂" (p. 16)
    - "Recommend, to only use these if you understand the underlying framework." (p. 17). DEEP_LEARNING_SOFTWARE-23
        proposes a language fix.
- Voice markers:
    - The two "?" circles, revealed as Hype and Software (pp. 6–7)
    - The hand-written NumPy layer as the argument for frameworks (p. 8). DEEP_LEARNING_SOFTWARE-04 fixes its code.
    - "How to make a modern DL framework" as a three-circle recipe, one circle per slide (pp. 10–13)
    - "I highly recommend…" with the framework-affair comic (p. 16). DEEP_LEARNING_SOFTWARE-24 covers the image.
    - Emoji bullets: 🔥 for history, 💡 for reasons, ⚡ for features

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): the Lightning
    (boilerplate) module folds into Deep Learning Software, which becomes PyTorch + Lightning on Day 1. Lightning is
    optional but preferred, because Hugging Face projects may not need it. → see DEEP_LEARNING_SOFTWARE-12 and
    DEEP_LEARNING_SOFTWARE-13
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651):
    Rock-Paper-Scissors replaces corrupted MNIST, pending a real-photo check. The deck names no exercise dataset, but a
    slide on the exercise would. → see DEEP_LEARNING_SOFTWARE-16
- [Lock module numbering for the new layout](https://github.com/SkafteNicki/dtu_mlops/issues/656): numbers are not
    settled, so the deck should cite modules by name.
- [Agentic modules for the MLOps course](https://github.com/SkafteNicki/dtu_mlops/issues/618): this session gets
    modules on setting up a coding agent and on agent safety. The lecture may carry agentic content, but per
    [Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681) it does not
    otherwise reference the A-modules. → see DEEP_LEARNING_SOFTWARE-20
- [Which decks does the January 2027 course need?](https://github.com/SkafteNicki/dtu_mlops/issues/723) is open and may
    move content between decks. → affects DEEP_LEARNING_SOFTWARE-12 and DEEP_LEARNING_SOFTWARE-18

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased (e.g. CLOUD-01) -->

### DEEP_LEARNING_SOFTWARE-01 · The "ChatGPT" example is GPT-3 output

- **Category:** wrong
- **Priority:** must
- **Pages:** 4
- **Finding:** The question-and-answer text captioned "ChatGPT" (favourite animal, giraffe eyes, three-legged animals)
    matches a 2020 blog post that tests GPT-3 (see evidence).
- **Proposal:** Relabel it as GPT-3 with a citation, or replace it with a real current chat example.
- **Evidence:** "Giving GPT-3 a Turing Test", 2020-07-06 (author's name is in the URL; the typos hook rewrites it),
    <https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html> (accessed 2026-11-24). <!-- codespell:ignore -->

### DEEP_LEARNING_SOFTWARE-02 · U-Net was not about real-time segmentation

- **Category:** wrong
- **Priority:** must
- **Pages:** 4
- **Finding:** U-Net is captioned "Real-time semantic segmentation" over a street scene. The paper is on biomedical image
    segmentation and claims "less than a second" for a 512x512 image, which is not a real-time claim.
- **Proposal:** Keep U-Net with its biomedical claim and a matching image, or keep the street scene and name the model
    that produced it. The origin of the street image is `unverified`.
- **Evidence:** Ronneberger et al. 2015, abstract, <https://arxiv.org/abs/1505.04597> (accessed 2026-11-24).

### DEEP_LEARNING_SOFTWARE-03 · AlphaFold predicts structure

- **Category:** wrong
- **Priority:** must
- **Pages:** 4
- **Finding:** AlphaFold is captioned "Solving protein engineering". The Nobel committee separates the two: Hassabis and
    Jumper were honoured for "protein structure prediction", Baker for "computational protein design".
- **Proposal:** Change the caption to structure prediction, and optionally mention the 2024 Nobel Prize.
- **Evidence:** Nobel Prize in Chemistry 2024 press release,
    <https://www.nobelprize.org/prizes/chemistry/2024/press-release/>; Jumper et al. 2021,
    <https://www.nature.com/articles/s41586-021-03819-2> (accessed 2026-11-24).

### DEEP_LEARNING_SOFTWARE-04 · The NumPy layer has a wrong backward pass

- **Category:** wrong
- **Priority:** must
- **Pages:** 8
- **Finding:** `x_gradient` uses `gradient @ self.weight`; it needs the transpose. The example also passes a `(10, 5)`
    gradient for a `(10, 1)` output, which hides the error: it runs, but the weight silently becomes `(5, 5)`. With a
    correctly shaped gradient it raises a matmul shape error.
- **Proposal:** Fix both lines in a real code block. Optionally use the bug as the teaching point, since it is the kind
    of mistake autodiff removes (author's call). Keeps the voice marker.
- **Evidence:** The slide's code re-typed and run with NumPy on 2026-11-24: output shape `(10, 1)`, weight shape after
    `update` `(5, 5)`, `ValueError` on a `(10, 1)` gradient.

### DEEP_LEARNING_SOFTWARE-05 · Timeline has an error and stops in 2017

- **Category:** wrong
- **Priority:** must
- **Pages:** 2
- **Finding:** The timeline dates Deep Boltzmann Machines to 2006; the paper is from 2009. It also prints "McCulloch &
    Pitt" and leaves out the Transformer, although it runs to 2017 (capsule networks).
- **Proposal:** Redraw a shorter timeline from primary sources, adding the Transformer. Goodfellow et al.'s three waves
    of development could structure it.
- **Evidence:** Salakhutdinov & Hinton 2009, <https://proceedings.mlr.press/v5/salakhutdinov09a.html>; Vaswani et al.
    2017, <https://arxiv.org/abs/1706.03762>; Goodfellow et al., section 1.2.1,
    <https://www.deeplearningbook.org/contents/intro.html> (accessed 2026-11-24). Other dates on the image were not
    checked one by one (`unverified`).

### DEEP_LEARNING_SOFTWARE-06 · Market forecast doesn't show industry usage

- **Category:** outdated
- **Priority:** should
- **Pages:** 5
- **Finding:** The claim is about usage in industry, but the chart is a US revenue forecast with a 24.1% growth rate
    (2022–2030). The same publisher now gives 22.0% for 2024–2030 from USD 14.97 billion in 2023.
- **Proposal:** Replace it with adoption data: the 2026 AI Index reports AI adoption at 88% of surveyed organizations in
    2025 and generative AI at 70%. That measures AI, not deep learning alone, so word it that way.
- **Evidence:** Stanford HAI, AI Index 2026, "Economy", <https://hai.stanford.edu/ai-index/2026-ai-index-report/economy>;
    Grand View Research, <https://www.grandviewresearch.com/industry-analysis/us-deep-learning-market-report> (accessed
    2026-11-24).

### DEEP_LEARNING_SOFTWARE-07 · "Since then" stops around 2022

- **Category:** outdated
- **Priority:** should
- **Pages:** 4
- **Finding:** None of the five examples is newer than early ChatGPT. Generative AI "reached 53% adoption in three years",
    and coding assistants and agents are now the leading AI use among surveyed developers (66%).
- **Proposal:** Keep AlphaGo, AlphaFold and image generation, and add or swap in one example of today's LLMs or agents.
- **Evidence:** AI Index 2026, as in DEEP_LEARNING_SOFTWARE-06; Stack Overflow Developer Survey 2026, "AI",
    <https://survey.stackoverflow.co/2026/ai> (accessed 2026-11-24).

### DEEP_LEARNING_SOFTWARE-08 · Star history is old and argues against the next slide

- **Category:** outdated
- **Priority:** should
- **Pages:** 14
- **Finding:** The plot ends at the start of 2024 and names `google/jax`, now `jax-ml/jax`. Stars were 201,711 for
    TensorFlow, 105,703 for PyTorch and 36,677 for JAX, so they still rank TensorFlow first while p. 15 calls PyTorch
    "absolutely dominant".
- **Proposal:** Drop the plot, or re-plot it and say that stars measure attention over time rather than use.
- **Evidence:** GitHub API for `tensorflow/tensorflow`, `pytorch/pytorch`, `jax-ml/jax` (accessed 2026-11-24).

### DEEP_LEARNING_SOFTWARE-09 · "Dominant" rests on a 2022 chart with a dead link

- **Category:** outdated
- **Priority:** must
- **Pages:** 15
- **Finding:** The Hugging Face chart's "Reference" link (AssemblyAI, 2022) returns 404. The module uses a newer Linux
    Foundation survey figure: PyTorch 63%, TensorFlow 50%, JAX 6%. That is a lead, not the absolute dominance claimed;
    winning competition solutions show a bigger gap (44 PyTorch, 1 TensorFlow in 2025).
- **Proposal:** Re-plot the module's figure with its citation, add the competition count, and name the area each number
    covers. Say that the survey is n=113 organizations building generative models.
- **Evidence:** `s1_development_environment/deep_learning_software.md`, "Software Landscape for Deep Learning"; Linux
    Foundation 2024, Figure 9, <https://www.linuxfoundation.org/hubfs/LF%20Research/lfr_genai24_111924.pdf>; ML Contests
    2025, <https://mlcontests.com/state-of-machine-learning-competitions-2025/> (accessed 2026-11-24). A current share of
    research papers is `unverified`: only secondary blogs turned up.

### DEEP_LEARNING_SOFTWARE-10 · The three frameworks are not interchangeable

- **Category:** gap
- **Priority:** should
- **Pages:** 14
- **Finding:** The slide says all three "support the same underlying feature set". The module stresses their different
    programming paradigms (object-oriented vs functional). The ecosystem differs too: Transformers v5 is "sunsetting"
    Flax/TensorFlow support with PyTorch as the sole backend.
- **Proposal:** Keep the shared core (p. 10's three circles) and add one line on paradigm and one on ecosystem.
- **Evidence:** `s1_development_environment/deep_learning_software.md`, "Software Landscape for Deep Learning"; Hugging
    Face, "Transformers v5", <https://huggingface.co/blog/transformers-v5> (accessed 2026-11-24).

### DEEP_LEARNING_SOFTWARE-11 · High-level framework list has aged

- **Category:** outdated
- **Priority:** should
- **Pages:** 17
- **Finding:** Haiku's own README recommends Flax for new projects and describes maintenance mode. Sonnet's last release
    is from January 2024. Keras 3 now runs on JAX, TensorFlow and PyTorch. The module lists a different, PyTorch-only set.
- **Proposal:** Show fewer logos, grouped by base framework, with Haiku replaced by Flax and Sonnet dropped. Make
    Lightning stand out as the course's choice.
- **Evidence:** <https://github.com/google-deepmind/dm-haiku> README; `google-deepmind/sonnet` releases (GitHub API);
    <https://keras.io/keras_3/> (accessed 2026-11-24); `s4_debugging_and_logging/boilerplate.md`, "Minimizing
    boilerplate".

### DEEP_LEARNING_SOFTWARE-12 · Lightning has no real slide

- **Category:** gap
- **Priority:** should
- **Pages:** 17
- **Finding:** After the restructure the module is PyTorch + Lightning, but the deck shows only a logo and "We get back
    top one of these". The module explains boilerplate, `LightningModule` (`training_step`, `configure_optimizers`) and
    `Trainer`.
- **Proposal:** Add one or two slides after p. 17: what counts as boilerplate, and a plain training loop next to the
    Lightning version. Replace the "we get back" line.
- **Evidence:** [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653);
    `s4_debugging_and_logging/boilerplate.md`, "Minimizing boilerplate", "LightningModule" and "Trainer".

### DEEP_LEARNING_SOFTWARE-13 · When not to use Lightning

- **Category:** gap
- **Priority:** should
- **Pages:** 17
- **Finding:** The restructure makes Lightning optional but preferred and names Hugging Face projects as the case where
    another abstraction makes no sense. The deck has no Hugging Face. Nine of the 44 winning PyTorch solutions in 2025
    also used Lightning, so most did not.
- **Proposal:** Add a short "when to use which" line: plain PyTorch, Lightning, or Hugging Face libraries. Keep the
    "understand the underlying framework" advice (keep list).
- **Evidence:** [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653); ML
    Contests 2025, as in DEEP_LEARNING_SOFTWARE-09.

### DEEP_LEARNING_SOFTWARE-14 · Installing PyTorch is missing

- **Category:** gap
- **Priority:** should
- **Pages:** 15
- **Finding:** The module warns that a naive install takes around 6.6 GB against 729 MB for CPU only, and that students
    will reinstall it in environments, containers and CI. The deck says nothing about this.
- **Proposal:** Add one slide on choosing the CPU or GPU build, pointing to PyTorch's selector and the uv configuration.
    Re-measure the sizes before quoting them and don't hard-code a CUDA version.
- **Evidence:** `s1_development_environment/deep_learning_software.md`, "A note on installing PyTorch"; PyTorch 2.15.0
    release notes, <https://github.com/pytorch/pytorch/releases/tag/v2.15.0> (accessed 2026-11-24). Current install sizes
    are `unverified`.

### DEEP_LEARNING_SOFTWARE-15 · No link from deep learning to MLOps

- **Category:** gap
- **Priority:** should
- **Pages:** 5, 9
- **Finding:** The module opens with technical debt and explains that the course uses deep learning because data and model
    size bring specific challenges. The deck never says why an MLOps course starts with this.
- **Proposal:** Add a short bridge after p. 5: training the model is easy now, and running it in production is the rest
    of the course. Keep it light, because the MLOps introduction deck covers technical debt.
- **Evidence:** `s1_development_environment/deep_learning_software.md`, introduction; `slides/day2_introduction_to_mlops.pdf`
    (text: "Hidden Technical Debt").

### DEEP_LEARNING_SOFTWARE-16 · No slide on what students will do

- **Category:** gap
- **Priority:** should
- **Pages:** 18
- **Finding:** The module has a recurring final exercise (baseline `model.py`, `data.py`, `main.py`) that the rest of the
    course builds on, and insists on scripts rather than notebooks. The deck ends without mentioning it.
- **Proposal:** Add a closing slide that replaces p. 19 (see DEEP_LEARNING_SOFTWARE-25): refresh PyTorch, build the
    baseline in scripts, have it ready for the next session. Name Rock-Paper-Scissors only if it ships.
- **Evidence:** `s1_development_environment/deep_learning_software.md`, "❔ Final exercise";
    [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651).

### DEEP_LEARNING_SOFTWARE-17 · "More accelerators" shows 2021-era chips

- **Category:** outdated
- **Priority:** should
- **Pages:** 12, 18
- **Finding:** The examples are Apple M1 Pro/Max and an unnamed chip. PyTorch now has a device-neutral
    `torch.accelerator` API, and its 2.15 notes cover NVIDIA, AMD, Intel and Apple hardware. The module's solution already
    picks CUDA, MPS or CPU.
- **Proposal:** Make the point "same code, many devices": replace chip photos with a vendor list and the API.
- **Evidence:** PyTorch 2.15 docs, "torch.accelerator", <https://docs.pytorch.org/docs/2.15/accelerator.html>; 2.15.0
    release notes as in DEEP_LEARNING_SOFTWARE-14 (accessed 2026-11-24);
    `s1_development_environment/deep_learning_software.md`, "❔ Final exercise" step 3.

### DEEP_LEARNING_SOFTWARE-18 · Show torch.compile and point to the scaling deck

- **Category:** outdated
- **Priority:** should
- **Pages:** 18
- **Finding:** "Compiled models" is shown with a TensorRT diagram, though the course is on PyTorch and the scaling deck
    tells students to use `torch.compile`. That deck also covers mixed precision and quantization, so p. 18 is a preview.
- **Proposal:** Replace the diagram with a one-line `torch.compile` example and say that compilation and precision come
    back in the scaling session. No PyTorch module in the first session covers this slide.
- **Evidence:** `slides/day12_scaling_applications.pdf` (text: "Remember to compile your model", "Automatic Mixed
    Precision (AMP)"); PyTorch 2.15 docs, `torch.compile`,
    <https://docs.pytorch.org/docs/2.15/generated/torch.compile.html> (accessed 2026-11-24).

### DEEP_LEARNING_SOFTWARE-19 · Precision figure has wrong labels and no bfloat16

- **Category:** wrong
- **Priority:** must
- **Pages:** 18
- **Finding:** The figure prints "PF32" and "FP16.16" for FP32 and FP16. It also omits bfloat16, which PyTorch AMP supports
    and the Lightning module teaches.
- **Proposal:** Redraw the bit layouts with correct names and add bfloat16.
- **Evidence:** PyTorch 2.15 docs, "Automatic Mixed Precision package", <https://docs.pytorch.org/docs/2.15/amp.html>
    (accessed 2026-11-24); `s4_debugging_and_logging/boilerplate.md`, "❔ Exercises" step 10.

### DEEP_LEARNING_SOFTWARE-20 · Coding agents lower the barrier again

- **Category:** addition
- **Priority:** should
- **Pages:** 9, 17
- **Finding:** The barrier argument stops at frameworks. In the 2026 Stack Overflow survey 73% use coding assistants and
    agents daily, and the session's editor module tells students they are responsible for the code, not the AI tool. The
    p. 17 advice applies to agents as well.
- **Proposal:** Extend the story by one step (math, framework, high-level framework, agent) with the same rule:
    understand the layer below. No A-module reference; the course introduction deck covers the course's rules on AI use.
- **Evidence:** Stack Overflow 2026, as in DEEP_LEARNING_SOFTWARE-07; `s1_development_environment/editor.md`, "AI
    assistance"; `slides/day1_introduction_to_the_course.pdf`, p. 22;
    [Agentic modules for the MLOps course](https://github.com/SkafteNicki/dtu_mlops/issues/618).

### DEEP_LEARNING_SOFTWARE-21 · Back the barrier argument with literature

- **Category:** addition
- **Priority:** could
- **Pages:** 9
- **Finding:** The argument has no source. Raff reports a higher reproduction rate for GPU papers and suspects that
    frameworks such as PyTorch and TensorFlow, which make GPUs relatively easy to use, are the reason.
- **Proposal:** Cite it as a suspicion from an empirical study. The drawn curve stays a sketch without data.
- **Evidence:** `literature/A_step_toward_quantifying_independently_reproducible_machine_learning_research.pdf`,
    PDF p. 7.

### DEEP_LEARNING_SOFTWARE-22 · ImageNet chart needs sources

- **Category:** addition
- **Priority:** could
- **Pages:** 3
- **Finding:** The chart is uncited and mixes "Human" into the year axis. Checked values fit what the bars show: 28.2%
    (2010), 25.8% (2011), 16.4% (2012), 11.7% (2013), about 5.1% human, 3.57% (ResNet 2015) and 2.25% (SENet 2017). Two
    2012 AlexNet results are published (16.4% and 15.3%).
- **Proposal:** Re-plot with citations, say which 2012 result is used, and draw human error as a horizontal line.
- **Evidence:** Russakovsky et al. 2015, result tables, <https://arxiv.org/abs/1409.0575>; He et al. 2015,
    <https://arxiv.org/abs/1512.03385>; Hu et al., <https://arxiv.org/abs/1709.01507> (accessed 2026-11-24). The VGG 2014
    and 2016 ensemble bars are `unverified`.

### DEEP_LEARNING_SOFTWARE-23 · Language slips, some in keep-list quotes

- **Category:** wrong
- **Priority:** must
- **Pages:** 3, 7, 8, 9, 14, 15, 17
- **Finding:** Slips include "on in the imagenet benchmark" (p. 3), "Common agreed on factors" (p. 7), "Why do need", "log
    of simple math", "backpropgation" and "ourself" (p. 8), "Barrier to enter" and "mean it accessible" (p. 9), "Pytorch"
    and "Tensorflow" (pp. 14–15), "Recommend, to" and "We get back top one" (p. 17).
- **Proposal:** Fix spelling and grammar only, keeping wording and rhythm. This touches the keep-list quotes on pp. 3, 8,
    9 and 17, so it needs the author's sign-off.
- **Evidence:** Page text as listed; product spelling from `s1_development_environment/deep_learning_software.md`,
    "Software Landscape for Deep Learning".

### DEEP_LEARNING_SOFTWARE-24 · Framework-affair comic: keep, regenerate or swap

- **Category:** image
- **Priority:** should
- **Pages:** 16
- **Finding:** A seven-panel comic has PyTorch and TensorFlow as rivals in a love affair, ending "Remember, it's not a
    competition." It carries the slide's point, but its author and license are `unverified` and JAX is missing.
- **Proposal:** Author's call: find the source and permission, or regenerate the idea as an original comic, possibly with
    JAX in it.
- **Evidence:** [How should images be recreated, and what may we reuse from Gamma exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705)

### DEEP_LEARNING_SOFTWARE-25 · Drop the closing meme

- **Category:** image
- **Priority:** should
- **Pages:** 19
- **Finding:** The deck ends on "Meme of the day", which the cross-deck rules remove. The punchline uses Theano, and the
    image includes film stills and a photo of a real person.
- **Proposal:** Drop the page and end on the exercise slide (DEEP_LEARNING_SOFTWARE-16).
- **Evidence:** [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710#issuecomment-7114948971),
    cross-deck rule 2.

### Also noticed (no id, not decided)

- p. 4 "Stable diffusion": the astronaut-on-a-horse prompt is OpenAI's DALL·E 2 demo
    (<https://openai.com/index/dall-e-2/>, accessed 2026-11-24). Which model made this particular picture is `unverified`.
- p. 13 terminal screenshot shows Python 3.8.5; PyTorch 2.15 wheels need 3.11 or newer. It becomes a code block anyway.
- p. 13 could show the `zero_grad`/`backward`/`step` trio from the module's knowledge check.
- p. 12 could connect the matrix-multiplication figure to accelerators in one line.
- The session's command-line, package-manager and editor modules have no slides here. Whether they should is a question
    for [Which decks does the January 2027 course need?](https://github.com/SkafteNicki/dtu_mlops/issues/723)
- p. 1 is dated 2 January 2024; date the new deck January 2027.

## Module follow-ups (out of scope for the deck)

- `s1_development_environment/deep_learning_software.md`, "A note on installing PyTorch": pins `cu126`, which PyTorch 2.15
    no longer publishes (default is now `cu132`), so the command stays on 2.14 or older. The GB/MB sizes need re-measuring.
- `s1_development_environment/deep_learning_software.md`, "Software Landscape for Deep Learning": JAX link still
    `google/jax` (now `jax-ml/jax`), and the survey figure is from 2024.
- `s1_development_environment/deep_learning_software.md`, "Software Landscape for Deep Learning": says the deep learning
    book is in the literature folder, but `literature/deep_learning.pdf` (66 pages) has only the contents, website page
    and bibliography.
- `s1_development_environment/README.md`, package-manager card: says `conda` and `pip`, while the module recommends `uv`.
- `s4_debugging_and_logging/boilerplate.md`, "Minimizing boilerplate": Composer is archived; Catalyst's last release is
    from 2022.
- `s4_debugging_and_logging/boilerplate.md`, "❔ Exercises" (closing list): lightning-bolts is archived.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | "Deep Learning Timeline" 1940–2017, credited to Favio Vázquez | concept diagram | Mermaid/PlantUML/Excalidraw | stale | DEEP_LEARNING_SOFTWARE-05 |
| 3 | ImageNet thumbnail mosaic | photo | replaced by theme / licensed stand-in | | |
| 3 | Bar chart of ILSVRC top-5 error 2010–2017 | chart/paper figure | re-plot/redraw + citation | | DEEP_LEARNING_SOFTWARE-22 |
| 4 | Question-and-answer text under "ChatGPT" | tool screenshot | fresh screenshot, or code block + citation | stale | DEEP_LEARNING_SOFTWARE-01 |
| 4 | Ke Jie playing AlphaGo | photo | licensed stand-in / generated | | |
| 4 | Segmented street scenes with boxes and arrows | chart/paper figure | redraw + citation, or replace | | DEEP_LEARNING_SOFTWARE-02 |
| 4 | Astronaut on a horse | photo (AI-generated) | generated | | |
| 4 | AlphaFold architecture (embedding, trunk, heads) | chart/paper figure | re-plot/redraw + citation | | DEEP_LEARNING_SOFTWARE-03 |
| 5 | Grand View Research US market bars 2020–2030 | chart/paper figure | re-plot + citation (other data) | stale | DEEP_LEARNING_SOFTWARE-06 |
| 6–7 | Five red circles with brackets | concept diagram | Mermaid/PlantUML/Excalidraw, click build | | |
| 8 | NumPy `Linear` class | code/config screenshot | code block | | DEEP_LEARNING_SOFTWARE-04 |
| 9 | Hand-drawn adoption vs barrier line | concept diagram | Excalidraw | | DEEP_LEARNING_SOFTWARE-21 |
| 10–13 | Three red circles: tensors, acceleration, autodiff | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 11 | Tensors from rank 0 to rank 4 | concept diagram | Excalidraw | | |
| 12 | CPU, GPU and TPU product photos | photo | licensed stand-in / generated | stale | DEEP_LEARNING_SOFTWARE-17 |
| 12 | Matrix multiplication, row times column | concept diagram | Excalidraw | | |
| 13 | Forward and backward computational graph | chart/paper figure | redraw + citation (source `unverified`) | | |
| 13 | Terminal with `torch.ones(..., requires_grad=True)` | code/config screenshot | code block | stale | |
| 14 | Star History of TensorFlow, PyTorch, JAX | chart/paper figure | drop, or re-plot + citation | stale | DEEP_LEARNING_SOFTWARE-08 |
| 14, 15 | PyTorch, TensorFlow and JAX logos | tool logo | Iconify logo (`logos:pytorch-icon`, `logos:tensorflow`, `logos:jax`) | | |
| 15 | "Number of Models on HuggingFace" bars | chart/paper figure | re-plot + citation (other data) | stale | DEEP_LEARNING_SOFTWARE-09 |
| 16 | Seven-panel comic, PyTorch vs TensorFlow | meme/joke | keep / regenerate / swap? | voice | DEEP_LEARNING_SOFTWARE-24 |
| 17 | Lightning, fast.ai, Keras, Haiku, Ignite (identified by eye) and Sonnet logos | tool logo | Iconify logo where available (`simple-icons:keras` exists; others `unverified`), else text | stale | DEEP_LEARNING_SOFTWARE-11 |
| 18 | Apple M1 Pro/Max and an unidentified chip | photo | drop, or licensed stand-in | stale | DEEP_LEARNING_SOFTWARE-17 |
| 18 | Un-optimized vs TensorRT-optimized network | concept diagram | code block, or redraw + citation | stale | DEEP_LEARNING_SOFTWARE-18 |
| 18 | FP32, FP16 and INT8 bit layouts | concept diagram | Excalidraw | | DEEP_LEARNING_SOFTWARE-19 |
| 19 | Six-panel "What I actually do" deep learning meme (`from theano import *`) | meme/joke | drop (cross-deck rule) | voice, stale | DEEP_LEARNING_SOFTWARE-25 |
| 1–19 | DTU logo, red bars, footer | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo ·
meme/joke · decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated ·
replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| *Giving GPT-3 a Turing Test*, 2020 (author's name as in the URL) | <https://lacker.io/ai/2020/07/06/giving-gpt-3-a-turing-test.html> | DEEP_LEARNING_SOFTWARE-01 <!-- codespell:ignore --> |
| Ronneberger et al. 2015, *U-Net* | <https://arxiv.org/abs/1505.04597> | DEEP_LEARNING_SOFTWARE-02 |
| Jumper et al. 2021, *Highly accurate protein structure prediction with AlphaFold* | <https://www.nature.com/articles/s41586-021-03819-2> | DEEP_LEARNING_SOFTWARE-03 |
| Nobel Prize 2024, *Chemistry press release* | <https://www.nobelprize.org/prizes/chemistry/2024/press-release/> | DEEP_LEARNING_SOFTWARE-03 |
| Salakhutdinov & Hinton 2009, *Deep Boltzmann Machines* | <https://proceedings.mlr.press/v5/salakhutdinov09a.html> | DEEP_LEARNING_SOFTWARE-05 |
| Vaswani et al. 2017, *Attention Is All You Need* | <https://arxiv.org/abs/1706.03762> | DEEP_LEARNING_SOFTWARE-05 |
| Goodfellow et al. 2016, *Deep Learning*, section 1.2 | <https://www.deeplearningbook.org/contents/intro.html> | DEEP_LEARNING_SOFTWARE-05, p. 2 "Deep learning is old" |
| Stanford HAI, *AI Index 2026* | <https://hai.stanford.edu/ai-index/2026-ai-index-report/economy> | DEEP_LEARNING_SOFTWARE-06, DEEP_LEARNING_SOFTWARE-07 |
| Stack Overflow, *Developer Survey 2026: AI* | <https://survey.stackoverflow.co/2026/ai> | DEEP_LEARNING_SOFTWARE-07, DEEP_LEARNING_SOFTWARE-20 |
| Linux Foundation 2024, *Shaping the Future of Generative AI* | <https://www.linuxfoundation.org/hubfs/LF%20Research/lfr_genai24_111924.pdf> | DEEP_LEARNING_SOFTWARE-09 |
| ML Contests, *State of Machine Learning Competitions 2025* | <https://mlcontests.com/state-of-machine-learning-competitions-2025/> | DEEP_LEARNING_SOFTWARE-09, DEEP_LEARNING_SOFTWARE-13 |
| Hugging Face, *Transformers v5* | <https://huggingface.co/blog/transformers-v5> | DEEP_LEARNING_SOFTWARE-10 |
| Keras, *Keras 3* | <https://keras.io/keras_3/> | DEEP_LEARNING_SOFTWARE-11 |
| Google DeepMind, *Haiku* README | <https://github.com/google-deepmind/dm-haiku> | DEEP_LEARNING_SOFTWARE-11 |
| PyTorch, *2.15.0 release notes* | <https://github.com/pytorch/pytorch/releases/tag/v2.15.0> | DEEP_LEARNING_SOFTWARE-14, DEEP_LEARNING_SOFTWARE-17 |
| PyTorch docs, *torch.compile* | <https://docs.pytorch.org/docs/2.15/generated/torch.compile.html> | DEEP_LEARNING_SOFTWARE-18 |
| PyTorch docs, *Automatic Mixed Precision* | <https://docs.pytorch.org/docs/2.15/amp.html> | DEEP_LEARNING_SOFTWARE-19 |
| Raff 2019, *A Step Toward Quantifying Independently Reproducible ML Research* | `literature/` | DEEP_LEARNING_SOFTWARE-21 |
| Russakovsky et al. 2015, *ILSVRC* | <https://arxiv.org/abs/1409.0575> | DEEP_LEARNING_SOFTWARE-22, p. 3 |
| He et al. 2015, *Deep Residual Learning* | <https://arxiv.org/abs/1512.03385> | DEEP_LEARNING_SOFTWARE-22, p. 3 |
| Hu et al., *Squeeze-and-Excitation Networks* | <https://arxiv.org/abs/1709.01507> | DEEP_LEARNING_SOFTWARE-22, p. 3 |
