# Review brief: project (`slides/day5_project.pdf`)

- **Modules compared:** `pages/projects.md`, `reports/` (`README.md`, `report.py`), `pages/faq.md`,
    `pages/timeplan.md`, `pages/before.md`, `canvas/`, `s2_organisation_and_version_control/code_structure.md` @
    `main` d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b. Pending text was also read on `dev` @ 6b1b567.
- **Other inputs:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`,
    `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, GitHub API and web sources (accessed
    2026-11-24).
- **Pages:** 22 PDF pages → 22 slides as exported (21 after PROJECT-23). There are no click builds, only the repeated
    title on pages 11–13, which are three different screenshots. There are no Gamma "Error uploading image"
    placeholders. Text extraction drops the 💡, ⚠️, ✅, 📊 and similar bullet emojis and turns p. 15's ellipses into "&".

## Deck outline (narrative order to keep)

1. Title: page 1
2. The job, and why the model is not the point: pages 2–4
3. Open-source frameworks to start from (landscape, three examples, 80/20): pages 5–9
4. First task and where ideas come from: pages 10–13
5. General recommendations: page 14
6. Summary of today's work and the canvas: pages 15–16
7. Checklist and evaluation: pages 17–18
8. Report template and hand-in: pages 19–20
9. Agent helper: page 21
10. Meme and project link: page 22

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "You (and your group) are just hired as an MLOps engineers at a start-up." (p. 2)
    - "Importantly: You are judged not by how great the model is but how fast you can setup a pipeline to solve the
        task." (p. 2). PROJECT-07 proposes a change.
    - "Why you do not need to care about the model?" (p. 3)
    - "You already have all the tools for the pipeline, you just need a good starting model." (p. 4). PROJECT-06
        proposes a change.
    - "A open-source framework can usually get you 80% of the way" (p. 9). PROJECT-19 proposes a change.
    - "Find your sweet spot!" (p. 10)
    - "You do not need to do everything to pass, the list is meant to be exhaustive" (p. 17)
    - "A little helper for you guys" and "*If I find time* I will update this to be an agent skill that dynamically
        can interact with the content of the course" (p. 21). PROJECT-05 proposes a change.
- Voice markers:
    - The start-up job framing that opens the deck (p. 2)
    - The red "Fast track this part" box on the MLOps loop (p. 4). PROJECT-11 proposes a change.
    - Emoji bullets: 💡 for the job, ✅ and ⚠️ for what is and is not evaluated (pp. 2, 4, 18)

## Pending restructure

<!-- decisions from the restructure and agentic maps that touch this deck -->

- [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647): the checklist must match the course content, including
    closing the loop, and the project page and report template are still to be updated. → see PROJECT-09, PROJECT-10,
    PROJECT-12
- [Design the week-3 loop closure](https://github.com/SkafteNicki/dtu_mlops/issues/655): open; decides what the
    checklist requires for the second pass. → see PROJECT-10
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): data
    reproducibility means a versioned pipeline with append-only `raw/`. → see PROJECT-11, PROJECT-12
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): Prefect 3, Label
    Studio and pandera become tools a project can use. The deck names no orchestrator, so no extra slide is proposed;
    the checklist carries it. → see PROJECT-09
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    DVC leaves the modules. Checklist items are a separate, still-open follow-up. The deck has no DVC text. → see
    PROJECT-05, PROJECT-12
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651):
    Rock-Paper-Scissors is the exercise dataset and does not bind projects. Pending a real-photo check. → see
    PROJECT-22
- [Find a maintained label-quality tool to replace cleanlab](https://github.com/SkafteNicki/dtu_mlops/issues/652):
    cleanlab stays pinned and is at most a sub-exercise; nothing on this deck.
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): Lightning is
    optional, and profiling and logging move, so the checklist order changes. → see PROJECT-09
- [Lock module numbering for the new layout](https://github.com/SkafteNicki/dtu_mlops/issues/656): open; module
    numbers on the checklist change. The checklist cites M and O only, and prose cites modules by name. → see
    PROJECT-09
- [Lock the M/O/A taxonomy and naming convention](https://github.com/SkafteNicki/dtu_mlops/issues/619): three module
    kinds; agentic modules are never on the checklist. → see PROJECT-09
- [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682): section 6 is the agent use
    rule and trace export. → see PROJECT-02, PROJECT-03, PROJECT-05
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621):
    original decision and reasoning for the rule and its five packs. → see PROJECT-02
- [Reconcile existing agentic branch work with the A-module
    plan](https://github.com/SkafteNicki/dtu_mlops/issues/623): the trace export is per student for the whole course,
    and its wording is on `dev`. → see PROJECT-03
- [Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681): the
    lecture may carry agentic content, but the slides do not otherwise reference A-modules. → limits PROJECT-02,
    PROJECT-03

## Findings

<!-- max ~25; ids are stable: PROJECT-<nn> -->

### PROJECT-01 · Wrong exam deadline

- **Category:** outdated
- **Priority:** must
- **Pages:** 19
- **Finding:** The deck gives the scrape as 23/1 at 23:59. The 2027 course ends on Friday 22 January 2027, and the
    hand-in is midnight on the final day.
- **Proposal:** Give the 2027 deadline with year and weekday, and keep the rule that later changes are not registered.
    The date needs the author's confirmation.
- **Evidence:** `pages/timeplan.md`, "Week 3"; `pages/projects.md`, "Exam" (still says the 23rd of January 2026).

### PROJECT-02 · Agent use rule is missing

- **Category:** gap
- **Priority:** must
- **Pages:** 18, 21
- **Finding:** The project deck explains the evaluation and offers agent helpers but never states the agent use rule.
    Agents are optional; a group that uses them must use one of the five approved skill packs, commit its config and
    answer a report question about it. Self-built packs need the course responsible's exception.
- **Proposal:** Add one slide with the rule, its reason (skill pack = agentic engineering, not vibe coding) and a link
    to the project page. Name the five packs; do not reference A-modules.
- **Evidence:** [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), section 6;
    [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621),
    resolution. The five packs' repositories: candidate references (accessed 2026-11-24). The wording is not on `main`
    yet, so re-check at execution.

### PROJECT-03 · Agent trace hand-in is missing

- **Category:** gap
- **Priority:** must
- **Pages:** 19, 20
- **Finding:** The deck lists two deliverables: today's description and the report. Every student who used an agent
    anywhere in the course also hands in a trace export on DTU Learn, per student and not per group, before the report
    deadline.
- **Proposal:** Add it as a third deliverable, with who, what and when, and link the export instructions. Say that
    agent sessions must start inside the course folder, so students do so from day 5 on.
- **Evidence:** [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), section 6;
    [Reconcile existing agentic branch work with the A-module
    plan](https://github.com/SkafteNicki/dtu_mlops/issues/623), resolution; `dev` @ 6b1b567, `pages/projects.md`,
    "Hand-in of coding agent traces", and `reports/export_agent_data.py` (not on `main`).

### PROJECT-04 · The helper command points at a redundant branch

- **Category:** outdated
- **Priority:** should
- **Pages:** 21
- **Finding:** The slide installs from `--checkout code_helpers`. That branch was squash-merged into `master` on
    2026-01-12: their file trees are identical today, and the leftover branch will not follow later template fixes.
- **Proposal:** Drop `--checkout` and use the module's command. State that agent support is a template option.
- **Evidence:** `s2_organisation_and_version_control/code_structure.md`, "❔ Exercises" 2;
    `SkafteNicki/mlops_template`, file trees of `master` and `code_helpers`, squash-merge commit #93,
    `cookiecutter.json` (accessed 2026-11-24).

### PROJECT-05 · Reconcile the helper slide with the agent use rule

- **Category:** structure
- **Priority:** should
- **Pages:** 21
- **Finding:** AGENTS.md and a course agent are course-provided helpers, not one of the approved skill packs, so
    presenting them as "a little helper" can read as satisfying the rule. The bundled agent also teaches DVC and old
    module numbers.
- **Proposal:** Say next to the helper that it does not count as a skill pack, and place it directly after PROJECT-02.
    Reword the "If I find time" promise (keep list) as shipped or drop it: author's call.
- **Evidence:** [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), section 6;
    `mlops_template` `master`, `.github/agents/dtu_mlops_agent.md` (DVC, M1x references) (accessed 2026-11-24);
    [Decide what replaces the DVC content in each affected
    module](https://github.com/SkafteNicki/dtu_mlops/issues/650).

### PROJECT-06 · "You already have all the tools" is not true on day 5

- **Category:** wrong
- **Priority:** must
- **Pages:** 4
- **Finding:** The slide says students already have all the tools for the pipeline. On day 5 they have only had week
    1; CI, cloud and deployment come in week 2, and monitoring in week 3.
- **Proposal:** Say they have the week-1 foundations and that the pipeline grows as the course goes on. Keep the point
    that a good starting model is the first need.
- **Evidence:** `pages/timeplan.md`, "Week 1" to "Week 3"; `pages/projects.md`, "Project checklist".

### PROJECT-07 · "Judged on how fast" contradicts the evaluation

- **Category:** wrong
- **Priority:** must
- **Pages:** 2, 18
- **Finding:** Page 2 says students are judged on how fast they can set up a pipeline. Speed is not a criterion on
    page 18 or on the project page, which asks for the tools to be used in a meaningful way.
- **Proposal:** Keep the hired-at-a-start-up framing and the "not how great the model is" contrast (keep list), and
    replace the speed half with the stated criterion.
- **Evidence:** `pages/projects.md`, "Project days" ("incorporate the tools that are taught throughout the course in a
    meaningful way"); deck p. 18.

### PROJECT-08 · Dataset size advice contradicts the project page

- **Category:** drift
- **Priority:** must
- **Pages:** 14
- **Finding:** The deck recommends <10 GB, while the project page recommends starting with less than 1 GB and
    subsampling. The many-small-files warning is also missing.
- **Proposal:** Use the page's 1 GB start-out advice, with one line on small files. If 10 GB is meant as an upper
    bound, say so on both.
- **Evidence:** `pages/projects.md`, "Project hints", "Data".

### PROJECT-09 · Checklist screenshot is old

- **Category:** outdated
- **Priority:** must
- **Pages:** 17
- **Finding:** The screenshot is the week-1 list from an earlier version: it names conda and `make_dataset.py`, which
    the current list has replaced with a generic environment and `data.py`. It also lacks module references, CLI,
    logging and sweeps.
- **Proposal:** Replace it with a live link and a short per-week overview (code, not a picture), so it cannot go
    stale. Wait for the updated checklist.
- **Evidence:** `pages/projects.md`, "Project checklist", "Week 1"; [Restructure the MLOps course for January 2027 to
    close the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/647), "Not yet specified" (checklist
    update).

### PROJECT-10 · Close the lifecycle loop in the project brief

- **Category:** gap
- **Priority:** should
- **Pages:** 2, 4, 18
- **Finding:** The deck stops at a deployed model. The restructure makes the project's checklist match a loop: deploy,
    collect new data, label, rerun the pipeline, retrain, redeploy. The current week-3 checklist already asks for
    input-output collection and drift detection.
- **Proposal:** Frame the job as a pipeline that survives a second pass, on p. 2 and p. 4, and ask groups to choose a
    problem where new data can be collected. Wait for the week-3 design before naming steps.
- **Evidence:** [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647), Destination and Notes; [Design the week-3 loop
    closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (open); `pages/projects.md`, "Project checklist",
    "Week 3".

### PROJECT-11 · "Fast track this part" box cuts out data engineering

- **Category:** gap
- **Priority:** should
- **Pages:** 4
- **Finding:** The red box fast-tracks Design and Model Development, including Data Engineering and Data Availability
    Check. Under the restructure, data foundations get their own day and the versioned pipeline is the spine of the
    project.
- **Proposal:** Keep the joke that the model is fast-tracked, but draw the box around model choice only. Redraw the
    figure with a credit to ml-ops.org (footer: CC BY 4.0).
- **Evidence:** [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647), Notes and [How do modern ML pipelines version
    data?](https://github.com/SkafteNicki/dtu_mlops/issues/648); figure source
    <https://ml-ops.org/content/mlops-principles> (accessed 2026-11-24). That the hand-drawn figure is theirs is by
    visual match only: `unverified`.

### PROJECT-12 · Evaluation slide: data versioning wording and missing monitoring

- **Category:** gap
- **Priority:** should
- **Pages:** 18
- **Finding:** The four criteria stop at deployment. The report also asks about monitoring, testing and cloud costs,
    and "data … version controlled" will mean a versioned pipeline rather than DVC.
- **Proposal:** Add monitoring/closing the loop as a criterion if the author agrees, and describe data reproducibility
    through the pipeline. The criteria are the author's, so this needs sign-off.
- **Evidence:** `reports/README.md`, Questions 7, 8, 25–27; [How do modern ML pipelines version
    data?](https://github.com/SkafteNicki/dtu_mlops/issues/648), resolution; [Decide what replaces the DVC content in
    each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650).

### PROJECT-13 · Generative AI and contribution statements are not mentioned

- **Category:** gap
- **Priority:** should
- **Pages:** 18, 19
- **Finding:** The report must state individual contributions and if and how generative AI was used (Question 31). DTU
    Library says use must be distinctly declared and is the author's responsibility. The deck mentions neither.
- **Proposal:** Add one line on each to the report or evaluation slide: own words, declared use, individual
    contributions from the git history.
- **Evidence:** `reports/README.md`, Question 31; `pages/faq.md`, "Can I use ChatGPT"; DTU Library, "How to declare
    the use of Generative AI" (accessed 2026-11-24).

### PROJECT-14 · Description requirements are incomplete

- **Category:** gap
- **Priority:** should
- **Pages:** 15, 20
- **Finding:** The project page requires at least 300 words and the link to the GitHub repository handed in on DTU
    Learn as a group. The slides give no length, and p. 20 shows the link only inside a 2024 screenshot. The deck's
    extra request for number of samples, size and modality is not on the page.
- **Proposal:** State length, place and format once, and align the data-description line with the page.
- **Evidence:** `pages/projects.md`, "Project days", "Day 1".

### PROJECT-15 · Group size and finding a group are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 20
- **Finding:** Nothing says groups are 3 to 5 or what to do without one. The hand-in slide covers only members missing
    from DTU Learn.
- **Proposal:** Add group size and how to find a group. `pages/faq.md` and the project page agree.
- **Evidence:** `pages/projects.md`, introduction; `pages/faq.md`, "How many should we be in a group".

### PROJECT-16 · Schedule and expected scope are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 10, 15
- **Finding:** The summary ends at "Work on the rest of project...". The page gives six project days, what each is
    for, and the advice to start small and parallelize while everyone understands the whole.
- **Proposal:** Add one timeline of the six project days (8, 14, 15, 20, 21 and 22 January 2027) with the deliverables
    on it.
- **Evidence:** `pages/projects.md`, "Project days"; `pages/timeplan.md`, "Week 1" to "Week 3".

### PROJECT-17 · Transformers description is out of date

- **Category:** outdated
- **Priority:** must
- **Pages:** 6
- **Finding:** The slide says NLP models for PyTorch, Jax and TensorFlow. Version 5 removed TensorFlow and Jax, and
    the project describes itself as text, vision, audio, video and multimodal.
- **Proposal:** Describe it as a PyTorch, multimodal model library. Retake the screenshot.
- **Evidence:** huggingface/transformers, `MIGRATION_GUIDE_V5.md`, "Removal of TensorFlow and Jax", README, and
    release v5.24.0 (accessed 2026-11-24).

### PROJECT-18 · Report: required structure and check command

- **Category:** drift
- **Priority:** should
- **Pages:** 19
- **Finding:** The tree shows `src/__init__.py`, but the template nests a package under `src/<project_name>/`. The
    project page requires only the root README and `reports/`, says to copy the folder first, and the template's
    `reports/` holds only placeholders. The "public" requirement is stated on the slide only.
- **Proposal:** Show the minimum structure, say to copy `reports/` first and to run `report.py check`, and settle
    "public" on both. How `data/` looks follows the restructure.
- **Evidence:** `pages/projects.md`, "Exam"; `reports/README.md`, introduction; `mlops_template` `master` file tree
    (accessed 2026-11-24).

### PROJECT-19 · The 80/20 claim has no source

- **Category:** addition
- **Priority:** should
- **Pages:** 9
- **Finding:** "80% of the way" is presented as a fact, and the remaining 20% includes deployment, integration and
    monitoring. Sculley et al. stress how much is left around a generic package: a mature system may be at most 5% ML
    code and at least 95% glue code. The 80% figure itself is `unverified`.
- **Proposal:** Keep it as an informal rule of thumb, about the model, and cite Sculley et al. for why the pipeline is
    most of the work. Trim the generic three-card text.
- **Evidence:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, "Glue Code".

### PROJECT-20 · Credit the role chart

- **Category:** addition
- **Priority:** could
- **Pages:** 3
- **Finding:** The bubble chart matches the Full Stack Deep Learning 2022 chart as described there (horizontal axis
    ML, bubble size communication and technical writing), but is uncited. "ML research" in the line above should read
    as a role.
- **Proposal:** Redraw with a citation, and make the subtitle's wording consistent with the chart.
- **Evidence:** FSDL 2022, Lecture 8, "Skills Required" (accessed 2026-11-24). The match is by axes and labels only.

### PROJECT-21 · Ground "smaller models first" in practice

- **Category:** addition
- **Priority:** could
- **Pages:** 14
- **Finding:** The curve chart (throughput, data, performance) has no source, and its shape is presented as general;
    origin `unverified`. Shankar et al. report that many practitioners prefer simple models whenever possible.
- **Proposal:** Redraw as an explicitly schematic sketch and cite Shankar et al. for the advice. Add the page's
    fine-tuning tip.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, section 4.5.5 "Keep it Simple";
    `pages/projects.md`, "Project hints", "Modelling".

### PROJECT-22 · Example projects and past groups

- **Category:** gap
- **Priority:** could
- **Pages:** 11–13
- **Finding:** The three "good idea" slides are three screenshots with one caption between them. The project page
    links four example projects, the newest pushed in January 2023. Students may also not know to check licence and
    collectability of a dataset.
- **Proposal:** Merge the three into one slide (GitHub dependents, Kaggle, Hugging Face datasets) with a line on each,
    and add a pointer to the example projects. Suggest the question "could you collect more of this data?" for the
    loop.
- **Evidence:** `pages/projects.md`, introduction; GitHub API push dates (accessed 2026-11-24); [Restructure the MLOps
    course for January 2027 to close the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/647),
    Destination; [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651)
    (capturable data is why that dataset was chosen).

### PROJECT-23 · Drop the closing meme

- **Category:** structure
- **Priority:** must
- **Pages:** 22
- **Finding:** The last page is a "Meme of the day" (an Incredibles still with a machine-learning caption) and a link
    to the project page.
- **Proposal:** Drop the meme and end on the project page link, per the cross-deck rule.
- **Evidence:** [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710),
    cross-deck rule 2; [How should a review brief for one deck be
    structured?](https://github.com/SkafteNicki/dtu_mlops/issues/702).

### PROJECT-24 · Canvas: say it is optional and how it is used

- **Category:** drift
- **Priority:** could
- **Pages:** 16
- **Finding:** The canvas has its own slide without saying it is optional. The page marks it optional and part of the
    README if filled out. Its Data, Monitoring and Feedback boxes are the loop's questions.
- **Proposal:** Mark it optional and point at those boxes as the link to the loop.
- **Evidence:** `pages/projects.md`, "Day 1" item 3; `canvas/canvas_en.pdf`.

### Also noticed (no id, not decided)

- p. 1: the title is "Day5 - Project" and January 2026; use "Project" and January 2027.
- Spelling: "healtcare" (p. 7), "an good idea" (pp. 11–13), "continues integration" (p. 18), "Pytorch" (pp. 4–6), "A
    open-source" (p. 9), "you reports" (p. 19).
- p. 8: PyTorch Geometric's own line is "structured data"; "irregular data such as graphs and points" is fine but
    older. p. 5: the landscape now lists 134 entries (<https://landscape.pytorch.org/data/full.json>, accessed
    2026-11-24), including Transformers, MONAI and PyTorch Geometric.
- pp. 6–8: three examples with a screenshot each could be one slide.
- p. 11: "used by" is GitHub's sidebar name for the dependents view, and is still shown on kornia's page (accessed
    2026-11-24).

## Module follow-ups (out of scope for the deck)

- `pages/projects.md`, "Exam": deadline still says January 2026, the 23rd ("23th"); should be 22 January 2027.
- `pages/faq.md`, "When will the exam take place": deadline still says January 2025, the 24th.
- `pages/projects.md`, "Day 4": calls the day the second last project day; it is the third last of six (20, 21 and 22
    January 2027).
- `pages/projects.md`, "Project days", introduction: "Approximately 1/3 of the course time" is 6 of 15 days in
    `pages/timeplan.md`.
- `pages/projects.md`, "Project hints" and `reports/README.md`, Question 10: still build on DVC, and `gsutil` is
    named.
- `reports/README.md`, "Week 1": profiling is cited as M12; `mkdocs.yml` and `pages/projects.md` have M13. The two
    checklists must be updated together.
- `pages/projects.md`, "Exam": does not say the repository must be public, while the deck does.
- `pages/projects.md` and `reports/README.md`: need the agent use rule, a skill-pack report question and trace export
    (`dev` has only the export). `pages/faq.md`'s ChatGPT answer needs to point at the rule.
- `mlops_template`, `.github/agents/dtu_mlops_agent.md`: embeds a copy of the checklist with DVC and old module
    numbers (separate repository).

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 3 | Role chart: software engineering vs ML | chart/paper figure | re-plot + citation | | PROJECT-20 |
| 4 | Design/development/operations loops, red box | concept diagram | Excalidraw redraw + credit | | PROJECT-11 |
| 5 | PyTorch landscape page | tool screenshot | fresh screenshot | stale | |
| 6 | Transformers README (says "PyTorch and TensorFlow 2.0") | tool screenshot | fresh screenshot | stale | PROJECT-17 |
| 7 | MONAI GitHub page | tool screenshot | fresh screenshot | stale | |
| 8 | PyTorch Geometric README (KDD Cup 2021) | tool screenshot | fresh screenshot | stale | |
| 9 | Three generated tech-art pictures | decorative | replaced by theme | | |
| 10 | Dataset, model and scale icons | decorative | replaced by theme | | |
| 11 | kornia GitHub page with "Used by" | tool screenshot | fresh screenshot | stale | PROJECT-22 |
| 12 | Kaggle start page | tool screenshot | fresh screenshot | stale | PROJECT-22 |
| 13 | Hugging Face datasets page | tool screenshot | fresh screenshot | | PROJECT-22 |
| 14 | Throughput/data/performance curves | chart/paper figure | re-draw as schematic | | PROJECT-21 |
| 16 | MLOps canvas v1.1 | concept diagram | own work: export from `canvas/canvas_en.pdf` | | PROJECT-24 |
| 17 | Old week-1 checklist | tool screenshot | replace by link and overview | stale | PROJECT-09 |
| 19 | Repository tree | code/config screenshot | code block | | PROJECT-18 |
| 19 | Report template on GitHub | tool screenshot | fresh screenshot | stale | PROJECT-18 |
| 20 | DTU Learn groups and a 2024 submission | tool screenshot | fresh screenshot, redacted | stale | PROJECT-14 |
| 21 | Cookiecutter command | code/config screenshot | code block | | PROJECT-04 |
| 22 | Incredibles meme, "it defines who I am" | meme/joke | drop | voice | PROJECT-23 |
| 1–22 | Background, emoji bullets, Gamma badge | decorative | replaced by theme (emojis stay as text) | | |

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | `literature/Hidden_technical_debt_in_machine_learning_systems.pdf` | PROJECT-19, p. 9 80/20 |
| Shankar et al. 2022, *Operationalizing Machine Learning* | <https://arxiv.org/abs/2209.09125> | PROJECT-21, p. 14 |
| Full Stack Deep Learning 2022, Lecture 8 | <https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/> | PROJECT-20, p. 3 |
| ml-ops.org, *MLOps Principles* | <https://ml-ops.org/content/mlops-principles> | PROJECT-11, p. 4 |
| Hugging Face, *Transformers v5 migration guide* | <https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md> | PROJECT-17, p. 6 |
| DTU Library, *How to declare the use of Generative AI* | <https://www.bibliotek.dtu.dk/en/publishing/reference-management/kunstig-intelligens/eksempler> | PROJECT-13 |
| mlops_template, squash-merge #93 | <https://github.com/SkafteNicki/mlops_template/pull/93> | PROJECT-04 |
| Skill pack: Pocock | <https://github.com/mattpocock/skills> | PROJECT-02 |
| Skill pack: Osmani | <https://github.com/addyosmani/agent-skills> | PROJECT-02 |
| Skill pack: pstack (port) | <https://github.com/michael-denyer/pstack-claude> | PROJECT-02 |
| Skill pack: superpowers | <https://github.com/obra/superpowers> | PROJECT-02 |
| Skill pack: spec-kit | <https://github.com/github/spec-kit> | PROJECT-02 |
