# Review brief: introduction_to_mlops (`slides/day2_introduction_to_mlops.pdf`)

- **Modules compared:** `s2_organisation_and_version_control/` (all modules) @ `main`
    d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b. The deck's own topic, what MLOps is, is covered in `pages/before.md`,
    and the course stack in `pages/overview.md`, so both were compared as well.
- **Other inputs:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf` (Sculley et al.),
    `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf` (Shankar et al.), web sources (accessed
    2026-11-24).
- **Pages:** 44 PDF pages → ~42 slides after merging click builds (pages 22=23 and, probably, 12=13: same title and
    text, emphasis added, different figure). Text was extracted with `pdftotext` and every page render was checked by
    eye. Four Gamma "Error uploading image" placeholders: p. 3 (1), p. 4 (1), p. 14 (2).

## Deck outline (narrative order to keep)

1. Title: page 1
2. AI is fantastic, errors happen, and production is the hard part: pages 2–6
3. AI is software, with software's last-mile and technical-debt problems: pages 7–9
4. DevOps as the starting point, with its definition and three ingredients: pages 10–14
5. MLOps defined, its three phases, and the MLOps engineer: pages 15–19
6. Why DevOps is not enough: data, then roles ("Where's Waldo?") and the engineer joke: pages 20–24
7. Why companies care: failures, jobs, pillars and payoffs, OpenAI and Google: pages 25–29
8. Tools: trend, 2006 vs today, picking a stack (4Cs), technical debt, too tool centric: pages 30–37
9. Case study, medical vial inspection: pages 38–42
10. Summing up, then the module list with the meme of the day: pages 43–44

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "Developing AI is easy – Running AI in production is hard" (p. 4)
    - "Because we teach people it!" (p. 5)
    - "This is what creates value" (p. 6, in the figure)
    - "This is both a joke and not." (p. 10)
    - "This is the closest to a definition that I could find" (p. 12)
    - "To MLOps (verb): To harmonize the creative process of data science with the discipline of software engineering,
        creating a continuous, automated loop that moves models from an experimental notebook to a reliable production
        environment." (p. 15)
    - "Because the devil is in the ~~details~~ data" (p. 21)
    - "Who could have foreseen that the company which have all possible advantages would win in the end?" (p. 29).
        INTRODUCTION_TO_MLOPS-12 asks whether it still holds.
    - "And most time this is not possible without actually trying to use both." (p. 35)
    - "In a nutshell MLOps is about dealing reducing technical debt" (p. 36). INTRODUCTION_TO_MLOPS-20 fixes the
        wording only.
    - "Summing up, MLOps at its core is..." with its three endings (p. 43). INTRODUCTION_TO_MLOPS-20 fixes "hole".
- Voice markers:
    - Linear teaching vs the real-world loop, same figure twice (p. 5–6)
    - ✅/❌ list for what AI inherits from software (p. 7)
    - DevOps/MLOps elephant meme (p. 10), see INTRODUCTION_TO_MLOPS-25
    - Technologies + Processes + Mindset = Value, which comes back with one box lit (p. 13, 30)
    - Where's Waldo? guessing game (p. 22–23)
    - Generated MLOps engineer: printed prompt and "Buff / Locked in / Many screens" (p. 24), see
        INTRODUCTION_TO_MLOPS-25
    - Sculley's boxes refilled with 2006 and current tools (p. 32, 34)
    - The 4Cs (p. 35)
    - Must/should requirements that come back ticked off in the case study (p. 40, 42)

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

- [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647): the course is built around closing the lifecycle loop.
    The deck's p. 6 loop is the natural place to introduce it. → see INTRODUCTION_TO_MLOPS-17. Module numbers and the
    session layout are not among that map's decisions yet, so use names, not M-numbers.
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    DVC goes away, and modules before Day 4 just drop their DVC exercises. DVC is on p. 34, 35 and 44, and in the p.
    33 landscape. → see INTRODUCTION_TO_MLOPS-01, INTRODUCTION_TO_MLOPS-07, INTRODUCTION_TO_MLOPS-09
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): version the
    pipeline, not the data. Page 27 says "Version everything—data, code, and models". → see INTRODUCTION_TO_MLOPS-17
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): Prefect 3, Label
    Studio Community and pandera. Label Studio is already on p. 34, Airflow is not the choice. → see
    INTRODUCTION_TO_MLOPS-07, INTRODUCTION_TO_MLOPS-08
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): loguru
    application logging moves into Good coding practice (today's session), and W&B moves to Day 3. → see
    INTRODUCTION_TO_MLOPS-01, INTRODUCTION_TO_MLOPS-16
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651) and [Find a
    maintained label-quality tool to replace cleanlab](https://github.com/SkafteNicki/dtu_mlops/issues/652): the deck
    names no course dataset or label-quality tool, so no slide changes.
- [Lock the M/O/A taxonomy and naming convention](https://github.com/SkafteNicki/dtu_mlops/issues/619): module kinds
    become M/O/A. The closing list should use names only. → see INTRODUCTION_TO_MLOPS-01
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621) and
    [Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681): lectures
    include agentic content but do not otherwise reference the A-modules. → see INTRODUCTION_TO_MLOPS-18

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased (e.g. CLOUD-01) -->

### INTRODUCTION_TO_MLOPS-01 · Closing module list is incomplete and lists DVC

- **Category:** outdated
- **Priority:** must
- **Pages:** 44
- **Finding:** The list has Git, Code structure, Data Version Control and Command Line Interfaces. The session also
    has Good Coding Practice, which the list omits, and Data Version Control is being removed.
- **Proposal:** List the restructured session by module name: Git, Code organization, Good coding practice (now with
    loguru logging), Command line interfaces. Keep the link.
- **Evidence:** `s2_organisation_and_version_control/README.md`, module cards; [Decide what replaces the DVC content
    in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650); [Decide where each current Day 4
    module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653).

### INTRODUCTION_TO_MLOPS-02 · The data-effort slide misreports its sources

- **Category:** wrong
- **Priority:** must
- **Pages:** 20
- **Finding:** The InfoQ report of the cited talk gives 85% as a failure rate from "historical studies", not as models
    that never reach production. Rexer's 32% is the share of people who say their models are usually deployed, not a
    pilot-to-production rate. The "50-80%" links don't back it: the Interreg page has no such figure and the ML6 page
    returns 404. The Zi quote is also not the published wording.
- **Proposal:** Reword the two numbers to what the sources say, quote Zi as published or drop the quote marks, and
    find a working source for 50-80% or drop it.
- **Evidence:** InfoQ 2024-11-22, <https://www.infoq.com/news/2024/11/why-ml-fails/>; Rexer Analytics,
    <https://www.rexeranalytics.com/data-science-survey>; Siegel 2024,
    <https://www.kdnuggets.com/survey-machine-learning-projects-still-routinely-fail-to-deploy>;
    <https://www.interregeurope.eu/embraisme>; <https://www.ml6.eu/blogpost/unlocking-the-full-potential-of-data>
    (404) (accessed 2026-11-24). The $3.4B and 31.1% match P&S Intelligence,
    <https://www.psmarketresearch.com/market-analysis/mlops-market>.

### INTRODUCTION_TO_MLOPS-03 · The course's own paper calls the failure statistic misguided

- **Category:** addition
- **Priority:** should
- **Pages:** 8, 20
- **Finding:** Both pages lean on "most models never reach production". Shankar et al. say of the 90% version that
    "this statistic is misguided", because constant experimentation should leave only a small fraction of versions in
    production. They name Velocity, Validation and Versioning as what decides success.
- **Proposal:** Keep the hard-truth slide, but add this counterpoint and, if space allows, the three Vs (they fit next
    to p. 27). Cite the "never create value" line on p. 8 or soften it.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, abstract and §1 (p. 1–2), §4.2.

### INTRODUCTION_TO_MLOPS-04 · Knight Capital was not a model, and the citation is misspelt

- **Category:** wrong
- **Priority:** must
- **Pages:** 25
- **Finding:** The slide introduces Knight Capital as an "automated model deployed with errors"; the SEC describes
    incorrectly deployed code in an order router. The quote also differs from the paper ("$465 million", not "$465
    millions") and the title is "Technical Debt", not "Technical depth".
- **Proposal:** Keep the quote, corrected, as an automated-system failure that MLOps practice (deployment checks,
    dead-code removal) would catch. Fix the title.
- **Evidence:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, "Dead Experimental Codepaths"; SEC
    2013-222, <https://www.sec.gov/newsroom/press-releases/2013-222> (search excerpt only, the page returns 403)
    (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-05 · Jobs chart is a 2018 forecast that ended in 2025

- **Category:** outdated
- **Priority:** should
- **Pages:** 25
- **Finding:** The chart is ©2018 Gartner and forecasts jobs lost and gained from AI until 2025. All its years are
    past, and it does not support the slide's point about the cost of errors.
- **Proposal:** Drop the chart.
- **Evidence:** Page 25 render (copyright and axis); Gartner 2017-12-13,
    <https://www.gartner.com/en/newsroom/press-releases/2017-12-13-gartner-says-by-2020-artificial-intelligence-will-create-more-jobs-than-it-eliminates>
    (search excerpt only) (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-06 · Case-study pipeline figures sit under the wrong headings

- **Category:** wrong
- **Priority:** must
- **Pages:** 41
- **Finding:** The figure under Containerization is the serving application (ECS, a GPU instance, FastAPI,
    CloudWatch). The figure under Inference Service is the CI pipeline: commit pushed, build the serving, trainer and
    TensorBoard images, push to ECR. The two are swapped.
- **Proposal:** Swap the first and third figures, redraw all three at readable size, and name the third box after what
    it shows (image builds in CI).
- **Evidence:** Page 41 render, figure labels.

### INTRODUCTION_TO_MLOPS-07 · DVC is shown as data verification, and is being removed

- **Category:** wrong
- **Priority:** must
- **Pages:** 34, 35
- **Finding:** The stack puts DVC on Sculley's "Data Verification" box. DVC versions data and models; it doesn't
    verify them. It also leaves the course.
- **Proposal:** Put pandera on Data Verification and remove DVC from both pages.
- **Evidence:** DVC README, <https://github.com/treeverse/dvc> (accessed 2026-11-24);
    `s2_organisation_and_version_control/dvc.md`, "DVC: What is it?"; [Choose the data-engineering tool
    stack](https://github.com/SkafteNicki/dtu_mlops/issues/649); [Decide what replaces the DVC content in each
    affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650).

### INTRODUCTION_TO_MLOPS-08 · The "MLOps now" stack is not the course's stack

- **Category:** outdated
- **Priority:** should
- **Pages:** 34, 35
- **Finding:** Besides DVC, the stack shows Airflow (the course chose Prefect 3), Datadog (the course uses Evidently,
    OpenTelemetry and Cloud Monitoring) and Kubernetes (only in the extra session). Page 35 also shows Amazon
    SageMaker, renamed Amazon SageMaker AI on 2024-12-03.
- **Proposal:** Refill the boxes with the tools the course teaches, so the slide doubles as a course map, and update
    the SageMaker name if it stays.
- **Evidence:** `pages/overview.md`, framework table; `s10_extra/kubernetes.md`; [Choose the data-engineering tool
    stack](https://github.com/SkafteNicki/dtu_mlops/issues/649); AWS,
    <https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html> (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-09 · The tool landscape is several years old

- **Category:** outdated
- **Priority:** should
- **Pages:** 33
- **Finding:** The landscape still shows Neptune (hosted service wound down on 2026-03-05 after the OpenAI
    acquisition), Tecton (joining Databricks, announced 2025-08-22) and DVC as an independent project (bought by
    lakeFS, 2025-11-18). Other logos were not checked.
- **Proposal:** Replace it with a small redrawn landscape of current tools by lifecycle stage. The churn itself is a
    good example of the p. 35 criteria.
- **Evidence:** Neptune, <https://docs.neptune.ai/transition_hub>; Databricks,
    <https://www.databricks.com/blog/tecton-joining-databricks-power-real-time-data-personalized-ai-agents>; lakeFS,
    <https://lakefs.io/media-mentions/lakefs-acquires-dvc-uniting-data-version-control-pioneers/> (accessed
    2026-11-24).

### INTRODUCTION_TO_MLOPS-10 · The hype cycle is from July 2021

- **Category:** outdated
- **Priority:** should
- **Pages:** 30
- **Finding:** The figure is marked "As of July 2021" and places MLOps just past the peak. The slide uses it for a
    present-tense trend. Gartner's public 2026 write-up is about controlling AI agents; a 2026 position for MLOps was
    not found.
- **Proposal:** Date it as history (redraw only MLOps and ModelOps on the curve) or replace it with a 2026 trend,
    keeping "Tools have been the main priority".
- **Evidence:** Page 30 render; Gartner, <https://www.gartner.com/en/articles/hype-cycle-for-artificial-intelligence>
    (search excerpt only) (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-11 · DevOps is expanded and dated inconsistently

- **Category:** wrong
- **Priority:** must
- **Pages:** 10, 12
- **Finding:** Page 10 says "Developer operations", while p. 12 combines software development and IT operations. Page
    10 dates the rise to 2007/2008; the first DevOpsDays was in 2009. Page 12's definition has no source.
- **Proposal:** Expand DevOps as development + operations, date the popularity to 2009 (DevOpsDays), and cite a
    definition next to the author's line on p. 12.
- **Evidence:** Wikipedia, *DevOps*, <https://en.wikipedia.org/wiki/DevOps> (secondary); AWS, *What is DevOps?*,
    <https://aws.amazon.com/devops/what-is-devops/> (accessed 2026-11-24). The 2007/2008 date as the start of the
    movement is `unverified`.

### INTRODUCTION_TO_MLOPS-12 · OpenAI won, Google is winning: check both claims

- **Category:** outdated
- **Priority:** should
- **Pages:** 28, 29
- **Finding:** "Service contract with Microsoft" predates the April 2026 amendment: the licence is now non-exclusive
    and OpenAI can serve on any cloud. "Google is winning right now" has no metric; late-2026 reports disagree on who
    leads, depending on users, revenue or enterprise use.
- **Proposal:** Keep the two-slide arc, label the Microsoft bullet as historical, and either give Google a dated
    metric or make "who is winning?" an open class question. The p. 29 quote is on the keep list, so this is the
    author's call.
- **Evidence:** OpenAI 2026-04-27, <https://openai.com/index/next-phase-of-microsoft-partnership/> (accessed
    2026-11-24). The state of the race is `unverified` (aggregator sites only).

### INTRODUCTION_TO_MLOPS-13 · Phase slides copy ml-ops.org without credit

- **Category:** wrong
- **Priority:** must
- **Pages:** 15–18
- **Finding:** The three phases' bullets and the hand-drawn figure match ml-ops.org (INNOQ, CC BY 4.0), which asks for
    attribution; the deck gives none. The deck calls phase two "Modelling" where its figure and the module say "Model
    development". Page 18 zooms the figure so the Operations loop it talks about is cut off.
- **Proposal:** Credit ml-ops.org, use "Model development", and redraw the figure so each slide highlights its own
    loop.
- **Evidence:** ml-ops.org, <https://ml-ops.org/content/mlops-principles> and licence note on <https://ml-ops.org/>
    (accessed 2026-11-24); `pages/before.md`, "MLOps: What is it?".

### INTRODUCTION_TO_MLOPS-14 · Borrowed figures without a source

- **Category:** addition
- **Priority:** should
- **Pages:** 19, 21–23, 32, 43
- **Finding:** Page 21 is the figure of Breck et al.'s ML Test Score, p. 22–23 match the roles chart in the Full Stack
    Deep Learning 2022 notes, p. 32 and 34 reuse Sculley et al. Fig. 1 without the p. 9 citation, and p. 43 is
    Dorard's Machine Learning Canvas v1.1. Sources for p. 5–6, 11–13 and 19 were not found.
- **Proposal:** Redraw with inline citations; the author supplies the unknown sources.
- **Evidence:** Breck et al. 2017, Fig. 1,
    <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>;
    FSDL 2022 lecture 8, <https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/> (text only, chart not
    compared); OWNML, <https://www.ownml.co/machine-learning-canvas> (accessed 2026-11-24);
    `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, Figure 1. Page 5–6, 11–13, 19 origins
    `unverified`.

### INTRODUCTION_TO_MLOPS-15 · Case-study results need explaining

- **Category:** wrong
- **Priority:** should
- **Pages:** 42
- **Finding:** The heatmaps that stand for "give reason for prediction" show lions, a dog and a bird, not vials. The <
    0.6 s requirement is failed although the plot shows a mean of 0.42 s with a few vials above the line. "Must not
    approve flawed vials" rests on 71 defective test vials.
- **Proposal:** Mark the heatmaps as method illustration or use vial heatmaps, say that the latency misses are
    outliers, and say "none approved in test". The author confirms that the company and thesis material may be shown.
- **Evidence:** Page 42 render: figure, latency legend (0.42 s) and confusion table (25 + 46 defects, none predicted
    Good by EfficientNet-B0). Source of the case and permission `unverified`.

### INTRODUCTION_TO_MLOPS-16 · Nothing connects the lecture to today's modules

- **Category:** gap
- **Priority:** should
- **Pages:** 27, 37, 44
- **Finding:** The deck argues for version control, reproducibility and collaboration, but goes from the case study
    straight to a list of links. The session's reason is that differences in how people organize and write code must
    be minimized in large groups, and its learning objectives are not shown.
- **Proposal:** Before the module list, tie the pillars to the session: Git for collaboration, code organization and
    good coding practice against technical debt, CLIs for automation.
- **Evidence:** `s2_organisation_and_version_control/README.md`, introduction and "Learning objectives";
    `code_structure.md`, introduction and Big Ball of Mud quote; `cli.md`, "Non-Python code".

### INTRODUCTION_TO_MLOPS-17 · Introduce the course's lifecycle loop and pipeline versioning

- **Category:** addition
- **Priority:** should
- **Pages:** 6, 27
- **Finding:** Page 6 already shows deploy, monitor and user feeding back into data collection, but not labelling new
    data and rerunning the pipeline as what the course builds. Page 27's "Version everything—data" conflicts with
    versioning the pipeline over immutable raw data.
- **Proposal:** Label p. 6 as the loop the course closes (deploy, collect, label, rerun, retrain, redeploy) and reword
    p. 27 to versioned code and pipeline with lineage to the exact data.
- **Evidence:** [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647); [How do modern ML pipelines version
    data?](https://github.com/SkafteNicki/dtu_mlops/issues/648); Google Cloud, continuous training,
    <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
    (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-18 · Generative AI and agents are missing from the picture of MLOps

- **Category:** addition
- **Priority:** should
- **Pages:** 30–37
- **Finding:** An introduction for 2027 never mentions operating LLM systems, or agents as tools for MLOps work. AWS
    and NVIDIA describe GenAIOps as extending MLOps (prompts, evaluation, guardrails, agents). DORA's 2025 report
    finds that AI mainly amplifies an organization's existing strengths and weaknesses, which fits p. 37.
- **Proposal:** Add one slide: what stays and what is added for GenAI/agent operations, and agents as tools that
    amplify the process. No A-module references.
- **Evidence:** AWS,
    <https://aws.amazon.com/blogs/machine-learning/operationalizing-generative-ai-how-it-differs-from-mlops>; NVIDIA,
    <https://developer.nvidia.com/blog/mastering-llm-techniques-llmops>; DORA 2025,
    <https://dora.dev/research/2025/dora-report/> (accessed 2026-11-24); [Update lecture slides for the January 2027
    course](https://github.com/SkafteNicki/dtu_mlops/issues/681).

### INTRODUCTION_TO_MLOPS-19 · Governance pillar without the EU AI Act

- **Category:** addition
- **Priority:** could
- **Pages:** 27
- **Finding:** Governance is one line. The EU AI Act became applicable on 2026-08-02, and after the AI Omnibus the
    rules for high-risk use cases apply from 2027-12-02 (2028-08-02 for systems built into products).
- **Proposal:** Add one line with the dates as the concrete European example of why logging, monitoring and
    documentation matter.
- **Evidence:** European Commission, *AI Act*,
    <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai> (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-20 · Misspelt key terms

- **Category:** wrong
- **Priority:** must
- **Pages:** 3, 7, 25, 36, 43
- **Finding:** "Technical Dept" is in the p. 36 title and the ghost says "Technical depth"; p. 3 says "unpresented
    scale"; p. 7 "inhere"; p. 43 "hole pipeline". Some titles have agreement errors (p. 3 "errors does", p. 25 "does
    companies").
- **Proposal:** Fix the spelling throughout. The author says whether the ghost's "depth" is a deliberate pun before it
    is changed (keep-list wording on p. 36 and 43).
- **Evidence:** Page renders; `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, title; Fowler 2019,
    *Technical Debt*, <https://martinfowler.com/bliki/TechnicalDebt.html> (accessed 2026-11-24).

### INTRODUCTION_TO_MLOPS-21 · Broken image among the AI failures

- **Category:** image
- **Priority:** must
- **Pages:** 3
- **Finding:** The third example is an "Error uploading image" placeholder. The other two are Tay's 2016 tweets, which
    show slurs in full, and a car crash.
- **Proposal:** Replace it with a sourced, recent AI failure, preferably from generative AI, and paraphrase or blur
    the Tay tweets.
- **Evidence:** Page 3 render; [How should images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

### INTRODUCTION_TO_MLOPS-22 · Broken image on the duality of AI

- **Category:** image
- **Priority:** must
- **Pages:** 4
- **Finding:** The only figure is an "Error uploading image" placeholder. The lost content is unknown.
- **Proposal:** Draw a simple contrast (notebook vs production system), or let the keep-list line stand alone. The
    author may remember the original.
- **Evidence:** Page 4 render; [How should images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

### INTRODUCTION_TO_MLOPS-23 · Two broken icons in the DevOps ingredients

- **Category:** image
- **Priority:** must
- **Pages:** 14
- **Finding:** The process and mindset icons are "Error uploading image" placeholders, and the surviving technology
    icon is dark on dark. The three rows can be recovered from p. 13.
- **Proposal:** Redraw with the three icons from the p. 13 equation as theme icons.
- **Evidence:** Page 13 and 14 renders; [How should images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

### INTRODUCTION_TO_MLOPS-24 · Drop the closing meme

- **Category:** image
- **Priority:** should
- **Pages:** 44
- **Finding:** The module list is titled "Meme of the day" with an altered owlturd.com comic carrying a clear.ml link.
    Decks no longer end on a meme slide.
- **Proposal:** Retitle the slide as the module list and drop the comic. If the author wants the technical-debt joke,
    recreate it at p. 36.
- **Evidence:** Page 44 render; [Which Day 7 Cloud review changes do we
    accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710), cross-deck rules.

### INTRODUCTION_TO_MLOPS-25 · Four joke images: keep, regenerate or swap

- **Category:** image
- **Priority:** should
- **Pages:** 10, 20, 24, 36
- **Finding:** The elephant meme (p. 10) is a cartoon screenshot and carries "This is both a joke and not". The xkcd
    strip (p. 20) is 1838, *Machine Learning*, which is CC BY-NC 2.5. The MLOps engineer (p. 24) was generated and can
    be regenerated from its printed prompt. The ghost (p. 36) depends on the pun.
- **Proposal:** Author's call per image. Suggested: redraw the elephant and penguin joke, keep xkcd with a credit
    link, regenerate p. 24 from the same prompt, redraw the ghost.
- **Evidence:** xkcd, <https://xkcd.com/1838/> and <https://xkcd.com/license.html> (accessed 2026-11-24); [How should
    images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705). The cartoon source on p. 10 is `unverified`.

### Also noticed (no id, not decided)

- p. 1 title is "Day2 - Introduction to MLOps", dated January 2026; topic-only naming gives "Introduction to MLOps",
    January 2027. `pages/timeplan.md` calls the lecture "MLOps: what is it?".
- p. 2: the Commission defines Industry 5.0 by human-centricity, sustainability and resilience
    (<https://research-and-innovation.ec.europa.eu/research-area/industrial-research-and-innovation/industry-50_en>,
    accessed 2026-11-24). It names AI as a supporting initiative, not a key component.
- p. 5 "90% of time" on model development has no source (`unverified`) and differs from p. 20's data-effort claim.
- p. 7 "30+ years" undersells it: the NATO software engineering conference was in 1968
    (<https://en.wikipedia.org/wiki/Software_engineering>, secondary, accessed 2026-11-24).
- p. 9: the three cards are written as if from Sculley et al., but the paper's terms are entanglement, unstable data
    dependencies and configuration debt. The citation needs "only a small fraction" with the figure, as decided for
    the Cloud deck.
- p. 8: the iceberg puts "Wall of Confusion" between data science and operations, and p. 11 uses the same term for Dev
    and Ops; one sentence linking them would help.
- p. 19 lists "Software developing" where the figure says DevOps.
- p. 26 is a fair summary of Cho's post (2024-12-21), but reads as two-year-old news; date it.
- p. 32 "ca. 2006" and the all-Java stack have no source (`unverified`).
- The restaurant analogy in `pages/before.md` is not in the deck and could open p. 15.

## Module follow-ups (out of scope for the deck)

- `s2_organisation_and_version_control/README.md`, "Learning objectives" and cards: drop DVC, and add an objective for
    command line interfaces, which has none (the rest is planned in the restructure).
- `s2_organisation_and_version_control/cli.md`, "Non-Python code": lists `dvc` as an encountered CLI tool.
- `s2_organisation_and_version_control/dvc.md`: the Google Drive workaround goes away with the module.
- `pages/overview.md`, framework table: drop DVC; "Weights and Bias" should be Weights & Biases.
- `pages/timeplan.md`, 5/1/27 row: lists DVC, and the lecture title differs from the deck.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | Early ChatGPT start page | tool screenshot | fresh screenshot | stale | |
| 2 | Self-driving segmentation; chest X-ray heatmap | photo | licensed stand-in / generated | | |
| 3 | Tay tweets (2016), slurs visible | tool screenshot | redraw as paraphrased text | | INTRODUCTION_TO_MLOPS-21 |
| 3 | Multi-car crash in a tunnel | photo | licensed stand-in | | INTRODUCTION_TO_MLOPS-21 |
| 3 | Gamma "Error uploading image" | — | propose replacement | broken | INTRODUCTION_TO_MLOPS-21 |
| 4 | Gamma "Error uploading image" | — | propose replacement | broken | INTRODUCTION_TO_MLOPS-22 |
| 5–6 | Linear project chain, then the loop with deploy, monitor and user | concept diagram | Mermaid/PlantUML/Excalidraw | | INTRODUCTION_TO_MLOPS-14, INTRODUCTION_TO_MLOPS-17 |
| 8 | Iceberg: data science, wall of confusion, deployment, IT operations | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 9 | Sculley et al. Fig. 1 | chart/paper figure | re-plot/redraw + citation | | |
| 10 | Cartoon meme: DevOps elephant, ML penguin, MLOps hybrid | meme/joke | keep / regenerate / swap? | voice | INTRODUCTION_TO_MLOPS-25 |
| 11–12 | Dev and Ops figures with a wall, then four arrows (misspelt label) | concept diagram | Mermaid/PlantUML/Excalidraw | | INTRODUCTION_TO_MLOPS-14 |
| 13 | Technologies + processes + mindset = value icons | concept diagram | Mermaid/PlantUML/Excalidraw | | INTRODUCTION_TO_MLOPS-23 |
| 14 | Two Gamma "Error uploading image" placeholders and one dark icon | — | propose replacement | broken | INTRODUCTION_TO_MLOPS-23 |
| 15–18 | Hand-drawn MLOps triple loop (ml-ops.org); p. 18 cropped | concept diagram | Mermaid/PlantUML/Excalidraw + citation | | INTRODUCTION_TO_MLOPS-13 |
| 19 | Venn diagram: ML, DevOps, data engineering | concept diagram | Mermaid/PlantUML/Excalidraw | | INTRODUCTION_TO_MLOPS-14 |
| 20 | xkcd 1838, Machine Learning | meme/joke | xkcd with credit link | voice | INTRODUCTION_TO_MLOPS-25 |
| 21 | Traditional vs ML testing and monitoring (Breck et al.) | chart/paper figure | re-plot/redraw + citation | | INTRODUCTION_TO_MLOPS-14 |
| 22–23 | Roles chart, empty then filled | chart/paper figure | re-plot/redraw + citation | | INTRODUCTION_TO_MLOPS-14 |
| 24 | Generated MLOps engineer at eight screens | meme/joke | generated | voice | INTRODUCTION_TO_MLOPS-25 |
| 25 | Gartner jobs bar chart, 2015–2025 | chart/paper figure | drop | stale | INTRODUCTION_TO_MLOPS-05 |
| 30 | Gartner hype cycle, July 2021 | chart/paper figure | re-plot/redraw + citation | stale | INTRODUCTION_TO_MLOPS-10 |
| 30 | Icon equation with Technologies lit | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 31 | Glowing circuit-board MLOps art | decorative | replaced by theme | | |
| 32 | Sculley boxes with Java labels | concept diagram | redraw + citation | | INTRODUCTION_TO_MLOPS-14 |
| 33 | Tool landscape by lifecycle stage | concept diagram | Iconify logo + redraw | stale | INTRODUCTION_TO_MLOPS-09 |
| 34 | Sculley boxes with tool logos | concept diagram | Iconify logo + redraw + citation | stale | INTRODUCTION_TO_MLOPS-07, INTRODUCTION_TO_MLOPS-08 |
| 35 | Logos on a stone stack, with SageMaker | concept diagram | Iconify logo + redraw | stale | INTRODUCTION_TO_MLOPS-07, INTRODUCTION_TO_MLOPS-08 |
| 36 | Ghost saying "Technical depth" | meme/joke | keep / regenerate / swap? | voice | INTRODUCTION_TO_MLOPS-20, INTRODUCTION_TO_MLOPS-25 |
| 38 | Robot arm over a vial line | photo | licensed stand-in / generated | | |
| 39 | Three vial crops: good, particle, chips and cracks | photo | author's original, if permitted | | INTRODUCTION_TO_MLOPS-15 |
| 40 | Vial line with time limits | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 41 | Serving, training and CI build diagrams | concept diagram | Mermaid/PlantUML | | INTRODUCTION_TO_MLOPS-06 |
| 42 | Latency scatter and confusion table | chart/paper figure | re-plot/redraw + citation (table as a real table) | | INTRODUCTION_TO_MLOPS-15 |
| 42 | Heatmaps on animals | chart/paper figure | author's originals, or redraw + citation | | INTRODUCTION_TO_MLOPS-15 |
| 43 | Machine Learning Canvas v1.1 (Dorard) | chart/paper figure | re-plot/redraw + citation | | INTRODUCTION_TO_MLOPS-14 |
| 44 | Technical-debt mask comic, altered | meme/joke | keep / regenerate / swap? (drop proposed) | voice | INTRODUCTION_TO_MLOPS-24 |
| 1–44 | Dark Gamma background, p. 7 side art, card and bullet icons, Gamma badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo · meme/joke
· decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo · fresh
screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated · replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | INTRODUCTION_TO_MLOPS-04, INTRODUCTION_TO_MLOPS-14, p. 9, 25, 32, 34 |
| Shankar et al., *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | INTRODUCTION_TO_MLOPS-03, p. 8 and 20 |
| Hoblitzell 2024, *QCon SF 2024 - Why ML Projects Fail to Reach Production* | <https://www.infoq.com/news/2024/11/why-ml-fails/> | INTRODUCTION_TO_MLOPS-02, p. 20 "85%" and quote |
| Rexer Analytics, *Data Science Survey* | <https://www.rexeranalytics.com/data-science-survey> | INTRODUCTION_TO_MLOPS-02, p. 20 "32%" |
| Siegel 2024, *Survey: Machine Learning Projects Still Routinely Fail to Deploy* | <https://www.kdnuggets.com/survey-machine-learning-projects-still-routinely-fail-to-deploy> | INTRODUCTION_TO_MLOPS-02, p. 20 "32%" |
| P&S Intelligence, *MLOps Market* | <https://www.psmarketresearch.com/market-analysis/mlops-market> | INTRODUCTION_TO_MLOPS-02, p. 20 "$3.4B" |
| SEC 2013, *Press release 2013-222* | <https://www.sec.gov/newsroom/press-releases/2013-222> | INTRODUCTION_TO_MLOPS-04, p. 25 Knight Capital |
| Visengeriyeva et al. (INNOQ), *MLOps Principles*, CC BY 4.0 | <https://ml-ops.org/content/mlops-principles> | INTRODUCTION_TO_MLOPS-13, p. 15–18 |
| Breck et al. 2017, *The ML Test Score* | <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/> | INTRODUCTION_TO_MLOPS-14, p. 21 |
| Full Stack Deep Learning 2022, *Lecture 8: ML Teams and Project Management* | <https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/> | INTRODUCTION_TO_MLOPS-14, p. 22–23 |
| Dorard 2015, *Machine Learning Canvas* | <https://www.ownml.co/machine-learning-canvas> | INTRODUCTION_TO_MLOPS-14, p. 43 |
| Cho 2024, *i sensed anxiety and frustration at NeurIPS'24* | <https://kyunghyuncho.me/i-sensed-anxiety-and-frustration-at-neurips24/> | p. 26 summary |
| OpenAI 2026, *The next phase of the Microsoft OpenAI partnership* | <https://openai.com/index/next-phase-of-microsoft-partnership/> | INTRODUCTION_TO_MLOPS-12, p. 28 |
| AWS, *What is DevOps?* | <https://aws.amazon.com/devops/what-is-devops/> | INTRODUCTION_TO_MLOPS-11, p. 12 |
| AWS, *Operationalizing generative AI: How it differs from MLOps* | <https://aws.amazon.com/blogs/machine-learning/operationalizing-generative-ai-how-it-differs-from-mlops> | INTRODUCTION_TO_MLOPS-18 |
| DORA 2025, *State of AI-assisted Software Development* | <https://dora.dev/research/2025/dora-report/> | INTRODUCTION_TO_MLOPS-18, p. 37 |
| European Commission, *AI Act* | <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai> | INTRODUCTION_TO_MLOPS-19, p. 27 Governance |
| Fowler 2019, *Technical Debt* | <https://martinfowler.com/bliki/TechnicalDebt.html> | INTRODUCTION_TO_MLOPS-20, p. 36 |
| Munroe, xkcd 1838, *Machine Learning* | <https://xkcd.com/1838/> | INTRODUCTION_TO_MLOPS-25, p. 20 |
