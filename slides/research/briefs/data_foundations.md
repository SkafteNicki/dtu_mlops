# Material brief: data_foundations (new deck, no source PDF)

- **Baseline:** `main` @ d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b
- **Slot:** 30–45 min lecture opening Day 4, Thursday 7 January 2027 (`pages/timeplan.md`). It comes after the W&B
    logging move to Day 3 and before CI (11 January), cloud (12 January) and monitoring (18 January).
- **Repo inputs:** `literature/`; `docs/agents/research-*.md` on `research/pipeline-versioning`,
    `research/data-engineering-tool-stack`, `research/label-quality-tool` and `research/lifecycle-dataset` (called
    **PV**, **TS**, **LQ** and **LD** below); the pushed briefs; `slides/day4_debugging_ML_code.pdf` (pages 2, 3, 8,
    10, 11, 13 and 18–23 checked as renders, the rest as text).
- **Web sources:** accessed 2026-11-24. The four research branches did their own checks on 2026-10-21; items resting
    on them say so.
- **How to read this:** topics are raw material in no fixed order and the IDs don't imply one. Priorities are a
    recommendation. There is no slide copy or outline here; the author decides those in [What does the data
    foundations lecture contain?](https://github.com/SkafteNicki/dtu_mlops/issues/749).

## Decided, and not decided

Decided by the restructure, so the deck can rely on it:

- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): version the
    pipeline. Append-only `raw/`, deterministic code, W&B reference artifacts for lineage. Table formats are a
    mention.
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): Prefect 3, Label
    Studio Community, pandera; a Cloud Run job on GCP.
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651):
    Rock-Paper-Scissors (RPS), pending the real-photo check.
- [Find a maintained label-quality tool to replace cleanlab](https://github.com/SkafteNicki/dtu_mlops/issues/652):
    cleanlab 2.9.0, at most a sub-exercise.
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    DVC is gone. Cloud, GitHub Actions and CML reuse the Day 4 mechanism.
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): W&B is taught on
    Day 3, so students know runs and artifacts before this lecture.
- [Which decks does the January 2027 course need?](https://github.com/SkafteNicki/dtu_mlops/issues/723): the debugging
    deck is not converted, and the loop closure lives in `monitoring`.

Not decided, so the deck must not assume an answer; see [Dependencies](#dependencies-on-open-tickets).

## Candidate topics

### DATA_FOUNDATIONS-01 · Data work is undervalued, and it costs you

- **Priority:** must
- **Gist:** Open on "Everyone wants to do the model work, not the data work": 53 interviewed practitioners, and data
    cascades with 92% prevalence, which the authors call invisible, delayed, but often avoidable.
- **Why:** The strongest sourced hook, and it needs no tool knowledge. It replaces the weakly sourced 50–80% effort
    figures that INTRODUCTION_TO_MLOPS-02 flags.
- **Evidence:** Sambasivan et al. 2021 (abstract). 92% is a share within this interview study, not of all projects.

### DATA_FOUNDATIONS-02 · Data is a dependency without a compiler

- **Priority:** must
- **Gist:** "Data Dependencies Cost More than Code Dependencies": unstable and underutilized inputs, with little
    static analysis to help. Code has lockfiles and type checkers; input data usually has neither.
- **Why:** Bridges from Days 1–3 (code and environments pinned) to today: the one input that is still unpinned.
- **Evidence:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, §3 (p. 3).

### DATA_FOUNDATIONS-03 · Silent failure: data bugs don't crash

- **Priority:** must
- **Gist:** A crash stops you, a bad batch trains. Errors can show as a small quality drop, easily missed if only some
    slices are affected; interviewees describe hard, soft and drift errors.
- **Why:** The reason validation exists, and the natural carry-over of the debugging deck's "Silent Failures" (p. 8).
- **Evidence:** Breck et al. 2019, §1; `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, §5.1.2
    (p. 9–10).

### DATA_FOUNDATIONS-04 · Where data sits in the loop

- **Priority:** must
- **Gist:** One map of the course: deploy, collect, label, rerun the pipeline, retrain, redeploy. Today builds the
    part that is rerun.
- **Why:** The restructure's litmus test is whether something serves the loop. Students otherwise see Day 4 as a
    detour.
- **Evidence:** [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647); Shankar et al., abstract (p. 1): MLOps as a continual
    loop that starts with data collection and labeling.

### DATA_FOUNDATIONS-05 · The pipeline in four verbs: ingest, validate, process, log

- **Priority:** must
- **Gist:** The shape students build: build a manifest of all raw batches, check it, transform deterministically,
    record what came out. Validation fails the run before anything is logged.
- **Why:** The lecture's concrete object; the topics below hang on it. Google gives the same order: validate before
    training and stop on schema skew.
- **Evidence:** TS, "How the stack composes"; Google Cloud, "Data and model validation". Module steps are unsettled
    (see Dependencies).

### DATA_FOUNDATIONS-06 · Ingest: raw is immutable and append-only

- **Priority:** must
- **Gist:** New data is a new batch under `raw/`; nothing is edited in place. Raw retains history, so everything
    downstream can be rebuilt.
- **Why:** The one rule the rest rests on, and what makes the second pass easy.
- **Evidence:** PV §1; Databricks (bronze keeps the raw state, "is appended incrementally and grows over time",
    enables reprocessing). Databricks says append-only; "immutable" is the discipline added on top.

### DATA_FOUNDATIONS-07 · Version the pipeline, not the data

- **Priority:** must
- **Gist:** Data is a function of immutable raw inputs, versioned code and parameters. Store the function and the
    inputs, and recompute the rest.
- **Why:** The central idea of the day, and what replaces DVC. It also corrects "Version everything—data"
    (INTRODUCTION_TO_MLOPS-17).
- **Evidence:** PV, "Recommended mental model"; [How do modern ML pipelines version
    data?](https://github.com/SkafteNicki/dtu_mlops/issues/648). Beauchemin's essay is the usual source, but it
    returned 403 and only search summaries were seen: wording `unverified`.

### DATA_FOUNDATIONS-08 · Why not DVC, and when data versioning is right

- **Priority:** should
- **Gist:** An honest counterpoint: recompute breaks if the transform isn't deterministic, raw is deleted, or
    recompute is too costly. Then you pin outputs. Snapshot tools are one way to give data an immutable ID.
- **Why:** Students meet DVC in older projects. "Never version data" would be wrong, and Sculley et al. describe
    versioned copies with their costs.
- **Evidence:** PV §2, §5 and trade-offs; Sculley et al. §3 (p. 3); lakeFS acquired DVC (announcement title; date not
    checked here).

### DATA_FOUNDATIONS-09 · Lineage: which data made this model?

- **Priority:** must
- **Gist:** raw → pipeline run → processed → training run → model. Reference artifacts store metadata; with bucket
    versioning on, download returns the object versions as they were when logged.
- **Why:** The answer to "can you rebuild the model from last month?", using the Day 3 tool.
- **Evidence:** W&B, "Track external files" and "Explore artifact lineage graphs"; Google Cloud, "Metadata
    management"; PV §3–4.

### DATA_FOUNDATIONS-10 · Determinism is the price

- **Priority:** should
- **Gist:** Pipeline versioning only holds if the same inputs give the same outputs: seeded splits, pinned
    dependencies, no live API calls, stable file order.
- **Why:** The failure mode most likely to break a student pipeline without notice; ties back to Day 3 (seeds,
    lockfiles).
- **Evidence:** PV §2 and "Trade-offs"; REPRODUCIBILITY_AND_SOFTWARE-11 and -15. Stable file order is added here and
    `unverified`.

### DATA_FOUNDATIONS-11 · Validation: unit tests for data

- **Priority:** must
- **Gist:** A schema codifies what correct data is: columns, types, allowed labels, ranges, uniqueness. Checks run
    inside the pipeline and a failure stops it.
- **Why:** The course's validation decision in one concept, with production evidence from Google and Amazon.
- **Evidence:** Breck et al. 2019 (schema, deployed in TFX); Schelter et al. 2018 ("unit tests" for data); Shankar et
    al. §4.5.4 (p. 7: bounds, completeness, schema checks); pandera docs.

### DATA_FOUNDATIONS-12 · Validating images: check the manifest

- **Priority:** should
- **Gist:** Images aren't tables, so derive one: path, hash, size, channels, label, batch. Then "allowed labels, no
    duplicate hashes, every image labelled" are ordinary schema checks.
- **Why:** The RPS exercise needs it, and students otherwise conclude that validation is only for tabular data.
- **Evidence:** TS, "Image-data note"; [Choose the data-engineering tool
    stack](https://github.com/SkafteNicki/dtu_mlops/issues/649).

### DATA_FOUNDATIONS-13 · Look at your data anyway

- **Priority:** must
- **Gist:** Summary statistics, label distribution and a few samples, before any schema. Schemas only encode what
    someone thought of.
- **Why:** Carry-over of debugging p. 10. Hand-written constraints alone were reported as unsustainable when their
    owners left.
- **Evidence:** `slides/day4_debugging_ML_code.pdf` p. 10; Shankar et al. §5.1.2 (p. 9–10).

### DATA_FOUNDATIONS-14 · Data leakage

- **Priority:** must
- **Gist:** Leakage kinds tied to the pipeline step that prevents them: split before fitting, split by group,
    deduplicate by hash, respect time. Add serving-time leakage: training on data that does not exist when serving.
- **Why:** Carry-over of debugging p. 11. A leaky split is a pipeline bug, so it belongs here. Errors were found in 17
    fields, affecting 329 papers.
- **Evidence:** Kapoor & Narayanan 2022 (abstract, 8 leakage types); Shankar et al. §5.1.1 (p. 8); debugging deck p.
    11.

### DATA_FOUNDATIONS-15 · False alarms: how strict is strict enough?

- **Priority:** could
- **Gist:** Too many checks produce false positives. With many columns, the chance that at least one violates a
    constraint is high. Stop on hard errors, report soft ones.
- **Why:** Gives a decision rule rather than a list of checks; sets up the "Goldilocks" point in monitoring.
- **Evidence:** Shankar et al. §5.1.2 (p. 9–10). The stop/report split is this brief's reading; Google says stop on
    schema skew and retrain on value skew.

### DATA_FOUNDATIONS-16 · Orchestration: from script to flow

- **Priority:** must
- **Gist:** An orchestrator adds retries, state tracking, caching and scheduling around functions that already exist.
    In Prefect it starts as decorators on `data.py`.
- **Why:** The "pipelines" part of Day 4. Students need the problem (reruns, partial failure) before the tool.
- **Evidence:** Prefect, "Flows" and "Caching"; TS, "Prefect 3". Latest stable on PyPI is 3.9.4 (2026-11-19); the
    research tested 3.9.0.

### DATA_FOUNDATIONS-17 · You may not need an orchestrator

- **Priority:** should
- **Gist:** A script with a schedule (Cloud Run job + Cloud Scheduler) is a valid baseline. Say what it lacks, and
    name Airflow, Dagster and others as industry context only.
- **Why:** Matches the course's preference for picking tools by need (the 4Cs, Introduction to MLOps p. 35) and
    counters pipeline jungles.
- **Evidence:** TS, "Others"; Sculley et al. §4 (p. 5, "Pipeline Jungles"); Google Cloud, "Execute jobs on a
    schedule".

### DATA_FOUNDATIONS-18 · Idempotent reruns

- **Priority:** should
- **Gist:** Running twice gives the same result as running once. That is why retries, backfills and the week-3 rerun
    are safe.
- **Why:** The least intuitive property for students, and one question ("what happens if it dies halfway?") makes it
    clear.
- **Evidence:** Prefect, "Caching" (idempotent on retry); PV §1. Note that the default cache policy includes the flow
    run ID, so a fresh flow run does not reuse results unless configured.

### DATA_FOUNDATIONS-19 · Labels are data, and they are often wrong

- **Priority:** must
- **Gist:** Ten common benchmark test sets have an estimated average of at least 3.3% label errors, ImageNet
    validation at least 6%. Correcting them can change which model wins.
- **Why:** Short, surprising and sourced. Labels are treated as truth everywhere else in the course.
- **Evidence:** Northcutt et al. 2021 (abstract); flipped labels are also a recurring bug in Shankar et al. (p. 11).

### DATA_FOUNDATIONS-20 · Finding label errors with the model itself

- **Priority:** could
- **Gist:** Out-of-sample probabilities rank suspicious labels and a person decides. Flagged is not wrong: 51% of
    flagged candidates were confirmed by human review.
- **Why:** The conceptual hook for the cleanlab sub-exercise, wherever it lands.
- **Evidence:** Northcutt et al. 2021; LQ smoke test (80 of 90 injected flips caught, 309 flagged); [Find a maintained
    label-quality tool to replace cleanlab](https://github.com/SkafteNicki/dtu_mlops/issues/652).

### DATA_FOUNDATIONS-21 · Where labels come from

- **Priority:** should
- **Gist:** Hand labels, natural labels, weak supervision, automatic batch labels: each has a cost and a
    characteristic error. Labels are another append-only raw input (`raw/labels/<batch>/`).
- **Why:** Label Studio appears in the loop. The vial exercise (debugging p. 10) is about the risk of batch labels.
- **Evidence:** Ratner et al. 2017 (weak supervision); Shankar et al. §4.5.1 (p. 7, teams labelling live data); TS,
    "Label Studio" (why to export snapshots).

### DATA_FOUNDATIONS-22 · Same code in training and serving

- **Priority:** should
- **Gist:** The most common cause of training/serving skew is a different code path for the two. Put preprocessing in
    one place and reuse it when deploying.
- **Why:** A decision made today that bites in week 2, when RPS photos arrive through an API.
- **Evidence:** Breck et al. 2019, "Training-Serving Skew"; Google Cloud, "Experimental-operational symmetry".

### DATA_FOUNDATIONS-23 · How this feeds week 3

- **Priority:** must
- **Gist:** Close on the promise: new photos and labels are new raw batches, the same pipeline reruns, and lineage
    shows the new model's data. The second pass is no harder than the first.
- **Why:** Payoff and reason for every constraint above. Keep it a promise: the loop can't close on Day 4.
- **Evidence:** [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647), Notes; [Design the week-3 loop
    closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (open); Google Cloud, "ML pipeline triggers" (new
    data).

### DATA_FOUNDATIONS-24 · Privacy and deletion versus "keep raw forever"

- **Priority:** should
- **Gist:** Append-only collides with the right to erasure. Bucket versioning keeps noncurrent copies and soft delete
    holds objects for 7 days by default, so deletion must reach those too. Strip metadata from phone photos.
- **Why:** Students collect photos of their own hands, and this is the obvious objection to topic 06.
- **Evidence:** GDPR Art. 17 (via gdpr-info.eu); GCS object versioning and soft delete docs; LD (hands not faces,
    strip EXIF). The exact duty for students' photos is `unverified`; a question for the author.

### DATA_FOUNDATIONS-25 · Document and govern the dataset

- **Priority:** could
- **Gist:** A datasheet records motivation, composition, collection and use. EU law asks high-risk systems for data
    governance, naming annotation, labelling and cleaning, and for data as free of errors as possible.
- **Why:** Gives the practice a "why" outside the course. High-risk duties start 2 December 2027, so present it as
    coming.
- **Evidence:** Gebru et al. 2021 (abstract); AI Act Art. 10 (artificialintelligenceact.eu); European Commission, AI
    Act page (2 December 2027). The amending regulation's text was not read. REPRODUCIBILITY_AND_SOFTWARE-20 and
    INTRODUCTION_TO_MLOPS-19 also touch regulation; keep one home.

### Also noticed (no id, not decided)

- Table formats (Delta, Iceberg) as one "what you'd reach for at scale" line: their time travel is bounded by
    retention and snapshot expiry. Mention only, per the decision (PV §5).
- Medallion names as a translation for industry vocabulary; naming is undecided.
- ML Test Score (28 tests) as further reading; CONTINUOUS_INTEGRATION-14 already uses it.
- Cost: every noncurrent version is billed and there is no default limit, so lifecycle rules matter; better in Cloud.
- Feature stores, streaming, DuckLake: out, no module support.
- Data for LLMs and agents: no primary source collected, so no item; raise with the author if the deck should have an
    agent angle (agentic content is allowed in lectures, see [Update lecture slides for the January 2027
    course](https://github.com/SkafteNicki/dtu_mlops/issues/681)).

## Candidate examples

| # | Example | Supports | Evidence and caveats |
| --- | --- | --- | --- |
| E1 | Unity, 2022: ingested bad data from a large customer; about $110 million impact estimated for 2022 | 01, 03, 11 | Company prepared remarks, read. The estimate covers **two** issues (reduced Pinpointer accuracy and the bad data), so never state it as the cost of bad data alone |
| E2 | Public Health England, 2020: an XLS template capped at about 65,000 rows (about 1,400 cases each); further cases were dropped | 03, 11 | BBC explainer, read. The total number of missed cases was not verified here |
| E3 | COVID-19 imaging models: 62 studies reviewed, none of potential clinical use; duplicates inside "Frankenstein datasets" | 14 | Roberts et al. 2021, read |
| E4 | Chest X-ray models: internal test AUC 0.931, external site 0.815; prevalence differences between sites | 14, 19 | Zech et al. 2018, read in part. More about generalization; pick E3 or E4 |
| E5 | Vial inspection: defects introduced by hand in batches, so labels follow the batch | 21 | Debugging p. 10. Same case as Introduction to MLOps p. 38–42; permission and source `unverified` there |
| E6 | RPS: CGI hands on plain white backgrounds, made to test whether CGI-trained models classify real images | 04, 23 | Author's dataset page, read; 2,520 train and 372 test images (TFDS). Real-photo behaviour unknown |
| E7 | cleanlab smoke test: 90 flipped labels, 309 flagged, 80 found | 20 | LQ; rerun before quoting |

## Candidate discussion exercises

Two minutes in pairs, question then solution, matching the other decks. Pick two or three.

- **X1 (carry-over):** "What are the potential problem with automatic labeling of batches?" (debugging p. 10). The
    answer isn't in the deck; points to raise: the batch predicts the label, and every item gets its batch's label. →
    21
- **X2 (carry-over):** "Why is my model 'perfect'?" (debugging p. 22), with the bug list replaced by leakage and data
    checks so it doesn't refer to dropped slides. → 14
- **X3:** Given a new batch of RPS photos, write five checks; which ones stop the run and which only warn? → 11, 12,
    15
- **X4:** The pipeline dies halfway and you rerun it. What must be true for that to be safe? → 06, 10, 18
- **X5:** Someone asks for their photo to be deleted. Where are all the copies? → 24
- **X6:** Your teammate "fixed" three images in `raw/`. What breaks? → 06, 07, 09
- **X7:** One-minute labelling race on ambiguous hand photos, then compare answers. Needs real photos. → 19, 21

X3 is close to Continuous Integration's data contract exercise (pp. 19–20): keep one as "which checks", the other as
"where do they run".

## Candidate figures

Types and routes from [How should images be recreated, and what may we reuse from Gamma
exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

| # | Shows | Type | Route | Topics |
| --- | --- | --- | --- | --- |
| F1 | Course loop with today's part highlighted | concept diagram | Mermaid/PlantUML/Excalidraw; same drawing as Introduction to MLOps and Monitoring | 04, 23 |
| F2 | raw → ingest → validate → process → log, with "fail, nothing logged" | concept diagram | Mermaid | 05, 11 |
| F3 | data = f(raw, code, params), then what is stored versus recomputed | concept diagram | Excalidraw | 07, 10 |
| F4 | Lineage graph: raw, processed, run, model | concept diagram, or tool screenshot | Mermaid; fresh W&B screenshot once the module runs | 09 |
| F5 | Bar chart of estimated label errors per test set | chart/paper figure | re-plot + citation (Northcutt et al.); per-dataset numbers not collected here | 19 |
| F6 | Hard → soft → drift spectrum | concept diagram | redraw + citation (Shankar et al.) | 03, 15 |
| F7 | Plain `data.py`, then the same with Prefect decorators | code/config screenshot | code block (from exercise files where possible; do not exist yet) | 16 |
| F8 | pandera schema over a manifest | code/config screenshot | code block (from exercise files where possible; do not exist yet) | 12 |
| F9 | CGI hand next to a real hand photo | photo | RPS image (CC BY 2.0, credit) + licensed stand-in or the author's own photo | 04, 23 |
| F10 | Leakage kinds as four small split diagrams | concept diagram | Excalidraw | 14 |
| F11 | Prefect, pandera, Label Studio and W&B logos | tool logo | Iconify logo (availability not checked) | 16, 21 |
| F12 | Label Studio on RPS | tool screenshot | fresh screenshot | 21 |
| F13 | xkcd 2054 "Data Pipeline" ("a giant house of cards built from random scripts") | meme/joke | xkcd with credit link (CC BY-NC 2.5) | 16, 17 |
| F14 | xkcd 1838 "Machine Learning" (pile of linear algebra, "stir the pile") | meme/joke | xkcd with credit link | 01, 13 |

## Overlap with other decks

| Deck | Already covers | Suggested split |
| --- | --- | --- |
| `introduction_to_mlops` | Devil in the data (p. 21), the loop (p. 6), Sculley boxes (p. 32–36), vial case; INTRODUCTION_TO_MLOPS-02, -07, -17 | Recall the loop, don't redraw a new one. It introduces, this deck goes deep. Use other Sculley material (§3) than the boxes |
| `reproducibility_and_software` | Configs, dependencies, containers; proposed seeds and lockfiles (-11, -15), logging and artifacts (-12); regulation (-20) | Assume W&B runs and artifacts; show only lineage over data here |
| `cloud` | Accepted: bucket with `raw/` + `processed/`, "versioned", and Cloud Run jobs (CLOUD-14, -15) | Concept here, cloud services there. Reuse F2's names on both |
| `continuous_integration` | Data contract exercise (pp. 19–20), data tests (-14, -15), pipeline-change trigger (-12, -13), pandera as answer (-22) | Checks inside the pipeline here, checks on a pull request there |
| `monitoring` | Drift, natural labels and label lag (p. 11–14), hard/soft/drift split (-14), Goldilocks (p. 28), loop walk (-11), RPS drift (-12) | Label quality and origin here, label lag there. Hard/soft/drift: introduce here, reuse there |
| `project` | Canvas, "could you collect more of this data?" (PROJECT-22), fast-track box (PROJECT-11), checklist waits | Point at the next-day project start; no checklist promises |
| `scaling_applications` | Data loading and throughput, `s9_scalable_applications/data_loading.md` (brief not read for this) | Not repeated here |

The numbered findings above are proposals until each deck's own review; only CLOUD-14 and CLOUD-15 are accepted.

## Carry-over from the debugging deck

From `slides/day4_debugging_ML_code.pdf` (23 pages). The PDF stays linked from the debugging module; nothing here
removes it. Whether data checks and leakage move is the author's decision.

| Page | Content | Suggestion |
| --- | --- | --- |
| 8 | "Silent Failures" framing; xkcd 1700 | Reuse the term for 03; the comic optional, with credit |
| 10 | "2. Check your data": statistics, label distribution, samples; vial batch-label question | Carry over to 13, 21 and X1. Change one line: "If you are working on datasets you know, you may skip these steps" conflicts with 19, since known benchmarks have label errors |
| 11 | "3. Look out for data leakage": normalization, time series, duplicates, group | Carry over to 14. Re-word "includes batch normalization": a global statistic computed before the split is the trap, and the batch-norm claim is `unverified`. The definition omits serving-time leakage |
| 13 | "5. Make everything deterministic"; xkcd 221 | Only the determinism idea, for 10 |
| 22 | "Why is my model 'perfect'?" | Carry over as X2; "from our list (Data, Math, or Overfitting)" must change |
| 2–7, 9, 12, 14–21, 23 | Debuggers, stack traces, math, loss, modes, memes; p. 23 closing meme | Leave. Closing memes are dropped by rule |

Possible voice markers to bring: "Your starting point should always be the data in ML!" (p. 10), "Silent Failures" (p.
8), the 💡 and ✅ lists, and the 🤔 pair exercise (p. 22).

## Questions for the author

1. Which 10–15 of the 25 topics fit 30–45 minutes? All 14 `must` items plus two exercises is already about a full
    slot.
2. Which hook opens: the data-cascade quote (01), an incident (E1 or E2), or the vials (E5)?
3. Do data checks and leakage carry over, and at full size or as one slide each?
4. May the vial images and story be shown publicly? The same question is open for Introduction to MLOps.
5. How firm is "version the pipeline, not the data": a slogan with a counterpoint slide (08), or a design choice among
    several?
6. Does labelling get a lecture section on Day 4, or is it only named here and taught in week 3?
7. Live demo (run the flow, break a batch, see it stop) or slides only?
8. Privacy (24): is there a course rule for student photos, and who answers deletion requests?
9. Should the deck carry an agent angle? No material has been collected for one.
10. Are memes wanted beyond one xkcd (F13)?

## Dependencies on open tickets

| Ticket | What it decides | Waiting items |
| --- | --- | --- |
| [Outline the Day 4 data-foundations session](https://github.com/SkafteNicki/dtu_mlops/issues/654) | Modules and objectives; local or bucket references on Day 4; layer names; table formats; 10k object limit; Prefect version | 05, 06, 09, 12, 16, F2, F7, F8 |
| [Design the week-3 loop closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) | Where and how the loop closes, and the new-data trigger | 04, 21, 23, F1, F12 |
| [Check the Rock-Paper-Scissors model on real phone photos](https://github.com/SkafteNicki/dtu_mlops/issues/659) | Whether RPS stays; fallback is corrupted MNIST plus drawn digits | E6, F9, F12, X3, X7, 12, 24 |
| [Restructure the MLOps course for January 2027 to close the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/647) | "Not yet specified": place of the cleanlab sub-exercise, module numbers, checklist | 20, closing links |
| [What does the data foundations lecture contain?](https://github.com/SkafteNicki/dtu_mlops/issues/749) | The outline itself | all |

Also: on 11 December 2026 the Day 4 modules must be on `dev`, or 7 January uses the old PDF ([Which decks does the
January 2027 course need?](https://github.com/SkafteNicki/dtu_mlops/issues/723)). No Day 4 exercise file or module
exists on `main` yet, so every code figure waits.

## Unverified, and to re-check

- Beauchemin's essay (403), the Delta and Iceberg retention defaults beyond PV's reading, and the date of the lakeFS
    and DVC deal.
- Prefect without a managed server: TS observed it on 3.9.0, but the settings page disagreed; untested on 3.9.4 and in
    a Cloud Run job.
- W&B docs now redirect to docs.coreweave.com. Whether `file://` references restore overwritten files was not tested;
    a local directory has no object versions, so assume not until tested.
- Accuracy on real RPS photos; GDPR duty for student photos; AI Act amending text (EUR-Lex did not load).
- Papers marked "abstract" were read in abstract only: Sambasivan, Kapoor & Narayanan, Northcutt, Gebru, Schelter and
    Ratner et al. Breck et al. was read in part.
- pandera released 0.36.0 (2026-11-10) since the research used 0.34.1; cleanlab (2.9.0) and Label Studio (1.23.2)
    unchanged.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Sambasivan et al. 2021, *"Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI* | <https://research.google/pubs/everyone-wants-to-do-the-model-work-not-the-data-work-data-cascades-in-high-stakes-ai/> | 01 |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | 02, 08, 17 |
| Shankar et al. 2022, *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | 03, 04, 11, 13, 14, 15, 19, 21 |
| Breck et al. 2019, *Data Validation for Machine Learning* | <https://proceedings.mlsys.org/paper_files/paper/2019/hash/928f1160e52192e3e0017fb63ab65391-Abstract.html> | 03, 11, 22 |
| Schelter et al. 2018, *Automating Large-Scale Data Quality Verification* | <https://www.vldb.org/pvldb/vol11/p1781-schelter.pdf> | 11 |
| Google Cloud, *MLOps: Continuous delivery and automation pipelines in machine learning* | <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning> | 05, 09, 15, 22, 23 |
| Databricks, *Medallion architecture* | <https://docs.databricks.com/aws/en/lakehouse/medallion> | 06 |
| Beauchemin, *Functional Data Engineering* (not read) | <https://maximebeauchemin.medium.com/functional-data-engineering-a-modern-paradigm-for-batch-data-processing-2327ec32c42a> | 07 |
| lakeFS, *lakeFS Acquires DVC* | <https://lakefs.io/media-mentions/lakefs-acquires-dvc-uniting-data-version-control-pioneers/> | 08 |
| W&B, *Track external files* | <https://docs.coreweave.com/products/wandb/artifacts/track-external-files> | 09 |
| W&B, *Explore artifact lineage graphs* | <https://docs.coreweave.com/products/wandb/artifacts/explore-and-traverse-an-artifact-graph> | 09 |
| pandera documentation | <https://pandera.readthedocs.io/en/stable/> | 11, 12 |
| Kapoor & Narayanan 2022, *Leakage and the Reproducibility Crisis in ML-based Science* | <https://arxiv.org/abs/2207.07048> | 14 |
| Roberts et al. 2021, *Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19* | <https://www.nature.com/articles/s42256-021-00307-0> | 14, E3 |
| Zech et al. 2018, *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs* | <https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683> | E4 |
| Prefect, *Flows* and *Caching* | <https://www.prefect.io/docs/v3/concepts/flows>, <https://www.prefect.io/docs/v3/concepts/caching> | 16, 18 |
| Google Cloud, *Execute jobs on a schedule* | <https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule> | 17 |
| Northcutt et al. 2021, *Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks* | <https://arxiv.org/abs/2103.14749> | 19, 20 |
| Ratner et al. 2017, *Snorkel: Rapid Training Data Creation with Weak Supervision* | <https://arxiv.org/abs/1711.10160> | 21 |
| Label Studio, *Cloud storage* | <https://labelstud.io/guide/storage> | 21 |
| GDPR Art. 17 | <https://gdpr-info.eu/art-17-gdpr/> | 24 |
| Google Cloud, *Object Versioning*; *Soft delete* | <https://docs.cloud.google.com/storage/docs/object-versioning>, <https://docs.cloud.google.com/storage/docs/soft-delete> | 24 |
| Gebru et al. 2021, *Datasheets for Datasets* | <https://arxiv.org/abs/1803.09010> | 25 |
| AI Act, Art. 10; European Commission, *AI Act* | <https://artificialintelligenceact.eu/article/10/>, <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai> | 25 |
| Unity, *Q1 2022 Prepared Remarks* | <https://s205.q4cdn.com/200788659/files/doc_events/2022/05/Q1-2022-Prepared-Remarks.pdf> | E1 |
| BBC 2020, *Excel: Why using Microsoft's tool caused Covid-19 results to be lost* | <https://www.bbc.com/news/technology-54423988> | E2 |
| Moroney, *Rock Paper Scissors Dataset* (CC BY 2.0); TFDS catalog | <https://laurencemoroney.com/datasets.html>, <https://www.tensorflow.org/datasets/catalog/rock_paper_scissors> | E6, F9 |
| xkcd 2054 *Data Pipeline*; 1838 *Machine Learning*; licence | <https://xkcd.com/2054/>, <https://xkcd.com/1838/>, <https://xkcd.com/license.html> | F13, F14 |
