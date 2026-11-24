# Review brief: monitoring (`slides/day11_monitoring.pdf`)

- **Modules compared:** `s8_monitoring/` (`README.md`, `data_drifting.md`, `monitoring.md`, `exercise_files/`) @
    `main` d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b. Project checklist checked in `pages/projects.md`, date in
    `pages/timeplan.md`.
- **Other inputs:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`,
    `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, web sources (accessed 2026-11-24).
- **Pages:** 32 PDF pages → 25 slides after merging the click build (pages 21=22=23=24=25=26=27) and dropping the
    duplicate (12=13). No Gamma "Error uploading image" placeholders. Text extracted with pdftotext and PyMuPDF
    (links); every page render checked by eye. Glyph repairs: "4" in the text layer is an em dash on pp. 5, 6, 19, 28;
    "³" is → on p. 16, "9" is an apostrophe there; "D" bullets on p. 20 are 💡; the p. 7 header glyph is 🇩🇰.

## Deck outline (narrative order to keep)

1. Title: page 1
2. Why monitor (value, endgame): pages 2–3
3. What can fail, and software failures: pages 4–5
4. ML failures, drift types, detecting and addressing drift, other errors: pages 6–10
5. Getting labels, label-lag exercise, cold start: pages 11–15
6. Business failures and failure-mode exercise: pages 16–18
7. From failures to monitoring, three pillars, the monitored system build: pages 19–27
8. Alerts, retraining triggers, human in the loop: pages 28–30
9. Tools: page 31
10. Meme of the day: page 32 (dropped, MONITORING-25)

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "We Are in the Endgame Now" (p. 3)
    - "This can fail" / "But these can/will also fail" (p. 4)
    - "'Software is never done (only abandoned)', John Saddington" (p. 5); attribution `unverified`, see Also noticed
    - "Drift drift and more drift" (p. 7)
    - "data distributions does not overlap enough anymore" (p. 8)
    - "To retrain you need labels - how to get them?" (p. 11)
    - "An example (what do we think OpenAI can get out of me):" (p. 11). MONITORING-23 changes the link.
    - "Any retrained model must traverse the same rigorous gated promotion pipeline as the original model." (p. 29)
    - "Remember the Human in the Loop" (p. 30)
- Voice markers:
    - The lemon that rots in production, with "You?" (p. 3). MONITORING-24 redraws it.
    - The burning packet ship next to software failures (p. 5)
    - The software / ML / business Venn diagram as the frame for the deck (p. 4), reused in pp. 5, 6, 16 and 19
    - 🇩🇰 healthcare examples: Copenhagen, rural Jutland, Rigshospitalet (p. 7). MONITORING-02 changes one row.
    - Question-then-solution exercises: loan and music app (pp. 13–14, 17–18). MONITORING-04 renumbers them.
    - The monitored system built up box by box, ending with the unhappy stick figure who gets the alerts (pp. 21–27).
        MONITORING-18 adds a step.
    - The "Goldilocks problem" (p. 28; also in the module)

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

- [Design the week-3 loop closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (still **open**): drift,
    collect production photos, label, rerun the Day 4 pipeline, retrain, redeploy; likely hosted in this session. Not
    a decision yet; re-check before grilling. → see MONITORING-11, MONITORING-22
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): new data is a new
    batch in append-only `raw/` plus a rerun of the same pipeline, with W&B lineage. → see MONITORING-11
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): Prefect 3, Label
    Studio Community and pandera; the pipeline is a Cloud Run job on GCP. → see MONITORING-06, MONITORING-07,
    MONITORING-11
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651):
    Rock-Paper-Scissors gives CGI-to-photo drift, pending a real-photo check. → see MONITORING-09, MONITORING-12
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    the new-data trigger is left out of CML and handed to the loop closure. → see MONITORING-11, MONITORING-13
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): W&B logging
    moves to Day 3 and loguru to Day 2, so logs and experiment tracking are known before p. 20 and p. 31. No change
    needed.
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621):
    Always-on agents sits in this session. Lectures may carry agentic content but do not otherwise reference A-modules
    ([Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681)). → see
    MONITORING-19

## Findings

<!-- max ~25; ids are stable -->

### MONITORING-01 · Unrendered LaTeX with missing parentheses

- **Category:** wrong
- **Priority:** must
- **Pages:** 7
- **Finding:** The definitions print as source: `$P(Y\|X$`, `$P(X\|Y$`, `$P_{train}(X)$`. Two of them also lack a
    closing parenthesis, so the formal definition of all three shifts is broken.
- **Proposal:** Typeset as math: covariate shift changes P(X) and keeps P(Y|X), label shift changes P(Y) and keeps
    P(X|Y), concept shift changes P(Y|X).
- **Evidence:** Page render of p. 7; the shift definitions agree with the paper figure on p. 6.

### MONITORING-02 · Concept-shift example rests on a wrong health claim

- **Category:** wrong
- **Priority:** must
- **Pages:** 7
- **Finding:** "High fat intake" is called "a clear risk factor for diabetes" that LCHF diets made obsolete. A
    dose-response meta-analysis of prospective studies found "no association between total fat intake and the
    incidence of T2D". A diet becoming popular also moves P(X) by itself; the slide does not show why P(Y|X) changes.
- **Proposal:** Keep the Danish healthcare framing (keep list) but swap in an example where the labelling rule itself
    changes, e.g. a changed diagnostic criterion; check that example against a source first.
- **Evidence:** Neuenschwander et al. 2020, <https://pmc.ncbi.nlm.nih.gov/articles/PMC7710077/> (accessed 2026-11-24).

### MONITORING-03 · Drift is presented as degradation

- **Category:** wrong
- **Priority:** should
- **Pages:** 14
- **Finding:** P. 14 says input shifts "indicate likely performance degradation". Evidently notes that prediction
    drift "does not always imply model deterioration" and that feature and prediction drift can occur "without a decay
    in the model quality". The interviewed engineers' most common pain point was alerts firing when ML performance was
    adequate.
- **Proposal:** Call drift a reason to investigate, not evidence of failure, and make explicit that only labels or a
    trusted proxy confirm degradation. Google's CT guide is closer to the slide (changes "suggest that your model has
    gone stale"), so this is a nuance to settle in grilling.
- **Evidence:** Evidently, "Data drift" guide, <https://www.evidentlyai.com/ml-in-production/data-drift> (accessed
    2026-11-24), "prediction drift" section; Google Cloud, "MLOps: Continuous delivery and automation pipelines in
    machine learning", "ML pipeline triggers",
    <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
    (accessed 2026-11-24); `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 9 (challenges
    with data errors, false-positive alerts).

### MONITORING-04 · Exercise numbering and a duplicated exercise page

- **Category:** wrong
- **Priority:** must
- **Pages:** 12, 13, 14, 17, 18
- **Finding:** The natural-label question appears twice, as "Exercise 2" (p. 12) and "Exercise 1" (p. 13); the
    failure-mode exercise that follows (pp. 17–18) is also "Exercise 1".
- **Proposal:** Drop p. 12 and number in narrative order: natural-label lag is 1, failure modes is 2.
- **Evidence:** Page renders of pp. 12, 13, 14, 17, 18.

### MONITORING-05 · Tool slide lists discontinued products

- **Category:** outdated
- **Priority:** must
- **Pages:** 31
- **Finding:** WhyLabs, Inc. "is discontinuing operations". Neptune announced it was being acquired by OpenAI and
    would end its hosted service on 2026-03-05. Arize is now part of Dynatrace (completed 2026-10-01).
- **Proposal:** Remove WhyLabs and Neptune and mark Arize as part of Dynatrace. Cut the list to tools the course uses
    plus one or two alternatives per box.
- **Evidence:** WhyLabs, <https://whylabs.ai>; Neptune transition hub, <https://docs.neptune.ai/transition_hub>;
    Dynatrace press release, <https://www.dynatrace.com/news/press-release/dynatrace-completes-acquisition-of-arize/>
    (all accessed 2026-11-24).

### MONITORING-06 · "Vertex AI" and pipeline tools on the tool slide

- **Category:** outdated
- **Priority:** should
- **Pages:** 31
- **Finding:** Google's docs now sit under "Gemini Enterprise Agent Platform". The Cloud deck decided on "Vertex AI"
    with a small "now part of the Gemini platform" note. The restructure also picked Prefect 3 as the course
    orchestrator, and GCP Monitoring (which the module uses) is missing.
- **Proposal:** Apply the Cloud wording, show Prefect as the course's choice and add Cloud Monitoring under
    infrastructure.
- **Evidence:** [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710),
    CLOUD-06; [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649);
    `s8_monitoring/monitoring.md`, "Cloud monitoring".

### MONITORING-07 · Detection sources on p. 8 are mixed up

- **Category:** wrong
- **Priority:** should
- **Pages:** 8
- **Finding:** The three links read as sources for drift detection, but Great Expectations describes itself as a
    framework for validating data against expectations. The table is from alibi-detect's README, but no marker ties it
    to link [3]; alibi-detect's current licence is Business Source License 1.1, which states it is "not an Open Source
    license".
- **Proposal:** Attribute the table to alibi-detect, separate validation from drift detection (pandera is the
    restructure's validation tool) and mention the licence.
- **Evidence:** GX, <https://docs.greatexpectations.io/docs/core/introduction/gx_overview>; alibi-detect README and
    LICENSE, <https://github.com/SeldonIO/alibi-detect> (accessed 2026-11-24); [Choose the data-engineering tool
    stack](https://github.com/SkafteNicki/dtu_mlops/issues/649).

### MONITORING-08 · How a drift test actually decides

- **Category:** gap
- **Priority:** should
- **Pages:** 8
- **Finding:** The deck says only that distributions do "not overlap enough anymore". The module has students run and
    tune tests; Evidently's defaults use hypothesis tests (KS, chi-squared; p < 0.05) up to 1000 reference rows and
    distances (Wasserstein, Jensen–Shannon; 0.1) above that, and flag dataset drift by the share of drifted columns.
- **Proposal:** Add one slide: reference vs current window, per-column test or distance, threshold, dataset verdict.
    Replace the 13-row detector table as the main content.
- **Evidence:** `s8_monitoring/data_drifting.md`, exercises 3–4; Evidently, "Data drift" explainer,
    <https://docs.evidentlyai.com/metrics/explainer_drift> (accessed 2026-11-24).

### MONITORING-09 · Images and text need features first

- **Category:** gap
- **Priority:** should
- **Pages:** 8
- **Finding:** The module spends exercise 6 on turning images and text into features (brightness, descriptors, CLIP
    embeddings) before drift detection. The deck shows only a one-dimensional feature, although the course's loop data
    will be photos.
- **Proposal:** Add a small flow: raw input, features or embedding, drift test.
- **Evidence:** `s8_monitoring/data_drifting.md`, exercise 6; Rabanser et al. 2019, <https://arxiv.org/abs/1810.11953>
    (two-sample tests on dimensionality-reduced representations) (accessed 2026-11-24); [Find the best dataset for the
    lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651).

### MONITORING-10 · Marginal tests miss joint drift

- **Category:** gap
- **Priority:** should
- **Pages:** 8
- **Finding:** The module closes with the limit that its tests consider only marginals, and points to MMD. The deck
    lists MMD in a table without saying why it matters.
- **Proposal:** Redraw the module's marginals figure as one visual next to the detection slide.
- **Evidence:** `s8_monitoring/data_drifting.md`, closing points (figure `data_drift_marginals.png`).

### MONITORING-11 · Close the lifecycle loop on slides

- **Category:** addition
- **Priority:** should
- **Pages:** 2, 9, 19, 29
- **Finding:** The deck names the parts (p. 2 loop, p. 9 retraining, p. 19 "Close the loop", p. 29 triggers) but never
    walks the loop. The restructure's destination is deploy, collect, label, rerun the pipeline, retrain, redeploy,
    likely hosted here.
- **Proposal:** Reuse the p. 2 loop as a recurring map and add a worked pass near the end: drift signal, new `raw/`
    batch, Label Studio, rerun the Day 4 pipeline, gate, redeploy. Hold details until the design issue closes.
- **Evidence:** [Design the week-3 loop closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (open);
    [Restructure the MLOps course for January 2027 to close the lifecycle
    loop](https://github.com/SkafteNicki/dtu_mlops/issues/647); [How do modern ML pipelines version
    data?](https://github.com/SkafteNicki/dtu_mlops/issues/648); [Choose the data-engineering tool
    stack](https://github.com/SkafteNicki/dtu_mlops/issues/649).

### MONITORING-12 · Use the course's own drift story

- **Category:** addition
- **Priority:** could
- **Pages:** 8, 12–14
- **Finding:** No example in the deck ties to the exercises. The planned Rock-Paper-Scissors data gives real drift,
    from CGI hands to phone photos; labels take seconds, unlike the 5-year loan.
- **Proposal:** Add a CGI-versus-photo pair as the drift example and contrast its label lag with the loan. Depends on
    the pending real-photo check.
- **Evidence:** [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651);
    [Check the Rock-Paper-Scissors model on real phone photos](https://github.com/SkafteNicki/dtu_mlops/issues/659).

### MONITORING-13 · Retraining threshold has no source

- **Category:** wrong
- **Priority:** should
- **Pages:** 29
- **Finding:** "Typically 5-10% of original training set" carries no citation. None of the participants in the
    interview study reported a scientific procedure for choosing the cadence, which ranged from hourly to every few
    months. Google lists on demand, schedule, new data, performance degradation and significant distribution changes,
    with no percentage.
- **Proposal:** Drop the percentage, contrast daily scheduled retraining with triggered retraining and cite both
    sources. Keep the "Critical Requirement" box.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 7, §4.5.1; Google Cloud,
    "MLOps: Continuous delivery and automation pipelines in machine learning", "ML pipeline triggers",
    <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
    (accessed 2026-11-24). No source for 5–10% was found (origin `unverified`).

### MONITORING-14 · Natural drift versus broken data

- **Category:** addition
- **Priority:** should
- **Pages:** 6, 10, 28
- **Finding:** The interview study splits hard, soft and drift errors, and reports that engineers worried little about
    slow natural drift because retraining handled it; unnatural drift (missing, corrupted data) degraded models
    sharply. The deck lists data quality as a bullet among "other" errors (p. 10) after several drift slides.
- **Proposal:** Give p. 10 the hard/soft/drift split, with data-quality checks first, and cite it. The order of the
    slides can stay.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 9 (spectrum of data errors)
    and p. 10 (natural and unnatural data drift).

### MONITORING-15 · "Prevent failures with monitoring" overclaims

- **Category:** wrong
- **Priority:** should
- **Pages:** 19
- **Finding:** Monitoring detects and shortens failures; the slide's own list contains alerts, rollback and
    improvement. Two participants using black-box data monitoring services "lamented that their alerts did not prevent
    failures".
- **Proposal:** Reword the question to detecting and responding; keep the three-column flow.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 7, §4.5.4.

### MONITORING-16 · Metric types are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 20, 24
- **Finding:** The module's first exercises are Counter, Gauge, Histogram and Summary and a `/metrics` endpoint. The
    deck has the endpoint as a box but no types and no code.
- **Proposal:** Add one slide with the four types, each with an ML-service example, and a short block imported from
    `sentiment_api_prometheus_simple.py`.
- **Evidence:** `s8_monitoring/monitoring.md`, "Attaching metrics to an application", exercises 2–3.

### MONITORING-17 · SLOs and the four golden signals

- **Category:** gap
- **Priority:** should
- **Pages:** 19, 28
- **Finding:** P. 19 says "business SLOs" without defining an SLO; the module asks students to create one in GCP.
    Nothing says which system metrics to begin with.
- **Proposal:** Define SLO in one line and add latency, traffic, errors, saturation as the starting set; alert on
    symptoms.
- **Evidence:** `s8_monitoring/monitoring.md`, "Cloud monitoring" exercise 2; Google SRE Book, "Monitoring Distributed
    Systems", <https://sre.google/sre-book/monitoring-distributed-systems/> (accessed 2026-11-24).

### MONITORING-18 · Map the monitored system to what students deploy

- **Category:** gap
- **Priority:** should
- **Pages:** 21–27
- **Finding:** The seven-step build is generic. In the exercises the data database is a bucket, the drift check is a
    second Cloud Run service, and metrics are scraped by a sidecar into Managed Service for Prometheus. "Send computed
    signal" is also a push, while the metrics box next to it pulls.
- **Proposal:** Keep the build (keep list) and add a last step naming the GCP component on each box; expose drift
    results as scrapeable metrics or say they are written directly.
- **Evidence:** `s8_monitoring/data_drifting.md`, "Data drift in the Cloud"; `s8_monitoring/monitoring.md`, "Cloud
    monitoring"; Google Cloud, "Use the Prometheus sidecar for Cloud Run",
    <https://docs.cloud.google.com/stackdriver/docs/managed-prometheus/cloudrun-sidecar> (accessed 2026-11-24).

### MONITORING-19 · Agents reading the monitoring signals

- **Category:** addition
- **Priority:** could
- **Pages:** 30
- **Finding:** The session's agentic content is an always-on agent that reads monitoring and drift signals and opens
    an issue: it reports, it does not fix. The human-in-the-loop slide is the natural place.
- **Proposal:** Add one line or small diagram to p. 30 on agent triage with a human deciding. Do not name the module
    unless the author decides to.
- **Evidence:** [Propose the A-module topic menu, one-or-more per
    session](https://github.com/SkafteNicki/dtu_mlops/issues/621); [Update lecture slides for the January 2027
    course](https://github.com/SkafteNicki/dtu_mlops/issues/681).

### MONITORING-20 · LLM applications: what changes in monitoring

- **Category:** addition
- **Priority:** could
- **Pages:** 20
- **Finding:** The deck covers predictive ML only. OpenTelemetry now has semantic conventions for agent spans (create
    agent, invoke agent, execute tool), still marked Development, and Evidently's docs cover LLM output evaluation
    beside ML monitoring.
- **Proposal:** At most one slide or a traces bullet: traces record model and tool calls, and quality is scored by
    evaluations. No tool tour.
- **Evidence:** OpenTelemetry,
    <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md>;
    Evidently, <https://docs.evidentlyai.com/introduction> (accessed 2026-11-24).

### MONITORING-21 · Compliance trigger has no anchor

- **Category:** addition
- **Priority:** could
- **Pages:** 29, 16
- **Finding:** "Compliance-Driven Updates" is abstract. Article 72 of the EU AI Act requires providers of high-risk
    systems to "actively and systematically collect, document and analyse" performance data throughout the lifetime;
    the AI omnibus moved Annex III high-risk obligations to 2 December 2027.
- **Proposal:** Name the AI Act's post-market monitoring in the box, with the date. Re-check the date before the
    lecture.
- **Evidence:** AI Act Article 72, <https://artificialintelligenceact.eu/article/72/>; Orrick, "EU AI Act Update",
    <https://www.orrick.com/en/Insights/2026/07/EU-AI-Act-Update-Digital-Omnibus-Finalizes-8-Compliance-Changes>
    (accessed 2026-11-24). Secondary sources; the Official Journal text was not fetched, so any change to Article 72
    is `unverified`.

### MONITORING-22 · Deck topics the module does not cover

- **Category:** drift
- **Priority:** could
- **Pages:** 11–18, 29–30
- **Finding:** Label types, label lag, cold start, business failures, retraining triggers and human oversight have no
    module section; `s8_monitoring/` covers drift detection and telemetry only.
- **Proposal:** Keep on the slides: they are what the loop closure needs. Make sure the loop module picks up labels
    and triggers, so the slide and module agree.
- **Evidence:** `s8_monitoring/README.md`, learning objectives; `s8_monitoring/data_drifting.md` and
    `s8_monitoring/monitoring.md`, all headings; [Design the week-3 loop
    closure](https://github.com/SkafteNicki/dtu_mlops/issues/655).

### MONITORING-23 · ChatGPT example link is dead

- **Category:** outdated
- **Priority:** should
- **Pages:** 11
- **Finding:** The "what do we think OpenAI can get out of me" example links to a ChatGPT share that returned HTTP 404
    to an unauthenticated request.
- **Proposal:** Author's call: new share link, or show the feedback signals (thumbs, regenerate, follow-up) as a small
    mock-up. The line is on the keep list.
- **Evidence:** <https://chatgpt.com/share/68ff3a0d-f154-8000-a741-0c5c998b89ee> (accessed 2026-11-24); not checked in
    a logged-in browser.

### MONITORING-24 · Lemon figure: redraw the joke

- **Category:** image
- **Priority:** should
- **Pages:** 3
- **Finding:** The rotting-lemon chart carries the "Endgame" slide's joke ("You?"). Its origin is not credited, only
    the emoji source.
- **Proposal:** Author's call: redraw as an original three-stage lemon over a business-value axis, or swap. Present it
    as an illustration, not as data.
- **Evidence:** [How should images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705). Origin of the figure: `unverified`.

### MONITORING-25 · Closing meme goes

- **Category:** image
- **Priority:** must
- **Pages:** 32
- **Finding:** The deck ends on a "Meme of the day" built from a television cartoon still, against the cross-deck
    rule.
- **Proposal:** Drop p. 32 and move its exercise link to the end of p. 31 or the references slide.
- **Evidence:** [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710)
    (cross-deck rule 2).

### Also noticed (no id, not decided)

- p. 1: title without the day number and dated January 2027; the lecture is on 18/1/27 (`pages/timeplan.md`).
- p. 5: one web search found no source that credits the quote to John Saddington, and found the art version credited
    to others; attribution `unverified`.
- p. 5: the caption is half Danish ("af") and the work is a watercolor by Fitz Henry Lane, 1830, Cape Ann Museum
    (<https://fitzhenrylaneonline.org/catalog/entry.php?id=82>, accessed 2026-11-24).
- p. 7: diabetes prevalence "~8%" is unsourced; Diabetesforeningen gave 6.1% of the population in an earlier release
    (<https://via.ritzau.dk/pressemeddelelse/14373482/ny-rekord-aldrig-har-sa-mange-danskere-levet-med-diabetes>,
    search excerpt only). Whether P(X|Y) is really unchanged in a specialised clinic is `unverified`.
- pp. 6 and 10: "Training-Production Skew" and "Train-serving skew" are the same thing; pick one term.
- pp. 9, 10: p. 9 comes before the other ML errors on p. 10; swapping is possible but the default is to keep the
    order.
- pp. 2, 8, 21: grammar ("create", "does not overlap", "Let setup").
- p. 20: the logo row has no trace backend, and the Loki and Grafana wordmarks are barely legible on the dark theme.
- p. 30: Sculley et al. call relying on human response to alert pages "brittle for time-sensitive issues"
    (`literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, p. 7); a possible counterpoint to the
    human-in-the-loop slide.
- GCP alerting will be charged no sooner than 2027-09-01 (<https://cloud.google.com/products/observability/pricing>,
    accessed 2026-11-24), so the exercise is free in January 2027.
- p. 11: storing user inputs raises consent and retention questions that neither deck nor module mention (`unverified`
    against GDPR text).

## Module follow-ups (out of scope for the deck)

- `s8_monitoring/data_drifting.md`, intro: defines data drift as a change "that leads to model performance
    degradation", which conflicts with MONITORING-03, and calls retraining the only option.
- `s8_monitoring/data_drifting.md`, "❔ Exercises" intro: WhyLabs has shut down; whylogs' last push was 2025-01-10,
    NannyML's 2025-07-12, Deepchecks' last release 2024-12-15 (GitHub API, accessed 2026-11-24).
- `s8_monitoring/data_drifting.md`, exercise 6.3: describes CLIP as image captioning; the CLIP paper describes
    matching images to captions (<https://arxiv.org/abs/2103.00020>, abstract, accessed 2026-11-24).
- `s8_monitoring/data_drifting.md`, "Data drift in the Cloud": the intro promises an app that runs on a schedule, but
    exercise 3 deploys an on-demand, unauthenticated report endpoint.
- `s8_monitoring/monitoring.md`, "Alert systems": still says cloud function (now Cloud Run functions) and hard-codes a
    `cloudfunctions.net` URL in exercise 5.
- `s8_monitoring/monitoring.md`, intro: telemetry is not necessarily "wireless transmission" (`unverified`).
- `s8_monitoring/`: no 🧠 Knowledge check in either module; labels, retraining triggers and alert evaluation are only
    in the deck.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | ML lifecycle loop: select, collect, clean and label, develop, eval, deploy, monitor, user | concept diagram | Mermaid or Excalidraw | | MONITORING-11 |
| 3 | Rotting lemon "Model" over a business-value axis, emojis, "You?" | meme/joke | keep / regenerate / swap? (redraw) | voice | MONITORING-24 |
| 4 | Hand-drawn Venn diagram: ML, software, business | concept diagram | Excalidraw | | |
| 5 | Fitz Henry Lane, Burning of the Packet Ship "Boston" | photo | licensed stand-in (museum image with credit; licence unverified) or generated | | |
| 6 | Four-panel original, covariate, label, concept shift, with handwritten captions | chart/paper figure | re-plot/redraw + citation (source unverified) | | |
| 8 | alibi-detect README detector table | code/config screenshot | redraw as a short table + citation | | MONITORING-07, MONITORING-08 |
| 8 | Two overlapping density curves, "Feature 1" 2019 vs 2021 | chart/paper figure | re-plot (synthetic data) | | MONITORING-08 |
| 9 | Three AI-style illustrations | decorative | replaced by theme | | |
| 10 | Three Evidently illustrations: drift vs skew, quality, outliers | concept diagram | redraw + citation (Evidently) | | MONITORING-14 |
| 11, 14, 15, 18, 19, 28, 29 | Card and list icons | decorative | Iconify icons | | |
| 16 | KPI hexagons with a "PK Medier" logo | decorative | replaced by theme | | |
| 20 | Logos: Grafana Loki, OpenTelemetry, Grafana, Prometheus, Cloud Monitoring | tool logo | Iconify logo (availability unverified) | | |
| 21–27 | Monitored system, seven build steps | concept diagram | Mermaid or Excalidraw with v-click | voice (stick figure, p. 27; keep) | MONITORING-18 |
| 32 | Cartoon still meme, "ML Monitoring" takes the shot for "Model in production" | meme/joke | dropped | voice | MONITORING-25 |
| 1–32 | Dark Gamma background art, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo ·
meme/joke · decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated ·
replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Shankar, Garcia, Hellerstein, Parameswaran 2022, *Operationalizing Machine Learning: An Interview Study* | <https://arxiv.org/abs/2209.09125> | MONITORING-03, -13, -14, -15; p. 28 alerts (already cited) |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | p. 6 "Feedback Loops" (§4); p. 30 in Also noticed |
| Rabanser, Günnemann, Lipton 2019, *Failing Loudly* | <https://arxiv.org/abs/1810.11953> | MONITORING-09, p. 8 |
| Gretton et al. 2012, *A Kernel Two-Sample Test* (cited in the module; page not fetched) | <https://jmlr.org/papers/v13/gretton12a.html> | MONITORING-10 |
| Evidently, *Data drift* | <https://www.evidentlyai.com/ml-in-production/data-drift> | MONITORING-03, p. 10 figures |
| Evidently, *Data drift explainer* | <https://docs.evidentlyai.com/metrics/explainer_drift> | MONITORING-08 |
| Neuenschwander et al. 2020, *Intake of dietary fats and fatty acids and the incidence of type 2 diabetes* | <https://pmc.ncbi.nlm.nih.gov/articles/PMC7710077/> | MONITORING-02, p. 7 |
| Google Cloud, *MLOps: Continuous delivery and automation pipelines in machine learning* | <https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning> | MONITORING-13, p. 29 |
| Google SRE Book, *Monitoring Distributed Systems* | <https://sre.google/sre-book/monitoring-distributed-systems/> | MONITORING-17, p. 19 |
| Google Cloud, *Use the Prometheus sidecar for Cloud Run* | <https://docs.cloud.google.com/stackdriver/docs/managed-prometheus/cloudrun-sidecar> | MONITORING-18, pp. 21–27 |
| OpenTelemetry, *Semantic conventions for GenAI agent spans* | <https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md> | MONITORING-20 |
| EU AI Act, Article 72 | <https://artificialintelligenceact.eu/article/72/> | MONITORING-21, p. 29 |
