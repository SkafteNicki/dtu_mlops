# Review brief: deployment (`slides/day8_deployment.pdf`)

- **Modules compared:** `s7_deployment/` (`README.md`, `apis.md`, `cloud_deployment.md`, `testing_apis.md`,
    `ml_deployment.md`, `frontend.md`, `exercise_files/`) @ `main` d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b. Overlaps
    checked in `s9_scalable_applications/inference.md` and `pages/timeplan.md`.
- **Other inputs:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`,
    `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, web sources (accessed 2026-11-24).
- **Pages:** 32 PDF pages → 32 slides (31 once the closing meme goes). Pages 18 and 19 share a title but show different
    content, so there are no click builds to merge. There are no Gamma "Error uploading image" placeholders. Text was
    extracted with `pdftotext` and every page render was checked by eye. Glyph repairs: "&" is an ellipsis (pp. 2, 22),
    "4" is an em dash (pp. 4, 7), "3" is a dash (p. 8), "9" is an apostrophe and "<…=" are curly quotes (pp. 25–31).

## Deck outline (narrative order to keep)

1. Title and "remember this figure": pages 1–2
2. Why deployment matters, production requirements: pages 3–4
3. From model to service (registry, optimize, package, engine, API, container): pages 5–10
4. Where it runs (road to modern deployment, deployment types, cloud, "it depends"): pages 11–14
5. Rollout and architecture (strategies, monolith vs microservices, supporting components): pages 15–17
6. Real-life example by Petr Taborsky: pages 18–19
7. Group exercise: choosing the right infrastructure: page 20
8. Predictions vs decisions (human in the loop, conveying information, output forms, consumers): pages 21–26
9. Discussion exercise "What Was the Model Really Saying?": page 27
10. Accountability (decision trace, "before deploying, ask"): pages 28–30
11. Group exercise: automation vs human oversight: page 31
12. Meme of the day: page 32

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "We are now in the end part of the figure..." (p. 2)
    - "APIs are in charge of this conversion." (p. 9)
    - "Where should I run my stuff?" (p. 14)
    - "We often think prediction == decision in automated systems." (p. 21)
    - "Conveying information is hard...." (p. 22)
    - "Same model. Different decisions." (p. 23). DEPLOYMENT-22 changes the cards under it, not the title.
    - "What should we show to the decision maker? The number, the uncertainty, or the narrative?" (p. 23)
    - "A decision you can’t explain today becomes a liability tomorrow." (p. 28; no source found, see "Also noticed")
- Voice markers:
    - "Remember This Figure" as the recurring opening (p. 2)
    - The "Next Step: ..." chain ending in "Final step" (pp. 7–10, 15)
    - "Side step" for the history detour (p. 11)
    - The real-life example credited by name (pp. 18–19)
    - The two group scenarios and the who-bears-responsibility debate (pp. 20, 27, 31). DEPLOYMENT-21 repairs the links
        only.
    - "Taking a step back" as the turn from engineering to decisions (p. 21)
    - The misread IQ score as the opener on communication (p. 22). DEPLOYMENT-25 changes the image.
    - The "it works on my machine" Docker joke (p. 10). DEPLOYMENT-25 asks for a decision.

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

Links are to [Restructure the MLOps course for January 2027 to close the lifecycle
loop](https://github.com/SkafteNicki/dtu_mlops/issues/647) and [Agentic modules for the MLOps
course](https://github.com/SkafteNicki/dtu_mlops/issues/618).

- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651):
    Rock-Paper-Scissors becomes the running example (pending the real-photo check), so the deployed app is where photos
    of hands arrive. No slide names the old dataset. → optional example for DEPLOYMENT-19
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): the pipeline runs as a
    Cloud Run job, so Cloud Run on p. 14 covers jobs as well as services. → see DEPLOYMENT-15
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): data lineage lives in
    W&B reference artifacts. That is the concrete form of "data lineage" on p. 28 and of the "Model v2.3 / Trained on
    2025-09 data" node on p. 29. → no finding; use as the example
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): W&B logging moves
    to Day 3, so the registry example can assume it. → see DEPLOYMENT-03
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    GitHub Actions gets Workload Identity Federation as best practice; deploy credentials follow the same line. → see
    DEPLOYMENT-16
- Restructure map, destination (deploy → collect new data → label → retrain → redeploy; closes in week 3): the p. 31
    scenario (automatic retrain and redeploy) is this loop without audit, so it is a natural bridge. The week-3 layout
    is still open. → see DEPLOYMENT-21
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621) and
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682): the deployment session
    gets agent-assisted deployment. The slides may carry the idea but do not reference the module. → see DEPLOYMENT-24
- [Lock the M/O/A taxonomy and naming convention](https://github.com/SkafteNicki/dtu_mlops/issues/619): module numbers
    change, so refer to modules by name. The deck uses no numbers. → none

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased (e.g. CLOUD-01) -->

### DEPLOYMENT-01 · Sculley citation drops an author

- **Category:** wrong
- **Priority:** must
- **Pages:** 2
- **Finding:** The reference lists nine authors; the paper has ten. Jean-François Crespo is missing.
- **Proposal:** Correct the author list; redraw the figure with the same citation (the image appears in several decks).
- **Evidence:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, p. 1 author block.

### DEPLOYMENT-02 · The 80% inference figure has no source

- **Category:** wrong
- **Priority:** must
- **Pages:** 3
- **Finding:** The slide states "Around 80% of compute spend" as fact without a source. The figures found differ: AWS
    says customers report "up to 90%" of cost, Google measured about 3/5 of ML **energy**, and Meta reports 65% of the
    **carbon footprint** for a language model. None says 80%, and they measure different things.
- **Proposal:** Replace the number with one sourced figure and name what it measures, or make the claim qualitative.
    Drop the flat 80%.
- **Evidence:** AWS News Blog, Inf1 launch; Patterson et al. 2022 (arXiv 2204.05149); Wu et al. 2021 (arXiv 2111.00364),
    URLs under Candidate references (accessed 2026-11-24). Origin of 80%: `unverified`.

### DEPLOYMENT-03 · MLflow stages are deprecated

- **Category:** outdated
- **Priority:** should
- **Pages:** 5
- **Finding:** The diagram shows Staging / Production / Archived columns. MLflow deprecated stages in 2.9.0 in favour of
    aliases (for example `@champion`) and tags. The course uses W&B rather than MLflow, so the tool is also drift.
- **Proposal:** Redraw as a tool-neutral registry (versions, aliases, promotion by CI) using the W&B wording students
    already know.
- **Evidence:** MLflow docs, "Model Registry Workflows", <https://mlflow.org/docs/latest/ml/model-registry/workflow/>
    (accessed 2026-11-24); tool choice: `s4_debugging_and_logging/logging.md`, W&B sections.

### DEPLOYMENT-04 · Pruning is not free

- **Category:** wrong
- **Priority:** must
- **Pages:** 6
- **Finding:** "without sacrificing accuracy" overstates it: Blalock et al. find pruning trades efficiency against
    quality, "(typically) decreasing the latter".
- **Proposal:** Say accuracy is traded off and has to be re-measured after each optimization; the same holds for
    quantization.
- **Evidence:** Blalock et al. 2020, *What is the State of Neural Network Pruning?*, <https://arxiv.org/abs/2003.03033>
    (accessed 2026-11-24); `s9_scalable_applications/inference.md`, "Quantization" for the overlap.

### DEPLOYMENT-05 · The latency chart is unreadable and has no takeaway

- **Category:** image
- **Priority:** should
- **Pages:** 6
- **Finding:** The Dell H100 chart is dark grey on black: axis and legend text can't be read in the render. It is also
    LLM-specific, and the cited page refused a fetch, so the model and units behind it could not be checked.
- **Proposal:** Re-plot from the source with one highlighted comparison, or replace it with the module's own ONNX
    benchmark.
- **Evidence:** Page render p. 6;
    <https://infohub.delltechnologies.com/en-us/p/unlocking-llm-performance-advanced-quantization-techniques-on-dell-server-configurations/>
    returned Access Denied (accessed 2026-11-24); `s7_deployment/ml_deployment.md`, Exercises step 5.

### DEPLOYMENT-06 · Packaging slide: Glow is archived and ONNX carries the module

- **Category:** outdated
- **Priority:** should
- **Pages:** 7
- **Finding:** Glow's repository is archived (last push May 2024). The three cards also mix categories: ONNX is an
    interchange format, TensorRT an inference SDK and Glow a compiler. The module teaches ONNX plus ONNX Runtime.
- **Proposal:** Drop Glow, separate format (ONNX) from runtime/compiler (ONNX Runtime, TensorRT), and say ONNX is what
    the exercises use.
- **Evidence:** <https://github.com/pytorch/glow> (archived); NVIDIA TensorRT docs,
    <https://docs.nvidia.com/tensorrt/index.html> (accessed 2026-11-24); `s7_deployment/ml_deployment.md`, "Model
    Packaging".

### DEPLOYMENT-07 · TorchServe is archived, and the tool list needs a refresh

- **Category:** outdated
- **Priority:** must
- **Pages:** 8
- **Finding:** TorchServe leads the list with no caveat, but its repository is archived and its docs warn of no security
    patches; the module says so too. Triton is now NVIDIA Dynamo-Triton, and the module's list also has LitServe,
    OpenVINO and Seldon.
- **Proposal:** Follow the module's list: mark or drop TorchServe, use "Triton Inference Server" (the module's name,
    still in use on GitHub) with a note on the new name, and add LitServe. vLLM stays.
- **Evidence:** `s7_deployment/ml_deployment.md`, intro (table and "Community" bullet);
    <https://github.com/pytorch/serve>; NVIDIA, <https://developer.nvidia.com/dynamo-triton> (accessed 2026-11-24).

### DEPLOYMENT-08 · The benchmark table has no source and compares an archived tool

- **Category:** image
- **Priority:** should
- **Pages:** 8
- **Finding:** The table compares vanilla gRPC, TorchServe and Triton with no source, hardware or model. One of the
    three is archived.
- **Proposal:** Drop it, or replace it with a chart of latency and throughput under load from the module's exercise.
- **Evidence:** Page render p. 8 (no citation on the slide); `s7_deployment/ml_deployment.md`, BentoML Exercises step 3;
    DEPLOYMENT-07. Origin of the table: `unverified`.

### DEPLOYMENT-09 · Why a serving framework: adaptive batching

- **Category:** gap
- **Priority:** should
- **Pages:** 8
- **Finding:** The slide lists engines but never says what they add. The module motivates them with dynamic/adaptive
    batching, async inference, GPU support and model composition.
- **Proposal:** Add one slide that contrasts FastAPI with a serving framework, with an adaptive-batching diagram
    (maximum batch size or timeout).
- **Evidence:** `s7_deployment/ml_deployment.md`, intro and "BentoML" Exercises steps 3–6.

### DEPLOYMENT-10 · HTTP and REST basics are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 9
- **Finding:** The core module has a full section on client/server, methods, status codes, JSON payloads and
    statelessness. The slide jumps to what an ML API defines.
- **Proposal:** Add one request/response slide before p. 9 (method, URL, payload, status code, stateless), with the
    FastAPI screen kept as the server side.
- **Evidence:** `s7_deployment/apis.md`, "Requests" and "Creating APIs".

### DEPLOYMENT-11 · The Dockerfile's `EXPOSE $PORT` contradicts the module

- **Category:** wrong
- **Priority:** must
- **Pages:** 10
- **Finding:** The slide has `EXPOSE $PORT`. The module explains that `EXPOSE` is evaluated at build time, when `$PORT`
    is unset, so it should stay at 8080. The file also copies `requirements.txt` without using it.
- **Proposal:** Use the module's dockerfile pattern as an imported code block: `EXPOSE 8080`, `--port $PORT`, `--host
    0.0.0.0`. Remove the unused copy.
- **Evidence:** `s7_deployment/cloud_deployment.md`, "Cloud Run" Exercises step 3, "Common problems";
    `s7_deployment/exercise_files/simple_fastapi_app.dockerfile`; Docker, "EXPOSE",
    <https://docs.docker.com/reference/dockerfile/#expose> (accessed 2026-11-24).

### DEPLOYMENT-12 · The road to deployment stops before serverless

- **Category:** gap
- **Priority:** should
- **Pages:** 11
- **Finding:** The timeline ends at Kubernetes. Both deployment targets in the core module are serverless, and the
    module names Kubernetes as the optional next step.
- **Proposal:** Add serverless (containers and functions) as a stop, with scale-to-zero and cold starts as its
    trade-off.
- **Evidence:** `s7_deployment/cloud_deployment.md`, intro and closing paragraph; Google Cloud, "About instance
    autoscaling", <https://docs.cloud.google.com/run/docs/about-instance-autoscaling> (accessed 2026-11-24).

### DEPLOYMENT-13 · Stream processing is not token streaming

- **Category:** wrong
- **Priority:** must
- **Pages:** 12
- **Finding:** The card defines stream processing as continuous data streams, then says it is "Perfect for LLM
    applications, chat interfaces ... progressive output". The figure next to it shows event streams through a broker.
    Chat output is a streamed HTTP response, so it belongs under request-response.
- **Proposal:** Give stream processing an event-stream example (the RabbitMQ case on p. 19 fits) and move token
    streaming to request-response.
- **Evidence:** Page render p. 12; FastAPI, "Server-Sent Events",
    <https://fastapi.tiangolo.com/tutorial/server-sent-events/> (accessed 2026-11-24) (streaming data to the client over
    HTTP).

### DEPLOYMENT-14 · The cloud slide names three platforms the module does not use

- **Category:** outdated
- **Priority:** should
- **Pages:** 13
- **Finding:** The slide is a title and three names, two of them renamed: SageMaker is SageMaker AI since December 2024,
    and Vertex AI Platform is now "Agent Platform". The gpt-oss-20B screenshot lists twelve API providers; the same page
    now lists four.
- **Proposal:** Keep one line on managed ML endpoints, "Vertex AI" with the "now part of the Gemini platform" note
    agreed for the Cloud deck, and replace the screenshot with managed endpoint vs Cloud Run vs hosted model API.
- **Evidence:** AWS, <https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html>; Google release notes,
    <https://docs.cloud.google.com/gemini-enterprise-agent-platform/release-notes>;
    <https://artificialanalysis.ai/models/gpt-oss-20b/providers> (accessed 2026-11-24); [Which Day 7 Cloud review
    changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710#issuecomment-7114948971) (CLOUD-06).

### DEPLOYMENT-15 · The "Where should I run my stuff?" figure predates today's Cloud Run

- **Category:** outdated
- **Priority:** should
- **Pages:** 14
- **Finding:** It shows Cloud Functions as a separate product and lists GPUs only under Compute and Kubernetes Engine.
    Functions are now Cloud Run functions, and Cloud Run supports GPUs that scale to zero. The module uses the same
    figure, which is third-party artwork.
- **Proposal:** Redraw as a small decision diagram with one Cloud Run box (services / jobs / functions), as agreed for
    the Cloud deck, with credit to the original. Keep the "it depends" line.
- **Evidence:** `s7_deployment/cloud_deployment.md`, intro and "Cloud Functions"; Google Cloud, "GPU support for
    services", <https://docs.cloud.google.com/run/docs/configuring/services/gpu> (accessed 2026-11-24); Cloud decision
    CLOUD-07 as in DEPLOYMENT-14.

### DEPLOYMENT-16 · Continuous deployment is missing

- **Category:** gap
- **Priority:** should
- **Pages:** 13–15
- **Finding:** Continuous deployment with a Cloud Build trigger is a learning objective and a core exercise, but no
    slide shows how a push becomes a running revision.
- **Proposal:** Add one flow slide: push → Cloud Build → Artifact Registry → Cloud Run revision, with secrets injected
    from Secret Manager. Reuse the Cloud deck's flow diagram.
- **Evidence:** `s7_deployment/README.md`, "Learning objectives"; `s7_deployment/cloud_deployment.md`, "Cloud Run"
    Exercises step 7 and "🧠 Knowledge check".

### DEPLOYMENT-17 · Testing before and after deployment is missing

- **Category:** gap
- **Priority:** should
- **Pages:** 15, 17
- **Finding:** API testing is a core module (functional tests with `TestClient`, load tests with Locust) and has no
    slide. Autoscaling and latency are mentioned without how to measure them.
- **Proposal:** Add one slide: functional test, then load test, then read p99 and requests per second, placed before the
    rollout strategies.
- **Evidence:** `s7_deployment/testing_apis.md`, "Testing for functionality", "Load testing" and "🧠 Knowledge check"
    question 3.

### DEPLOYMENT-18 · Rollout strategies: add rollback and the literature

- **Category:** addition
- **Priority:** should
- **Pages:** 15
- **Finding:** The slide covers shadow, canary and A/B but not rollback, and has no citation. Shankar et al. describe
    the stages, name the shadow stage and report keeping old versions as fallback models. Cloud Run does traffic
    splitting and rollback directly. No published module covers the topic.
- **Proposal:** Keep the three cards, add a rollback line with the Cloud Run mechanism, and cite the study. Note the
    drift; an unpublished module is tracked separately.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 6 §4.4.3 and p. 7 §4.5.2;
    Google Cloud, "Rollbacks, gradual rollouts, and traffic migration",
    <https://docs.cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration> (accessed 2026-11-24); [Complete
    deployment_testing.md module](https://github.com/SkafteNicki/dtu_mlops/issues/568).

### DEPLOYMENT-19 · Frontend and backend as the microservice example

- **Category:** gap
- **Priority:** should
- **Pages:** 16
- **Finding:** The slide stays abstract. The module introduces microservices through the Streamlit frontend and FastAPI
    backend that students deploy as two Cloud Run services.
- **Proposal:** Add that split as the concrete example next to the diagram.
- **Evidence:** `s7_deployment/frontend.md`, intro and "❔ Exercises" step 1–2.

### DEPLOYMENT-20 · The public endpoint and its risks

- **Category:** addition
- **Priority:** should
- **Pages:** 4, 17
- **Finding:** Authentication is one abstract card. The exercises deploy with `--allow-unauthenticated`, and the module
    only says authenticated invocations may come later.
- **Proposal:** Say what a public endpoint means (anyone can call it and you pay), and name the minimum controls:
    require authentication, cap instances, and keep secrets in Secret Manager.
- **Evidence:** `s7_deployment/cloud_deployment.md`, "Cloud Run" Exercises steps 3 and 5 and "🧠 Knowledge check"; Google
    Cloud, "Managing access", <https://docs.cloud.google.com/run/docs/securing/managing-access> (accessed 2026-11-24).

### DEPLOYMENT-21 · Both exercises link to a bare Excalidraw page

- **Category:** wrong
- **Priority:** must
- **Pages:** 20, 31
- **Finding:** The text asks students to open "the Excalidraw link (backup and press "Replace my content" if needed)",
    but the only link on either page goes to <https://excalidraw.com/>. No scene or backup URL survives in the PDF, so
    the p. 31 loop that groups should "complete" is lost.
- **Proposal:** Recover the two scenes from the author, or rebuild them as committed `.excalidraw` files and link those.
    Keep both scenarios.
- **Evidence:** Link annotations on pp. 20 and 31 of `slides/day8_deployment.pdf`. Whether scenes exist elsewhere:
    `unverified`.

### DEPLOYMENT-22 · "Same model" shows four models and the wrong interval

- **Category:** wrong
- **Priority:** must
- **Pages:** 23, 25
- **Finding:** The title says one model, but the cards are regression, forecasting, fraud classification and text
    generation. "87 units (±12)" is called a confidence interval, although an interval for a future observation is a
    prediction interval. This touches the keep list.
- **Proposal:** Show one model (demand) as a number, an interval, a threshold label and a narrative, so the title and
    the closing question hold. Say "prediction interval" on pp. 23 and 25.
- **Evidence:** Page renders pp. 23 and 25; Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd
    ed.), §5.5, <https://otexts.com/fpp3/prediction-intervals.html> (accessed 2026-11-24).

### DEPLOYMENT-23 · Tie accountability to the law

- **Category:** addition
- **Priority:** should
- **Pages:** 21, 28, 30
- **Finding:** The deck asks for traceability, decision logs and human oversight but cites no rule. The AI Act requires
    event logging (Art. 12) and human oversight (Art. 14) for high-risk systems, applying from 2 December 2027 (Annex
    III) or 2 August 2028 (Annex I). GDPR Art. 22 covers solely automated decisions.
- **Proposal:** Add one line per row on p. 28 or a small box on p. 21 with the article numbers and current dates.
    Re-check the dates before the lecture.
- **Evidence:** European Commission, "AI Act",
    <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai>; AI Act Arts. 12, 14 and 113 and GDPR
    Art. 22, URLs under Candidate references (accessed 2026-11-24).

### DEPLOYMENT-24 · Agents in the deployment step

- **Category:** addition
- **Priority:** should
- **Pages:** 15, 21
- **Finding:** The deck has no agentic content, though the course now uses agents as MLOps tools. The human-in-the-loop
    section already has the frame for it.
- **Proposal:** Extend p. 21: which deployment steps may an agent do alone (build, smoke test, read failed-deploy logs)
    and where a human approves (traffic shift, production credentials). Do not reference A-modules.
- **Evidence:** [Agentic modules for the MLOps course](https://github.com/SkafteNicki/dtu_mlops/issues/618) and [Update
    lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681); [Spec: Agentic
    modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), A8 row, for the split between building
    and triaging.

### DEPLOYMENT-25 · Three memes: one to drop, two for the author

- **Category:** image
- **Priority:** must
- **Pages:** 10, 22, 32
- **Finding:** The p. 32 closing meme goes under the cross-deck rule. The p. 10 "how Docker was born" strip uses film
    stills. The p. 22 post names a private person's account next to a vaccine claim; it carries the point, since IQ 80
    is the 9th percentile and "top 90.88%" is easily misread.
- **Proposal:** Drop p. 32. Author's call on p. 10 (regenerate as a cartoon, captions verbatim). For p. 22, rebuild only
    the score screen, without the account or the vaccine claim.
- **Evidence:** [How should images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705); no-"Meme of the day" rule in [Which Day 7 Cloud
    review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710#issuecomment-7114948971); P(X <
    80) = 0.0912 for N(100, 15), matching the 9.12% on the render.

### Also noticed (no id, not decided)

- p. 3: "Operations is Model Deployment" reads as if operations were only deployment, while the p. 2 figure also has
    monitoring.
- p. 4: the "Stability" icon is black on black. p. 9: the API screenshot shows monitoring endpoints that no exercise
    here builds.
- p. 12: the figure credits Aurimas Griciūnas (SwirlAI) inside the image only; the redraw should credit it on the slide.
- p. 15: "Final step" comes before eight more pages of deployment content; consider moving p. 16–17 before it.
- p. 19: the figure text has "MonoDB" (the bullet says MongoDB) and a misspelt "Intelligence", and cites no permission
    for the diagram.
- p. 28: the quote is in quotation marks without a source, and a search found none (2026-11-24). If it is the author's
    own, drop the marks.
- p. 29: "Potential failure successfully avoided" can't be observed; after an intervention the counterfactual is
    unknown. The date 2025-09 needs refreshing.
- p. 1: title is "Day8 - Deployment", January 2026. Topic-only naming would make it "Deployment", January 2027.
- FP32 to INT8 on p. 6 overlaps the inference module in `s9_scalable_applications/`; the two lectures should agree on
    who owns it.

## Module follow-ups (out of scope for the deck)

- `pages/timeplan.md`, Deployment row: lists Torchserve, which is archived and not taught, and "GCP Functions" (now
    Cloud Run functions).
- `s7_deployment/cloud_deployment.md`, "Cloud Functions": the heading and the figure caption keep the old name although
    the text explains the rename; the figure itself shows Cloud Functions and no GPUs on Cloud Run.
- `s7_deployment/cloud_deployment.md`, closing paragraph: recommends "Vertex AI Pipelines" docs for going further with
    Kubernetes; the product is renamed and the link does not match the sentence.
- `s7_deployment/apis.md`, "Creating APIs" exercises: builds `--port 80` containers while Cloud Run exercises need
    `$PORT`; a pointer would help.
- `s7_deployment/testing_apis.md`, "Load testing" step 6: uses a JSON key (`credentials_json`), where the restructure
    adds Workload Identity Federation.
- `s7_deployment/README.md`, "Learning objectives": typo "continuouss".
- Rollout strategies (p. 15) have no published module; see [Complete deployment_testing.md
    module](https://github.com/SkafteNicki/dtu_mlops/issues/568).

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | Sculley et al. Fig. 1 with its caption | chart/paper figure | re-plot/redraw + citation | | DEPLOYMENT-01 |
| 3 | Cloud, arrows and a person at a "y=1" screen | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 4 | Five small illustrations: data centre, laptops, latency vs throughput trucks, clock, padlock | concept diagram | Iconify icons; redraw the latency vs throughput pair in Excalidraw | | |
| 5 | MLflow tracking server and model registry with stages | concept diagram | Mermaid/PlantUML/Excalidraw | stale | DEPLOYMENT-03 |
| 6 | Dell "Total inference latency – H100" bars by batch size | chart/paper figure | re-plot/redraw + citation | | DEPLOYMENT-05 |
| 7 | TensorRT, ONNX and Glow logos | tool logo | Iconify logo (ONNX and TensorRT unverified); drop Glow | stale | DEPLOYMENT-06 |
| 8 | Response time plots and request counts: vanilla gRPC, TorchServe, Triton | chart/paper figure | replace with a cited plot, or drop | stale | DEPLOYMENT-08 |
| 9 | FastAPI docs page with `/predict` and two monitoring endpoints | tool screenshot | fresh screenshot | | |
| 10 | Three-panel film meme "it works on my machine ... how Docker was born" | meme/joke | keep / regenerate / swap? | voice | DEPLOYMENT-25 |
| 10 | Dockerfile (set as text) | code/config screenshot | code block from `exercise_files/simple_fastapi_app.dockerfile` | | DEPLOYMENT-11 |
| 11 | Road with four stops, physical servers to Kubernetes | concept diagram | Mermaid/PlantUML/Excalidraw | | DEPLOYMENT-12 |
| 12 | SwirlAI "Machine Learning Deployment Types" (batch, stream, request-response, edge) | concept diagram | Mermaid/PlantUML/Excalidraw, with credit | | DEPLOYMENT-13 |
| 13 | Artificial Analysis gpt-oss-20B provider page | tool screenshot | replace; fresh screenshot if kept | stale | DEPLOYMENT-14 |
| 14 | "Where should I run my stuff?" sketchnote (@pvergadia, thecloudgirl.dev) | concept diagram | Excalidraw redraw + credit | stale | DEPLOYMENT-15 |
| 16 | Monolith box vs eight connected services | concept diagram | Mermaid/PlantUML/Excalidraw | | DEPLOYMENT-19 |
| 18 | Three phone screens of a Danish car-insurance driving app | tool screenshot | ask for current screenshots and permission, or describe in text | | |
| 19 | AWS and Azure architecture of the same app | concept diagram | Mermaid/PlantUML/Excalidraw, with permission | | |
| 20, 31 | Excalidraw link card | tool logo | Iconify logo plus working link | | DEPLOYMENT-21 |
| 21 | Human-in-the-loop vs fully automated | concept diagram | Mermaid/PlantUML/Excalidraw | | DEPLOYMENT-24 |
| 22 | Social-media post misreading an IQ result (named account, vaccine claim) | meme/joke | regenerate (score screen only) | voice | DEPLOYMENT-25 |
| 29 | Deployment trace timeline, six steps | concept diagram | Mermaid/PlantUML/Excalidraw | stale | |
| 32 | "Small fix on a Friday" TV-cartoon meme | meme/joke | drop (cross-deck rule) | voice | DEPLOYMENT-25 |
| 7 | Banner of servers, a wireframe head and people | photo | replaced by theme | | |
| 1–32 | Dark Gamma background, card icons, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo · meme/joke ·
decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo · fresh screenshot
· re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated · replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html> | DEPLOYMENT-01, p. 2 figure |
| AWS News Blog, *Amazon EC2 Update – Inf1 Instances* | <https://aws.amazon.com/blogs/aws/amazon-ec2-update-inf1-instances-with-aws-inferentia-chips-for-high-performance-cost-effective-inferencing/> | DEPLOYMENT-02, p. 3 claim "80% of compute spend" |
| Patterson et al. 2022, *The Carbon Footprint of Machine Learning Training Will Plateau, Then Shrink* | <https://arxiv.org/abs/2204.05149> | DEPLOYMENT-02 |
| Wu et al. 2021, *Sustainable AI* | <https://arxiv.org/abs/2111.00364> | DEPLOYMENT-02 |
| MLflow, *Model Registry Workflows* | <https://mlflow.org/docs/latest/ml/model-registry/workflow/> | DEPLOYMENT-03 |
| Blalock et al. 2020, *What is the State of Neural Network Pruning?* | <https://arxiv.org/abs/2003.03033> | DEPLOYMENT-04, p. 6 "without sacrificing accuracy" |
| TorchServe README and docs | <https://github.com/pytorch/serve> | DEPLOYMENT-07, p. 8 |
| NVIDIA, *Dynamo-Triton* | <https://developer.nvidia.com/dynamo-triton> | DEPLOYMENT-07 |
| vLLM documentation | <https://docs.vllm.ai/en/latest/> | p. 8 claim "paged memory management" |
| Docker, *Dockerfile reference: EXPOSE* | <https://docs.docker.com/reference/dockerfile/#expose> | DEPLOYMENT-11 |
| Google Cloud, *About instance autoscaling* | <https://docs.cloud.google.com/run/docs/about-instance-autoscaling> | DEPLOYMENT-12, p. 17 "Autoscaler" |
| FastAPI, *Server-Sent Events* | <https://fastapi.tiangolo.com/tutorial/server-sent-events/> | DEPLOYMENT-13 |
| Google Cloud, *GPU support for services* | <https://docs.cloud.google.com/run/docs/configuring/services/gpu> | DEPLOYMENT-15, p. 20 scenario |
| Shankar et al., *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | DEPLOYMENT-18, p. 15 |
| Google Cloud, *Rollbacks, gradual rollouts, and traffic migration* | <https://docs.cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration> | DEPLOYMENT-18 |
| Fowler and Lewis, *Microservices* | <https://martinfowler.com/articles/microservices.html> | DEPLOYMENT-19, p. 16 (linked from `s7_deployment/frontend.md`) |
| Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, §5.5 | <https://otexts.com/fpp3/prediction-intervals.html> | DEPLOYMENT-22, p. 23 "confidence interval" |
| AI Act, Arts. 12, 14, 113 (consolidated text, unofficial site) | <https://artificialintelligenceact.eu/article/12/>, <https://artificialintelligenceact.eu/article/14/>, <https://artificialintelligenceact.eu/article/113/> | DEPLOYMENT-23 |
| European Commission, *AI Act* | <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai> | DEPLOYMENT-23 dates |
| GDPR, Art. 22 | <https://gdpr-info.eu/art-22-gdpr/> | DEPLOYMENT-23, p. 21 "fully automated" |
