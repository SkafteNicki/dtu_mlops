# Research: how industry monitors deployed ML models (2025-2026)

Input for the S8 Monitoring rework (`s8_monitoring/monitoring.md`, `s8_monitoring/data_drifting.md`). Researched
2026-11-27. Every claim has a source in the [Sources](#sources) list, referenced as [S1], [S2], and so on. Anything I
could not verify against a primary source is marked **(unverified)**. Release dates and star counts come from the
GitHub API, queried on 2026-11-27.

## TL;DR

- **Prometheus + Grafana is still the open-source default; OpenTelemetry (OTel) is the instrumentation and transport
    layer that is growing around it.** In Grafana's 2025 survey 67% of organisations ran Prometheus in production and
    41% ran OTel; 71% used both in some capacity [S1]. Prometheus 3.x can ingest OTLP metrics natively, so "OTel in the
    app, Prometheus as the store" is a supported setup [S2]. Teaching Prometheus + Grafana transfers.
- **Managed observability (Grafana Cloud, Datadog, cloud-native) is common, but PromQL, exporters, and alert rules
    carry over to it.** Grafana Cloud has an always-free tier with usage limits and 14-day retention [S3]. Datadog was not
    checked against a primary source **(unverified)**.
- **The dedicated drift-vendor landscape has thinned. For students, Evidently is the only well-maintained
    open-source option left.** WhyLabs shut down; whylogs has had no release since Dec 2024 [S4, S5]. NannyML was
    acquired by Soda in June 2025, has had no release since July 2025, and users report an unmaintained repo [S6, S7].
    Evidently (v0.7.23, 11 Sep 2026) and Arize Phoenix (v22.1.0, 26 Nov 2026) are actively released [S8, S9]. Phoenix
    is an LLM tracing/evaluation tool, not a tabular or image drift tool, judging by its repo description.
- **The cloud vendors' drift services are not a safe thing to teach.** AWS closed SageMaker Model Monitor to new
    customers and now documents a replacement built on **Evidently + MLflow + Athena + EventBridge/Lambda + SNS** [S10].
    That is "open-source library + scheduled batch job + alert", the same shape as the course design. Vertex AI Model
    Monitoring v2 is still in Preview [S11].
- **Industry drift defaults are simple distribution distances, not exotic tests.** Evidently's default per column is
    KS (numerical) or chi-squared (categorical) for <=1000 reference rows, and Wasserstein or Jensen-Shannon above
    that [S12]. Vertex uses Jensen-Shannon and L-infinity [S11]. PSI is a common finance/industry habit, available in
    Evidently but not its default [S13]. KS p-values become oversensitive at large N [S13], which is a good teaching
    point for the hand-rolled version.
- **For images there is no built-in default.** The standard approach is to reduce the image to features or
    embeddings first and then run a two-sample test or MMD on those [S14, S15]. Evidently's embedding drift is
    documented as "coming soon" in the new API [S16], so a hand-rolled or Alibi Detect path is needed for embeddings.
- **Drift jobs are scheduled batch jobs, not streams, at course scale.** Uber runs real-time feature checks on
    Flink/Pinot at its scale [S17]. AWS's reference design uses EventBridge-triggered Lambda on logged predictions [S10].
- **Alert on symptoms and SLO burn, put drift on dashboards or tickets.** This follows the Google SRE book,
    workbook, and Prometheus guidance [S18, S19, S20]. "Drift means page someone" is my inference, not a sourced industry
    rule. Uber's closest practice is blocking a model promotion on feature-health issues rather than paging [S17].

## 1. Metrics stack

### Prometheus + Grafana versus managed options

- **Adoption.** The Grafana Labs 2025 survey (published March 2025) reports 67% of organisations use Prometheus in
    production and 41% use OpenTelemetry in production, with 38% more investigating OTel. 71% use both in some
    capacity, and half of organisations increased investment in both for the second year [S1]. Grafana sells the stack, so
    treat this as an indication from its own user base. A later survey wave (77% investing in Prometheus, 76% in OTel;
    OTel used for metrics by 57%) is only reported by a secondary source, ItBrief **(unverified against Grafana)** [S21].
- **Self-hosted stays healthy.** Prometheus (v3.16.0, 5 Nov 2026), Alertmanager (v0.35.1, 24 Nov 2026), Pushgateway
    (v1.11.4, 8 Oct 2026), and Grafana (v13.3.2, 24 Nov 2026) all have releases within the last two months, and none is
    archived [S22].
- **Managed is the common alternative, and it is Prometheus-compatible.** Grafana Cloud's free tier is "Always free"
    with all services subject to usage limits and 14-day retention [S3]. Google also offers Managed Service for
    Prometheus (see section 3). Datadog and other vendors exist but I did not read their docs **(unverified)**.
- **Alerting.** Prometheus evaluates alert rules and Alertmanager routes and deduplicates notifications [S20]. Grafana's
    own alerting is the other common choice. I did not read Grafana's alerting docs in this run **(unverified)**, so
    compare the two before choosing one for the course.

### Where OpenTelemetry fits for metrics

- OTel is a vendor-neutral way to instrument code and ship telemetry. For **metrics** it often sits in front of
    Prometheus rather than replacing it. Prometheus can receive OTLP metrics over HTTP when started with
    `--web.enable-otlp-receiver` at `/api/v1/otlp/v1/metrics`. The receiver is off by default because Prometheus has no
    authentication [S2].
- OTLP/HTTP is required (not gRPC) for the Prometheus receiver. The Prometheus guide sets
    `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf` [S2].
- OTel's maturity differs by signal. The Grafana survey shows OTel production use at 41% overall [S1]. I did not
    check per-language metric stability, such as the Python SDK, against the OTel status page **(unverified)**. The
    Python SDK is actively released (v1.45.2, 29 Oct 2026) and the Collector is at v0.166.0 [S22].
- Grafana Alloy is described as an "OpenTelemetry Collector distribution" and is actively released (v1.22.0) [S22].
    That makes a Collector the likely future path from app to backend, but a direct `prometheus_client` `/metrics`
    endpoint stays simpler for a course.

## 2. Drift detection practice

### Tool status (checked 2026-11-27)

| Tool | Status | Evidence |
| --- | --- | --- |
| **Evidently** | Alive. v0.7.23 on 11 Sep 2026, not archived, about 8.1k stars. Cadence slowed in 2026 (v0.7.21 on 10 Mar, v0.7.23 on 11 Sep). | [S8] |
| **Arize Phoenix** | Very active (v22.1.0 on 26 Nov 2026). Repo is described as "AI Observability & Evaluation". It is focused on LLM tracing and evals, so it is not a fit for tabular or image drift (judged from the repo description only). | [S9] |
| **Arize AX** (commercial) | Client library `client_python` v8.66.0 released 23 Nov 2026, so the commercial platform appears alive. Feature set not checked. | [S9] |
| **Alibi Detect** | Maintained but slow. v0.13.0 on 11 Dec 2025, last push 11 Dec 2025. | [S23] |
| **NannyML** | **Effectively unmaintained.** Soda acquired it on 9 Jun 2025 [S6]. The last release (v0.13.1) and last commit were 12 Jul 2025 [S7]. Issue #450 (from Apr 2026) has users reporting install failures and asking whether it is maintained. A maintainer replied "it's not my intent to let the NannyML library die" (20 Apr 2026), then went quiet. Issue #451 is titled "Stale repository since soda acquisition" [S24]. The repo is not archived. I did not re-verify the Soda statement that the open-source project would continue **(unverified)**. |
| **WhyLabs (company)** | **Shut down.** whylabs.ai states "WhyLabs, Inc. is discontinuing operations" and that "the complete WhyLabs platform has been open sourced" [S4]. Platform source is in `whylabs/whylabs-oss`, created 22 Jan 2025 [S5]. Secondary sites give 23 Jan 2025 as the open-source date and 9 Mar 2025 as the end of hosted access **(unverified)**. An Apple acquisition is reported by third-party sites only and I found no primary confirmation **(unverified)**. |
| **whylogs** | Library remains open source, but the last release was v1.6.4 on 3 Dec 2024 and the last push 10 Jan 2025 [S5]. Treat as dormant. |
| **LangKit** (WhyLabs) | Last release v0.0.35 on 6 Nov 2024 [S5]. Dormant. |
| **Deepchecks** | Last release 0.19.1 on 15 Dec 2024, last push 28 Dec 2025 [S22]. Lightly maintained. |
| **Fiddler** | Not checked **(unverified)**. |
| **SageMaker Model Monitor** | Closed to new customers, "we do not plan to introduce new features". AWS documents a replacement from open-source monitoring solutions (SageMaker MLflow Apps + Evidently AI), Athena, EventBridge, Lambda, SNS, QuickSight, and CloudWatch [S10]. |
| **Vertex AI Model Monitoring** | v2 is in Preview. v1 is GA and tied to Vertex endpoints. It supports Jensen-Shannon (numerical and categorical) and L-infinity (categorical) for input and output drift, plus SHAP feature-attribution drift [S11]. The doc nav now labels the product "Agent Platform", so names are shifting. Whether it works for a model served on Cloud Run: I did not find a statement either way **(unverified)**. |

Takeaway: the "buy a drift platform" market looks smaller than it did in 2022-2023. One vendor closed, one was absorbed and
went quiet, and AWS pointed its own customers to open-source Evidently. The durable skills are the statistics and the
batch-job-plus-alert pattern.

### Which statistical tests are the defaults

- **Evidently's defaults** [S12]:
    - Reference rows <= 1000: two-sample KS for numerical columns (n_unique > 5), chi-squared for categorical columns
      (or numerical with n_unique <= 5), and a Z-score proportion test for binary columns. Drift score is a p-value, with
      drift at <= 0.05.
    - Reference rows > 1000: Wasserstein distance for numerical, Jensen-Shannon divergence for categorical, with
      drift at a distance >= 0.1.
    - PSI, KL divergence, and other methods are selectable per column [S12].
- **Evidently's own experiment on large datasets** (published June 2022, updated July 2025) compared KS, PSI,
    Wasserstein, KL, and Jensen-Shannon. KS "raises flags even for a minor change of 0.5%" once there are more than
    100,000 rows. Evidently recommends KS under about 1000 rows, or sampling before using it on large data. They
    recommend PSI for industries that already use it and want rule-of-thumb thresholds for stakeholders [S13].
- **Vertex** uses Jensen-Shannon and L-infinity [S11]. SageMaker Model Monitor used Deequ-based baseline constraints
    and statistics [S25].
- **A benchmark of three open-source tools** on two smart-building datasets found Evidently best for general drift
    detection and NannyML best at timing shifts and linking them to accuracy. Sample size is small (two use cases) [S26].
- **Practical reading:** KS is the textbook default, distance measures (Wasserstein, Jensen-Shannon, PSI) take over at
    scale, and everyone exposes the threshold as a knob. That is exactly the right framing for the hand-rolled
    exercise: p-value tests flag everything at large N, so students should see effect size and sample size interact.

### Images, embeddings, MMD

- The research baseline is *Failing Loudly* (NeurIPS 2019). Its abstract finds that two-sample testing on features
    reduced by a pre-trained classifier performs best across shifts, and that classifier-based methods do poorly with
    little data [S14]. Black-box shift detection (BBSD) uses the model's own softmax outputs for the test [S14].
- **Alibi Detect** implements this. Its MMD detector takes a `preprocess_fn` and the docs show a randomly initialised
    image encoder, or hidden-layer output of the model, as the dimensionality-reduction step, with a CIFAR-10
    example [S15].
- **Evidently's legacy embedding drift** default trains a classifier to separate reference from current embeddings
    (drift if ROC AUC is above 0.55 for >1000 rows) and also offers Euclidean, cosine, MMD, and share-of-drifted-embeddings
    methods. The current docs page says the method is "coming soon" to the new API and points to the old docs [S16].
    This is a gap for a 2026 course built on Evidently 0.7.
- **Uber's October 2025 post** says "statistical drift detection is no longer enough" for embedding-heavy systems [S17]
    (quote via a fetch of the post. The post does not name tests).
- **Implication for Rock-Paper-Scissors:** the cheapest transferable approach is (a) per-image summary features
    (brightness, contrast, colour channel means, edge density) with KS, (b) predicted-class distribution with PSI or
    chi-squared, and optionally (c) penultimate-layer embeddings from the trained CNN with a classifier two-sample test
    or MMD. (c) can be hand-rolled with scikit-learn in a few lines, or done with Alibi Detect.

### How much has Evidently's API changed?

A lot, and recently [S27]:

- Release dates for reference: 0.4.0 on 26 Jul 2023, 0.5.0 on 4 Dec 2024, 0.6.0 on 24 Jan 2025, 0.7.0 on 10 Apr 2025 [S8]. Older history (dashboards and JSON profiles deprecated in 0.1.59 and removed in 0.3.0) is on the old migration page **(seen via search snippet only)**.
- **0.6.0 (Jan 2025):** new core API available as `evidently.future`; legacy API co-existed through 0.6.7 [S27].
- **0.7.0 (Apr 2025): breaking.** New API becomes the default (`from evidently import Report`). `column_mapping`
    is replaced by `DataDefinition`, and data must be wrapped in an explicit `Dataset` before `report.run(...)` [S27].
    Anyone on the old API must rewrite code or pin `evidently<=0.6.7`.
- Embedding drift and some other legacy metrics are not yet in the new API (see above) [S16].

Teaching consequence: any exercise using Evidently has a shelf life of months, so pin the version in
`pyproject.toml` and keep the Evidently step short. The hand-rolled statistics (scipy) do not have this problem.

### Estimating performance without labels

- NannyML's CBPE (classification) uses the model's predicted probabilities to estimate the confusion matrix and any
    metric derived from it. Its docs state that, assuming calibrated probabilities and the stated assumptions, it gives an
    unbiased estimate without targets. A follow-up NannyML paper names the main weakness: reliance on calibration, which
    "is known to deteriorate under certain distributional shifts" [S28, S29]. DLE (regression) trains an extra LightGBM
    model to predict the loss per row [S28].
- Both are NannyML-specific, and NannyML is stale (above). I read the paper abstract, not the current docs' assumption list
    **(unverified in detail)**. For the course, the concept is worth one paragraph: a model's confidence is a free
    label-less signal, which also motivates the hand-rolled "PSI on predicted classes" check.
- In production at Uber, performance is also monitored by joining logged predictions to outcomes as labels arrive, a
    pattern the lifecycle loop in this course already imitates [S17] (Michelangelo's hourly batch comparison is reported
    by secondary sources only **(unverified)**).

## 3. Architecture

### Where monitoring runs relative to serverless inference

- **Cloud Run request-based billing (default):** instances are billed, and get CPU, only while processing requests
    and during start/shutdown. **Instance-based billing** keeps CPU allocated outside requests and the docs explicitly cite
    "monitoring agents like OpenTelemetry" as a reason to use it [S30]. Idle instances are not guaranteed: an instance
    never stays idle more than 15 minutes after a request unless minimum instances is set [S30].
- **Google's recommended path on Cloud Run** is the Managed Service for Prometheus sidecar, which scrapes the app
    container locally, does a start-up scrape after 10 s and a shut-down scrape, and recommends instance-based
    billing because scrapes "might fail if the CPU allocation is throttled" [S31]. A separate OTLP collector-sidecar
    tutorial exists [S32]. These are the GCP-native options the course decided to drop. They are the real-world answer
    for Cloud Run, so mention them as "what you would do in a GCP-only shop".
- **Prometheus on a VM scraping a Cloud Run URL:** the Prometheus-community answer for any load-balanced service is that
    scraping through a load balancer hits a random instance, which breaks counter monotonicity and gives gaps, so scrape
    each instance directly [S33]. I found no Cloud Run-specific doc on this, only mailing-list threads from 2020-2022,
    and scale-to-zero makes it worse **(inference)**. For a course demo it works if the service runs a single instance
    (min-instances=1, max-instances=1, instance-based billing). Say so explicitly, because students will otherwise copy
    it for a real autoscaled service.

### Pull versus push

| Option | Fits | Caveat |
| --- | --- | --- |
| Prometheus pull from the VM to the Cloud Run URL | Single-instance demo | Random instance per scrape when autoscaled [S33]. CPU throttling under request-based billing [S30, S31]. |
| Pushgateway | A **drift or batch job pushing its result** | Prometheus docs say the usual valid use is "the outcome of a service-level batch job". Not for per-request metrics, and stale series must be deleted manually [S34, S35]. |
| OTLP push to Prometheus (`--web.enable-otlp-receiver`) | App pushes its own metrics to the VM | Receiver is off by default and has no auth, so firewall or proxy it. OTLP/HTTP only [S2]. Background pushes need CPU outside requests [S30]. |
| `remote_write` | Agent/Collector in front of a hosted backend | Not tested here. Standard for Grafana Cloud and managed backends **(unverified)**. |
| Cloud Run sidecar + Managed Prometheus | GCP-native production | Dropped from the course (see above) [S31]. |

Cleanest split for the course: **request metrics via scrape** (single-instance demo, with the caveat stated) and **drift
results via Pushgateway** from the batch job, which is the textbook Pushgateway use case [S34].

### Logging prediction inputs and outputs

- **Uber** runs real-time checks that "continuously scan prediction logs" [S17].
- **AWS** reference replacement for Model Monitor: inference data lands in an Athena/Iceberg lake, an
  EventBridge-triggered Lambda runs the Evidently analysis on a schedule, SNS sends alerts, QuickSight shows dashboards [S10].
  The old Model Monitor used SageMaker Data Capture to S3 [S25].
- **GCP:** Vertex has request-response logging into BigQuery (I only saw it in the docs navigation) **(unverified)**.
  For Cloud Run the straightforward option is to write one structured record per request to Cloud Storage or BigQuery
  (via a Cloud Logging sink). I did not retrieve the sink docs **(unverified)**.
- For images, log **features and embeddings plus a reference to the stored image**, not the image bytes in the log
  stream. This matches the lifecycle loop, where uploaded phone photos are stored for labelling and retraining.

### Batch or stream?

- At Uber's scale, feature health is checked in real time, in a Flink job backed by Pinot, while model-promotion gates
  rely on offline baselines computed at training time (percentiles, averages, null rates) [S17].
- AWS's own reference design is **scheduled**: EventBridge triggers Lambda [S10]. Model Monitor itself ran scheduled
  processing jobs [S10, S25].
- For a university course, a scheduled batch job (Cloud Run job, cron on the VM, or Cloud Scheduler) over the latest
  N logged requests is the right model. Streams add complexity without teaching anything about drift.

## 4. Alerting practice for ML

- **Alert on symptoms, not causes.** The SRE book separates "what's broken" (symptom) from "why" (cause) and says
  rules that page humans "should be simple to understand and represent a clear failure". Dashboards cover the four
  golden signals (latency, traffic, errors, saturation) and ad hoc debugging [S18]. Prometheus' guidance summarises it
  as "alert on symptoms, have good consoles to allow pinpointing causes, and avoid having pages where there is nothing
  to do" [S20].
- **SLOs and burn rates.** The SRE workbook's recommended multi-window, multi-burn-rate setup pages at a 14.4x
  burn over 1 hour (2% of budget) or a 6x burn over 6 hours (5%), and opens a ticket at a 1x burn over 3 days (10%) [S19].
  A 99.9% availability or latency SLO on `/predict` is a natural fit for the course service and has a worked
  PromQL form [S19].
- **What is ML-specific.** None of the sources I read prescribes SLOs for drift. My recommendation (inference): page on
  the service SLO (errors, latency), and treat drift scores, prediction-class shifts, and data-quality checks as
  dashboard items that raise a ticket or message above a threshold. Uber's closest documented practice is
  automatic alerts plus, "in high-risk cases", blocking model promotion until feature issues are resolved [S17].
- **Alert fatigue.** The SRE book's guidance is that pager-bound rules must keep noise low and signal high [S18].
  Evidently's finding that KS flags a 0.5% shift at 100k rows [S13] is a concrete ML example of a noisy alert. A
  drift alert that fires every run teaches students to ignore alerts.
- **Delayed labels** mean accuracy cannot be alerted on in real time. That is why label-free proxies (drift, predicted-class
  distribution, confidence) are used, and why a ground-truth join runs later [S17, S28].

## 5. Recommendation for a 3-week university MLOps course

1. **Teach Prometheus + Grafana + Alertmanager (or Grafana alerting) with the app instrumented via `prometheus_client`.**
   It is the common open-source stack [S1], PromQL and alert rules transfer to managed offerings, and the course already
   runs it in docker compose. Add one paragraph on OTel: instrumentation standard, Collector, and the Prometheus OTLP
   receiver [S2], without making students build on it.
2. **Make the SRE ideas the alerting lesson.** Four golden signals on the dashboard, one symptom-based alert with a
   burn-rate flavour (even a simplified one), a ticket-level alert for slow burn [S18, S19]. Ask students to justify
   why drift is not a page.
3. **Be honest about serverless.** Use a single-instance Cloud Run demo for scraping. Show the failure mode (several
   instances, random scrape target) in a few sentences and name the managed sidecar as the production answer [S31, S33].
   Use Pushgateway only for the drift job [S34].
4. **Hand-roll the drift statistics first, then show a library.** Hand-rolled is worth teaching here because:
   (a) the tests are 5-10 lines of scipy/numpy (`ks_2samp`, a PSI function, `chi2_contingency`);
   (b) the real lesson is thresholds, sample size, and multiple testing, which a library hides behind defaults [S12, S13];
   (c) there is no stable library default for images, so students must pick features themselves [S14, S15, S16];
   (d) the library API has churned (0.6 to 0.7), while the statistics will not [S27].
   Then run the same check through Evidently (`DataDriftPreset`) and have students compare p-values and distances.
   The AWS replacement design for Model Monitor, which is Evidently on a schedule [S10], is a good "this is what
   production looks like" slide.
5. **Do not teach NannyML or WhyLabs as tools.** Both are effectively dead or stale [S4, S5, S6, S7, S24]. Mention
   CBPE as an idea. Mention managed drift services (Vertex, SageMaker) as examples of the same checks behind an API,
   noting SageMaker Model Monitor is closed to new customers [S10] and Vertex v2 is Preview [S11].
6. **Keep the drift job a scheduled batch.** Read the last N logged requests from storage, compute the statistics,
   push `drift_score` gauges to the Pushgateway, and let Prometheus alert or Grafana show it [S10, S34].
7. **Pin versions.** Pin Evidently (`<0.8`, or the exact tested version) and record why [S27].

Open questions for the author:

- Whether to include the embedding-based check, given Evidently has no new-API implementation yet [S16].
- Whether to check SageMaker/Vertex-style managed monitoring beyond a slide. I did not verify whether Vertex monitoring
  works for Cloud Run-hosted models.
- Netflix and Shopify: I did not find primary engineering posts on ML monitoring practice. Uber's post was readable;
  DoorDash's 2021 post returned 403 to my fetch and is only known through summaries **(unverified)**.

## Sources

Fetched or queried directly on 2026-11-27 unless noted.

- [S1] Grafana Labs, Observability Survey Report 2025: <https://grafana.com/observability-survey/2025/>
- [S2] Prometheus docs, "Using Prometheus as your OpenTelemetry backend": <https://prometheus.io/docs/guides/opentelemetry/>
- [S3] Grafana Cloud pricing page: <https://grafana.com/pricing/>
- [S4] WhyLabs notice: <https://whylabs.ai> and <https://docs.whylabs.ai>
- [S5] GitHub: <https://github.com/whylabs/whylabs-oss>, <https://github.com/whylabs/whylogs>, <https://github.com/whylabs/langkit> (repo metadata and releases via the GitHub API)
- [S6] Soda, "Soda has acquired ML monitoring startup NannyML" (9 Jun 2025): <https://www.soda.io/blog/soda-acquires-nannyml>
- [S7] GitHub: <https://github.com/NannyML/nannyml> (latest release v0.13.1, 12 Jul 2025, and commit history via the API)
- [S8] GitHub: <https://github.com/evidentlyai/evidently> (releases via the API)
- [S9] GitHub: <https://github.com/Arize-ai/phoenix>, <https://github.com/Arize-ai/client_python>
- [S10] AWS, "Amazon SageMaker Model Monitor availability change": <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-availability-change.html> and <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html>
- [S11] Google Cloud, Vertex AI Model Monitoring overview: <https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview>
- [S12] Evidently docs, "Data drift" explainer (default tests): <https://docs.evidentlyai.com/metrics/explainer_drift>
- [S13] Evidently, "Which test is the best? We compared 5 methods to detect data drift on large datasets" (published 20 Jun 2022, updated 16 Jul 2025): <https://www.evidentlyai.com/blog/data-drift-detection-large-datasets>
- [S14] Rabanser, Günnemann, Lipton, "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift", NeurIPS 2019: <https://papers.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html> (I read the abstract only)
- [S15] Alibi Detect docs, MMD drift detector: <https://docs.seldon.ai/alibi-detect/cd/methods/mmddrift>
- [S16] Evidently docs, "Customize Embedding Drift": <https://docs.evidentlyai.com/metrics/customize_embedding_drift.md>
- [S17] Uber, "Raising the Bar on ML Model Deployment Safety" (30 Oct 2025): <https://www.uber.com/us/en/blog/raising-the-bar-on-ml-model-deployment-safety/> (read via a fetch agent. Quotes are short and the post names no specific tests)
- [S18] Google SRE book, ch. 6 "Monitoring Distributed Systems": <https://sre.google/sre-book/monitoring-distributed-systems/>
- [S19] Google SRE workbook, ch. 5 "Alerting on SLOs": <https://sre.google/workbook/alerting-on-slos/>
- [S20] Prometheus docs, "Alerting" best practices: <https://prometheus.io/docs/practices/alerting/>
- [S21] ItBrief on a later Grafana survey wave (secondary): <https://itbrief.news/story/ai-cost-pressures-reshape-observability-in-new-survey>
- [S22] GitHub releases (via API) for <https://github.com/prometheus/prometheus>, <https://github.com/prometheus/alertmanager>, <https://github.com/prometheus/pushgateway>, <https://github.com/grafana/grafana>, <https://github.com/grafana/alloy>, <https://github.com/open-telemetry/opentelemetry-python>, <https://github.com/open-telemetry/opentelemetry-collector>, <https://github.com/deepchecks/deepchecks>
- [S23] GitHub: <https://github.com/SeldonIO/alibi-detect>
- [S24] NannyML issues #450 and #451: <https://github.com/NannyML/nannyml/issues/450>, <https://github.com/NannyML/nannyml/issues/451>
- [S25] AWS, SageMaker Model Monitor data capture and data quality: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture.html>, <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html>
- [S26] Müller, Abdelaal, Stjelja, "Open-Source Drift Detection Tools in Action: Insights from Two Use Cases" (2024): <https://arxiv.org/abs/2404.18673>
- [S27] Evidently docs, migration guide: <https://docs.evidentlyai.com/faq/migration.md>; Evidently GitHub releases [S8]
- [S28] NannyML docs on performance estimation: <https://github.com/NannyML/nannyml/blob/v0.6.0/docs/how_it_works/performance_estimation.rst> (old v0.6.0 page, not the current docs)
- [S29] "Estimating Model Performance Under Covariate Shift Without Labels" (NannyML authors): <https://arxiv.org/abs/2401.08348>
- [S30] Google Cloud Run, billing settings: <https://docs.cloud.google.com/run/docs/configuring/billing-settings>
- [S31] Google Cloud Run, Prometheus sidecar: <https://docs.cloud.google.com/run/docs/monitoring-managed-prometheus-sidecar>
- [S32] Google Cloud Run, OTLP collector sidecar tutorial: <https://docs.cloud.google.com/run/docs/tutorials/custom-metrics-opentelemetry-sidecar>
- [S33] Prometheus-users threads on scraping behind load balancers (2020-2022, community answers, not Cloud Run-specific): <https://groups.google.com/g/prometheus-users/c/mrVa9fnZPc8>, <https://groups.google.com/g/prometheus-users/c/B_zPPx-lnyA/m/m8id10qwBAAJ>
- [S34] Prometheus docs, "When to use the Pushgateway": <https://prometheus.io/docs/practices/pushing/>
- [S35] Pushgateway README: <https://github.com/prometheus/pushgateway>
