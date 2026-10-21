# Research: choose the data-engineering tool stack

Ticket: [#649](https://github.com/SkafteNicki/dtu_mlops/issues/649), part of map
[#647](https://github.com/SkafteNicki/dtu_mlops/issues/647). Builds on
[#648](https://github.com/SkafteNicki/dtu_mlops/issues/648) (version the pipeline, not the data: append-only `raw/`,
deterministic versioned ETL, lineage via W&B reference artifacts on a versioned GCS bucket). Researched 2026-10-21.

## Recommendation

| Slot | Pick | Runner-up | Not recommended |
| --- | --- | --- | --- |
| Orchestrator | **Prefect 3** (OSS, no server needed to run) | **Dagster** (asset model, heavier) | MageAI, ZenML, Airflow/Kestra/Flyte |
| Labeling | **Label Studio Community** (`uv add label-studio`, local, GCS storage) | **CVAT** (docker compose or CVAT Online free tier) | Argilla |
| Validation in the pipeline | **pandera** | Great Expectations (GX Core) | n/a |
| Cloud execution | **Cloud Run job** running the same `uv run` entrypoint (+ Cloud Scheduler if a schedule is wanted) | Prefect Cloud Hobby + `cloud-run` work pool | Self-hosted Prefect/Dagster server on GCP |

In one line: write the pipeline as plain Python functions with `@task`/`@flow` decorators, run it locally with
`uv run`, and in the cloud as a Cloud Run job, with no orchestrator server anywhere. Pandera checks sit inside the
validate task, and W&B reference artifacts are logged at the end. Students run Label Studio on their laptop against
the GCS bucket.

Why this fits the loop: the Day 4 pipeline is just a script that Prefect decorates. In week 3 it reruns unchanged on a
bigger `raw/`, and the only new piece is a labeling step that writes new objects to the bucket.

## Orchestrator

### Activity check (GitHub GraphQL API, default branch, queried 2026-10-21)

| Repo | License | Stars | Commits last 90 d | Commits last 180 d | Latest releases |
| --- | --- | --- | --- | --- | --- |
| mage-ai/mage-ai | Apache-2.0 | 8.8k | 17 | 35 | 0.9.80 (2026-10-09), 0.9.79 (2026-01-21), 0.9.78 (2025-09-18) |
| PrefectHQ/prefect | Apache-2.0 | 24k | 437 | 1000+ | 3.9.0 (2026-10-21), weekly dev releases |
| dagster-io/dagster | Apache-2.0 | 16k | 342 | 1163 | 1.13.26 (2026-10-09), roughly weekly |
| zenml-io/zenml | Apache-2.0 | 5.6k | 140 | 416 | 0.98.0 (2026-10-14) |
| apache/airflow | Apache-2.0 | 47k | 2525 | 4717 | 3.3.2 (2026-09-17) |
| kestra-io/kestra | Apache-2.0 | 30k | 1866 | 3012 | v2.0.7 (2026-10-20) |
| flyteorg/flyte | Apache-2.0 | 7.7k | 338 | 569 | v2.0.52 (2026-10-21) |

Source: `gh api graphql` on `defaultBranchRef.target.history(since:)` and `releases`.

### MageAI: maintained, but only just

- **About 1 OSS release in 9 months** (0.9.79 in Jan 2026, then 0.9.80 in Oct 2026), and 17 commits in the last 90 days.
  Most recent commits are security hardening, dependency pins, and *"Sync Mage Pro docs with OSS merges"*, with almost no
  feature work. Most commits come from one maintainer (`wangxiaoyou1993`).
  (`gh api repos/mage-ai/mage-ai/commits`)
- The README points to **Mage Pro** for "enterprise orchestration, collaboration, and AI-powered workflows" and says
  nothing about the OSS roadmap. [README](https://github.com/mage-ai/mage-ai/blob/master/README.md)
- Docs say "We recommend using Docker to get started". pip works, but the main experience is a notebook-style web
  editor where pipelines are made of UI-managed "blocks" stored in a Mage project folder.
  [Mage setup](https://docs.mage.ai/getting-started/setup)
- **uv test (run here):** `uv run --with mage-ai` installs on Python 3.12 (182 packages). It **fails on Python 3.13**
  because mage-ai 0.9.80 pins `aiohttp==3.10.0`, which has no 3.13 wheel and fails to build from source. The course
  `pyproject.toml` lists 3.11 to 3.14.
- Verdict: the OSS project gets little investment while the vendor focuses on Pro. It is UI/Docker-first, does not fit
  "plain Python + `uv`", and breaks on current Python. Drop it.

### Prefect 3 (recommended)

- Flows and tasks are decorators on ordinary functions, so the Day 4 `data.py` stays a normal script.
- **Runs with no server to manage.** Tested here with `uv run --no-project --python 3.12 --with prefect` (3.9.0) on a
  3-task flow. Prefect logged `Starting temporary server on http://127.0.0.1:…`, ran the flow, then
  `Stopping temporary server`. The first run on a fresh `PREFECT_HOME` also printed a noisy but harmless
  `sqlite3.OperationalError: database is locked` traceback while still completing. The second run was clean.
  - Discrepancy: the settings reference lists `server.ephemeral.enabled` (`PREFECT_SERVER_ALLOW_EPHEMERAL_MODE`) with
    default `False`, but the observed behaviour is that an ephemeral server starts.
    [Settings reference](https://docs.prefect.io/v3/develop/settings-ref). Trust the observed behaviour and recheck
    when writing the module.
- UI when wanted: `prefect server start` (localhost:4200, SQLite by default, "requires essentially no setup").
  Scheduling locally: `flow.serve(name=..., cron=...)`.
  [Quickstart](https://docs.prefect.io/v3/get-started/quickstart), [Server](https://docs.prefect.io/v3/concepts/server)
- GCP paths, from lightest to heaviest:
  1. Containerize the script and run it as a **Cloud Run job**, which the ephemeral server handles. No Prefect
     infrastructure is involved. Add Cloud Scheduler for a cron trigger (`gcloud scheduler jobs create http … --uri=…/jobs/JOB:run`).
     [Cloud Run jobs on a schedule](https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule). I did not test the
     ephemeral server inside a Cloud Run container.
  2. **Prefect Cloud Hobby**, which is free with 2 seats, 1 workspace, 5 deployments, 7-day run retention, and 500
     min/month serverless compute. [Pricing](https://www.prefect.io/pricing)
  3. A `cloud-run` work pool plus a worker deployed as a Cloud Run service. This needs Prefect Cloud *or* a
     self-hosted server reachable by the worker, and the course should not ask students to run that.
     [GCP worker guide](https://docs.prefect.io/integrations/prefect-gcp/gcp-worker-guide)
- Most active of the candidates (437 commits/90 d, weekly releases). `requires_python >=3.11,<3.16` (PyPI).
- Already in the repo: `pyproject.toml` on `dev` pins `prefect<=3.5`, and `s10_extra/orchestration.md` is an unfinished
  Prefect stub. It imports `from prefect import task, Flow`, which is Prefect 1 API, so it needs rewriting either way.
- W&B: no first-party integration is needed. Call `wandb.init` / `log_artifact` inside the flow.

### Dagster (runner-up)

- Install and scaffold are uv-native (`uvx create-dagster@latest project …`, `dg dev` on localhost:3000,
  `dg launch --assets "*"`). [Quickstart](https://dagster.io/docs/getting-started/quickstart)
- The software-defined **asset** model (raw → validated → processed as assets) matches the #648 mental model and has a
  first-party `dagster-wandb` integration (0.29.26 on PyPI).
- Costs: more concepts to learn in one day (assets, definitions, resources, the `dg` project layout), and it
  restructures `data.py` into a Dagster project instead of decorating it. Scheduling also needs long-running services:
  webserver, a **daemon** for "schedules, sensors, and run queuing", and code location servers.
  [OSS deployment architecture](https://dagster.io/docs/deployment/oss/oss-deployment-architecture). Running a job as a
  one-shot Cloud Run job is possible, but that bypasses most of what Dagster adds.
- Pick Dagster instead if the course wants to teach *asset lineage inside the orchestrator* rather than in W&B.

### Others

- **ZenML:** runs locally without a server ("By default, ZenML runs without a server connected to a local database").
  However, the GCP orchestrators are "only meant to be used within the context of a remote ZenML deployment scenario"
  and need a remote server, artifact store, and container registry. Its artifact versioning and lineage would also
  duplicate W&B. [Installation](https://docs.zenml.io/getting-started/installation),
  [Vertex orchestrator](https://docs.zenml.io/stacks/stack-components/orchestrators/vertex)
- **Plain invoke + Cloud Run jobs + Cloud Scheduler:** the zero-dependency baseline, and what Prefect reduces to on
  GCP in the recommended setup. On its own it adds no retries, caching, run history, or task graph, so students learn
  nothing about orchestration. Mention it as the "you may not need an orchestrator" sidebar.
- **Airflow, Kestra, Flyte:** very active, but each needs a scheduler/server stack (Airflow), a Java/YAML-first server
  (Kestra), or Kubernetes (Flyte). They are too much ops for one day. Mention them only as industry context.

## Labeling

### Label Studio Community (recommended)

- Apache-2.0, very active (856 commits/90 d, 1.23.2 on 2026-09-29). `pip install label-studio`, then `label-studio`
  serves on :8080 with SQLite storage. [Install](https://labelstud.io/guide/install)
- **uv test (run here):** `uvx --python 3.13 --from label-studio` installed 158 packages without trouble (1.23.2).
  PyPI `requires_python >=3.10,<4`. It is plain Python with no Docker.
- **GCS fits the bucket layout from #648**, and the basic GCS connector is in Community edition:
  - *Source storage* points at the `raw/` prefix of phone photos. Label Studio "does not automatically sync" (you click
    Sync or use the API). By default the browser gets presigned URLs, which need a credential that can sign, typically a
    service-account key. *Proxy mode* (presigned off) streams through the backend and only needs
    `storage.objects.get/list`. [Cloud storage](https://labelstud.io/guide/storage)
  - *Target storage* writes "a JSON-formatted export of each annotation" to a bucket prefix on submit or update.
  - **Caveat from source code:** the object key is `annotation.id` (or `task.id` + `.json` behind a feature flag), and
    `blob.upload_from_string` **overwrites** it on update.
    ([`gcs/models.py`](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/io_storages/gcs/models.py),
    [`base_models.py` `get_key`](https://github.com/HumanSignal/label-studio/blob/develop/label_studio/io_storages/base_models.py)).
    So the target prefix is *mutable*, which breaks the append-only rule unless handled. Two options: (a) point target
    storage at `labels/` and rely on GCS Object Versioning + the W&B reference to pin generations, which is exactly the
    #648 mechanism for `processed/`; or (b) at the end of a labeling session, export a snapshot via the SDK
    (`label-studio-sdk`, active, 2.1.2) into `raw/labels/<batch-id>/`. Option (b) keeps `raw/` strictly append-only and
    makes "labels are just another raw input" literal. I recommend (b).
- Week-3 ops: students run Label Studio **locally**, so nothing needs deploying, and the bucket is the only shared state.
  An ML backend for pre-annotations exists but is optional.

### CVAT (runner-up)

- MIT, very active (v2.80.0 on 2026-10-21). Strong for detection and segmentation in images and video.
- Self-hosting is "clone the repository and `docker compose up -d`" (multi-container), and "Google Chrome is the only
  browser that is supported". [Installation](https://docs.cvat.ai/docs/administration/community/basics/installation/)
- CVAT Online free plan: 1 project, 3 tasks, 1 GB, 1 cloud storage connector.
  [Pricing](https://www.cvat.ai/pricing/cvat-online). That is enough for one student project, but it is a SaaS
  account rather than OSS-on-your-laptop.
- Pick it instead if the course dataset needs boxes or masks and Label Studio's image tools feel limiting.

### Argilla (avoid)

- The README now says "The original authors have moved on … we won't be adding new features going forward" (bug fixes
  only). The last commit on `develop` is 2025-08-05, and the last release is v2.8.0 (2025-03-11). It is also NLP/LLM
  focused. [README](https://github.com/argilla-io/argilla/blob/develop/README.md)

## Validation inside the pipeline

- **pandera (recommended):** MIT, active (0.34.1, 76 commits/90 d). A schema is a few lines of Python
  (`DataFrameSchema` or the pydantic-like `DataFrameModel`) and `schema.validate(df)` raises on failure. It covers
  pandas, polars, pyspark, ibis, and others, plus `check_input`/`check_output`/`check_types` decorators that drop into a
  Prefect task. Install: `uv add 'pandera[pandas]'`. [Docs](https://pandera.readthedocs.io/en/stable/)
- **Great Expectations / GX Core (runner-up):** active (1.24.0; the repo now lives at `fivetran/great_expectations`).
  It has about 12 concepts (Data Context, Data Source, Data Asset, Batch Definition, Expectation Suite, Validation
  Definition, Checkpoint, Actions, Data Docs, …), which is too much for a sub-part of one day. It is the industry
  reference, so mention it. [GX overview](https://docs.greatexpectations.io/docs/core/introduction/gx_overview)
- **Image-data note:** pandera validates *tables*. For an image dataset, the validate step should build a
  **manifest DataFrame** (path, sha256, width, height, channels, label, source batch) and validate that, e.g. allowed
  labels, image size bounds, no duplicate hashes, and every image has a label. Pixel-level checks stay as plain Python
  asserts. A failed check means no artifact gets logged (#648).

## How the stack composes (Day 4 → week 3)

```text
raw/<batch>/ (append-only; Day 4: initial dataset; week 3: phone photos from the app + raw/labels/<batch>/ from Label Studio)
   └─ Prefect flow  (uv run src/<pkg>/data.py)   # same code locally and as a Cloud Run job
        ingest  → build manifest of all raw batches
        validate → pandera schema on the manifest   (fail → stop, nothing logged)
        process  → deterministic, seeded transform → processed/
        log      → W&B reference artifacts raw-data / processed-data (+ git SHA in run.config)
```

Week 3 adds only labeling (Label Studio → `raw/labels/<batch>/`) and new `raw/` batches. The flow itself is unchanged.

## Risks and open questions

- **Prefect ephemeral-server behaviour** goes against the settings doc default. Re-verify on the pinned version, and
  pin `prefect` in the course `pyproject.toml` (currently `<=3.5`; latest is 3.9.0). The first-run SQLite lock
  traceback will confuse students, so either warn them or set up `prefect server start` from the start.
- **Ephemeral server in Cloud Run** was not tested. If it causes trouble, set `PREFECT_API_URL` to Prefect Cloud Hobby
  or call the underlying functions directly (`flow.fn()`).
- **Label Studio presigned URLs** need a signing credential. User ADC from `gcloud auth application-default login`
  usually cannot sign, so the module should either use proxy mode or have students create a service-account key. Decide
  when writing the module.
- **Labeling task type** depends on the dataset choice (classification vs detection). That choice could flip Label
  Studio vs CVAT.
- Whether to show the Prefect UI on Day 4 (`prefect server start`) or keep it for an optional extra.
