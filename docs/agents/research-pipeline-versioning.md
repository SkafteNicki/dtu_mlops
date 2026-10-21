# Research: how do modern ML pipelines version data?

Ticket: [#648](https://github.com/SkafteNicki/dtu_mlops/issues/648), part of map
[#647](https://github.com/SkafteNicki/dtu_mlops/issues/647). Researched 2026-10-21.

## Answer in one paragraph

Mostly yes, with one caveat. Data engineering's established practice is **pipeline versioning**: raw inputs are kept
immutable and append-only, transformations are pure and deterministic code under git, and every derived dataset is
something you can recompute from (raw + code version). Medallion bronze/silver/gold is the vendor name for that layout.
DVC-style snapshotting of whole datasets is not how data teams version data. The caveat is that ML practice still
**materializes and pins the outputs** at the boundary where training consumes them (Sculley et al.'s "versioned copy of
a signal"), because recomputing is slow, can be non-deterministic, and depends on raw data still existing. In practice,
**model↔data lineage comes from a metadata store or experiment tracker** that records which code version produced which
data artifact and which run consumed it, not from the storage layer. For this course the smallest teachable version is:
**an append-only raw prefix in a versioned GCS bucket + a deterministic `uv run` pipeline script under git + W&B
reference artifacts (`add_reference` / `use_artifact`) to pin and link data to runs.** No table format is needed.

## Findings

### 1. Pipeline versioning is the data-engineering norm

- **Immutable raw + pure tasks lets you recompute everything.** Maxime Beauchemin (creator of Airflow and Superset) in
  "Functional Data Engineering: a modern paradigm for batch data processing" argues for pure, idempotent tasks that
  overwrite whole immutable partitions instead of running row-level UPDATE/DELETE, and for a *persistent* staging area
  that keeps all source data indefinitely. With both in place, the warehouse can in principle be recomputed from scratch
  to the same state, and derived tables can be backfilled at will.
  [Medium essay](https://maximebeauchemin.medium.com/functional-data-engineering-a-modern-paradigm-for-batch-data-processing-2327ec32c42a)
  (the page blocked automated fetch, so this was confirmed via search summaries;
  [talk page](https://www.aicouncil.com/functional-data-engineering-a-set-of-best-practices)).
- **Medallion layout is the same idea with names.** Databricks: bronze "contains and maintains the raw state of the data
  source in its original formats", "is appended incrementally and grows over time", and "enables reprocessing and auditing
  by retaining all historical data". Silver is "validated, cleaned, and enriched" (cleansing, dedup, normalization). Gold
  is aggregated, consumption-ready data. Bronze "serves as the single source of truth".
  [Databricks: medallion architecture](https://docs.databricks.com/aws/en/lakehouse/medallion)
  - Note: Databricks says *append-only*. It does not literally say *immutable*. Immutability is a discipline you add
    (e.g. never overwrite an object under `raw/`).

### 2. But ML pins materialized data at the training boundary

- Sculley et al., "Hidden Technical Debt in Machine Learning Systems" (NeurIPS 2015), on unstable data dependencies:
  "One common mitigation strategy for unstable data dependencies is to create a versioned copy of a given signal ...
  Versioning carries its own costs, however, such as potential staleness and the cost to maintain multiple versions of the
  same signal over time."
  [Paper PDF](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- So the realistic model is a **hybrid**. Raw is immutable and the pipeline is versioned, *and* the specific processed
  dataset a model trained on gets an identifier you can refer to later. This is because recompute-from-raw breaks when:
  (a) the transform is not deterministic (random splits without seeds, external API calls, library upgrades), (b) raw is
  deleted under retention/GDPR, or (c) recomputing is expensive. That ID does not have to be a DVC snapshot. A pointer to
  immutable objects is enough (section 4).

### 3. Lineage comes from metadata, not storage

- Google Cloud's MLOps reference architecture (level 1+) records, for each pipeline execution: "the pipeline and component
  versions that were executed", the parameter arguments, "the pointers to the artifacts produced by each step of the
  pipeline", and "a pointer to the previous trained model if you need to roll back". This is "to help with data and
  artifacts lineage, reproducibility, and comparisons."
  [Google Cloud: MLOps continuous delivery and automation pipelines](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- W&B Artifacts gives the same thing in the tool the course already teaches. A run declares inputs with
  `run.use_artifact()` and outputs with `run.log_artifact()`. The UI Lineage tab shows the run↔artifact graph, and the SDK
  traverses it with `logged_by()` / `used_by()`. `use_artifact` records the dependency without downloading.
  [W&B: explore and traverse an artifact graph](https://docs.wandb.ai/guides/artifacts/explore-and-traverse-an-artifact-graph),
  [W&B Registry lineage](https://docs.wandb.ai/guides/registry/lineage)

### 4. W&B reference artifacts + GCS object versioning give data pinning without copying data

- Reference artifacts "track and use files saved outside of" W&B. Only metadata is logged (ETag, size, MD5 checksums, and
  the object **version ID if bucket versioning is enabled**). "The files themselves never leave your system." GCS buckets
  are a supported backend.
- On `download()`, if versioning is on, W&B "retrieves the object version that corresponds to the state of the file at
  the time an artifact was logged". If the file was overwritten without versioning, download errors. W&B explicitly
  recommends enabling Object Versioning. Default limit: 10,000 objects per prefix (`max_objects=` to raise it).
  [W&B: track external files](https://docs.coreweave.com/models/artifacts/track-external-files) (docs.wandb.ai now
  redirects to docs.coreweave.com)
- GCS Object Versioning keeps replaced/deleted objects as noncurrent versions, each identified by a **generation number**
  (addressable as `gs://bucket/obj#<generation>`). There is no default version limit, and each noncurrent version is billed
  as storage, so use lifecycle rules to cap it. Google recommends **soft delete** (on by default, 7-day retention) rather
  than versioning for protection against accidental deletion.
  [GCS object versioning](https://docs.cloud.google.com/storage/docs/object-versioning),
  [GCS soft delete](https://docs.cloud.google.com/storage/docs/soft-delete)
- The current S6 module already tells students to enable Object versioning on their bucket
  (`s6_the_cloud/using_the_cloud.md`, the bucket-creation exercise), though only to support DVC's `version_aware` remote.
  The same bucket setting carries over unchanged.

### 5. Table formats are optional for the course, and their time travel is not long-term versioning

- **Delta Lake:** time travel is bounded by `delta.logRetentionDuration` (default 30 days) and
  `delta.deletedFileRetentionDuration` (default 7 days), plus `VACUUM`. The docs list audit/debug/rollback as uses, not
  archival. [Delta Lake: table batch reads and writes](https://docs.delta.io/latest/delta-batch.html)
- **Iceberg:** "Expiring old snapshots removes them from metadata, so they are no longer available for time travel
  queries". Regular expiry is recommended. [Iceberg maintenance](https://iceberg.apache.org/docs/latest/maintenance/)
- **DuckLake:** Parquet files on object storage (GCS supported) with metadata in a SQL database (Postgres or a DuckDB
  file). Snapshots and time travel come built in. It is the lightest-weight table format, and the easiest to adopt
  if the course wants one. [DuckLake manifesto](https://ducklake.select/manifesto/)
- Takeaway: table formats help with *mutable tables, concurrent writers, and SQL over the lake*. A course dataset of
  image/tabular files written by one pipeline needs none of that. Snapshot expiry also means you would still need to pin
  a snapshot/tag and record it in the tracker, which is the same lineage step as with plain objects. Present table formats
  as "what you'd reach for at scale", not as the mechanism.
- Dedicated data-versioning tools still exist (DVC, lakeFS "git for data", Delta/Iceberg snapshots). Industry hasn't
  abandoned data versioning. It is one implementation of "give the training data an immutable ID", used when the
  storage layer cannot provide immutability itself. (I did not fetch lakeFS docs successfully. Treat this as a category
  mention.)

## Recommended mental model for the course

> **Data is a function of (immutable raw inputs, versioned pipeline code, parameters). Version the inputs by never
> mutating them, version the function with git, and record the triple plus the output's ID in the experiment
> tracker.**

Then the lifecycle loop needs no new mechanism. New data and labels arrive as new objects under `raw/`, the same pipeline
reruns, a new processed artifact version appears, training consumes it, and W&B lineage links model vN to
`processed:vN`, which links to the pipeline's git SHA and the raw prefix. That is the map's "second pass no harder than
the first".

## Concrete mechanism (smallest teachable version)

1. **Bucket layout** (one GCS bucket, Object Versioning on, lifecycle rule capping noncurrent versions):
   - `raw/<batch-id or date>/...`: append-only. Write new batches only, never overwrite. Collected production data
     and new labels land here as new batches.
   - `processed/`: the pipeline output (silver). Overwriting is allowed because versioning + W&B pin the generation.
   - Optional `gold/` or "features" naming if the course wants the medallion vocabulary. Bronze/silver is enough for the
     dataset sizes involved.
2. **Pipeline** `data.py` (already the course's pattern): deterministic (seeded splits, pinned deps via `uv.lock`),
   reads `raw/`, writes `processed/`, and inside a W&B run:
   - `run.config` records the git SHA and params,
   - `add_reference("gs://bucket/raw/")` → log as artifact `raw-data` (input),
   - `add_reference("gs://bucket/processed/")` → log as artifact `processed-data` (output).
3. **Training** calls `run.use_artifact("processed-data:latest")` (or a pinned `:vN` / alias). The model artifact is logged
   from the same run, so the lineage graph reads raw → pipeline run → processed → train run → model → registry.
4. **Validation** (in scope per map) sits as checks inside step 2. A failed check means no artifact is logged.

What students learn: immutability, determinism, and lineage as three separate properties, each with a concrete
enforcement point (bucket convention, code + lockfile, tracker).

## Trade-offs and risks

- **Determinism is on the student.** Recompute-from-raw only holds if the pipeline is pure. Unseeded splits and unpinned
  deps silently break it. The W&B reference (which pins exact object generations) is the safety net, so teach both.
- **Append-only is a convention, not enforced.** GCS can enforce it with retention policies or IAM (write without
  overwrite), but that adds friction. Object Versioning + soft delete makes mistakes recoverable.
- **Cost:** noncurrent versions are billed. Add a lifecycle rule (e.g. keep N noncurrent versions or delete after X days),
  and note that this caps how far back you can reproduce.
- **W&B coupling:** lineage lives in W&B. Losing access to the W&B project means losing lineage. The git SHA
  in `run.config` plus the immutable raw prefix still let you recompute without it.
- **10,000-object default** for prefix references. Image datasets may exceed it, so pass `max_objects` or reference a
  sharded/archived form.
- **Local-only students** (no GCS yet on Day 4): the same pattern works with local `data/raw/` + `wandb.Artifact.add_dir`
  (copies data into W&B) or `add_reference("file://...")`, then moves to `gs://` in S6 unchanged in shape.
- **What is lost vs DVC:** `dvc pull` as a one-command "get me exactly this data" and git-native `.dvc` pointer files.
  `artifact.download()` replaces the former, and the W&B artifact version replaces the latter, but the pointer now
  lives in W&B rather than in git.

## Open questions for the human

- Should Day 4 (no cloud yet) use local `file://` references, or should the bucket move earlier?
- Use the medallion names (bronze/silver/gold) or plain `raw/processed`? The names help students map to industry
  vocabulary, but three layers is more than the dataset needs.
- Mention DuckLake/Delta/Iceberg as a "beyond the course" box only, or include a small optional exercise?
