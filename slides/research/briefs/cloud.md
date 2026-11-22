# Review brief: cloud (`slides/day7_cloud.pdf`)

- **Modules compared:** `s6_the_cloud/` (`README.md`, `cloud_setup.md`, `using_the_cloud.md`, `exercise_files/`) @ `main`
    e5f4e1743f78b096cc15c67655cabdb1411a8dbe. Overlaps checked in `s7_deployment/cloud_deployment.md` and
    `pages/timeplan.md`.
- **Other inputs:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`,
    `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, web sources (accessed 2026-11-22). Cross-check:
    the pilot Slidev draft on local branch `slidev/pilot-day7-cloud` (`slides/pilot/day7_cloud/slides.md`).
- **Pages:** 18 PDF pages → 18 slides. There are no click builds to merge and no Gamma "Error uploading image"
    placeholders; the pilot draft found the same. Extracted with `extract_gamma_pdf.py` from
    `research/gamma-pdf-extraction`, and every page render was checked by eye. Glyph repairs follow the pilot: p. 4
    "orchestration4each" is an em dash, p. 16 "is&hard" renders as "is...hard", and the p. 16 bullets are ⚠️ emojis.

## Deck outline (narrative order to keep)

1. Title: page 1
2. What the cloud is and why it exists (infinity framing, technical debt, why use it, the Bezos mandate, pros and cons,
    grandma meme, data centres): pages 2–8
3. Vendors and the service catalogue: pages 9–10
4. Class exercise "what do you need?": page 11
5. The GCP services that matter for ML, and the "want to do vs have to do" stack: pages 12–13
6. Access and interaction (IAM policy; console, gcloud, APIs): pages 14–15
7. Warning, then a live look at the console: pages 16–17
8. Meme of the day: page 18

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "It is to data scientists what advance calculus was to mathematicians in the early 17th century: The power of
        infinity" (p. 2). CLOUD-01 proposes a change.
    - "The cloud provides services to help lower the technical debt" (p. 3)
    - "It is this mindset of providing a lot of services that basically makes up cloud" (p. 5)
    - "We are going to take a look at GCP." / "(because they gave me credits)" (p. 9)
    - "Stuff you want to do VS stuff you have to do" (p. 13)
    - "Working in the cloud is...hard." (p. 16)
    - "Permissions, permissions," (p. 16)
    - "The only way to learn is to use it. If you can, start simple and then scale up" (p. 16)
- Voice markers:
    - The six-rule Bezos mandate with rule 5 in red, read as the origin story of the cloud (p. 5). CLOUD-03 changes the
        attribution, not the list.
    - The grandma "renting computers" meme as the answer to "What is the cloud?" (p. 7)
    - The open class question before the services are shown (p. 11)
    - Cake ("stuff you want to do") vs broccoli ("stuff you have to do") on the landing-zone pyramid (p. 13)
    - The ⚠️ warning list (p. 16) followed by a live console walk-through (p. 17)
    - Closing "Meme of the day" (p. 18)

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): the bucket becomes
    the versioned home of the pipeline's append-only `raw/` and `processed/` data, with lineage recorded as W&B reference
    artifacts on `gs://` URIs. The deck shows the bucket only as an icon. → see CLOUD-14
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): on GCP the Prefect 3
    pipeline runs as a Cloud Run job (Cloud Scheduler optional), so "Run" on p. 12 now covers batch jobs as well as
    services. → see CLOUD-15
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    `using_the_cloud.md` drops its DVC steps for `gcloud storage cp` plus the Day 4 pipeline, switches `gsutil` to
    `gcloud storage`, and the GitHub Actions module gets Workload Identity Federation as a best-practice admonition.
    → see CLOUD-12, CLOUD-13, CLOUD-14
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651): Rock-Paper-Scissors
    replaces corrupted MNIST in the exercises. The deck names no dataset, so nothing changes on the slides.
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): W&B logging moves
    to Day 3, so it is taught before the cloud session. That is what lets CLOUD-14 assume W&B artifacts.

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased (e.g. CLOUD-01) -->

### CLOUD-01 · Calculus analogy has the wrong century

- **Category:** outdated
- **Priority:** must
- **Pages:** 2
- **Finding:** The keep-list quote places calculus in "the early 17th century" (and says "advance"). Newton and Leibniz
    developed it independently in the late 17th century: Newton from 1665, Leibniz in the 1670s, first published by
    Leibniz in 1684.
- **Proposal:** Keep the "power of infinity" framing and fix the century and the typo. This changes a keep-list item,
    so it needs the author's sign-off.
- **Evidence:** Encyclopaedia Britannica, "Calculus", <https://www.britannica.com/science/calculus-mathematics>
    (accessed 2026-11-22).

### CLOUD-02 · Cite Sculley et al. on the technical-debt slide

- **Category:** addition
- **Priority:** could
- **Pages:** 3
- **Finding:** The figure is Fig. 1 of Sculley et al. (2015), and the slide does not cite it. The paper's caption says
    "Only a small fraction of real-world ML systems is composed of the ML code"; the slide says "A fraction".
- **Proposal:** Redraw the figure with a citation and keep the paper's "small fraction" wording.
- **Evidence:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, p. 4, Figure 1 caption.

### CLOUD-03 · The Bezos mandate is Yegge's 2011 paraphrase

- **Category:** outdated
- **Priority:** must
- **Pages:** 5
- **Finding:** The slide signs the six rules "2002, Jeff Bezos" as if quoted. The only source is Steve Yegge's 2011
    Google+ "platforms rant", which introduces them with "His Big Mandate went something along these lines" and dates
    them "back around 2002 I think, plus or minus a year".
- **Proposal:** Keep the list and the red rule 5 (keep list). Re-attribute it as "Bezos's API mandate (c. 2002), as
    recalled by Steve Yegge (2011)".
- **Evidence:** Yegge, "Stevey's Google Platforms Rant", mirror at
    <https://courses.cs.washington.edu/courses/cse452/22wi/papers/yegge-platform-rant.html> (accessed 2026-11-22).

### CLOUD-04 · Market-share chart stops at Q2 2024

- **Category:** outdated
- **Priority:** should
- **Pages:** 9
- **Finding:** The Synergy chart runs from Q4 2018 to Q2 2024. Synergy's Q3 2026 release puts Amazon, Microsoft and
    Google at 26%, 19% and 16%, and says Microsoft and Google "continue to achieve substantially higher growth rates".
    The slide's text still holds; the chart is two years old.
- **Proposal:** Re-plot from Synergy's latest release with a citation and keep the two text lines.
- **Evidence:** Synergy Research Group, "Cloud Market Growth Rate Surges to Well Over 50% per Year; Q3 Revenues Reach
    $168 Billion", 2026-10-29,
    <https://www.srgresearch.com/articles/cloud-market-growth-rate-surges-to-well-over-50-per-year-q3-revenues-reach-168-billion>
    (accessed 2026-11-22).

### CLOUD-05 · No slide on credits, and the free trial changed

- **Category:** gap
- **Priority:** should
- **Pages:** 9, 18
- **Finding:** The module opens with claiming the $50 DTU coupon, the $300 trial for non-DTU students, and watching
    billing reports; the deck only jokes about credits. Since 2026-11-16 the $300 trial credit can't pay for "Gemini
    platform (formerly known as Gemini Enterprise Agent Platform, Vertex AI)" services, and a non-billable trial
    account can't add GPUs to VMs.
- **Proposal:** Add one practical slide: where DTU credits come from, what the public trial no longer covers (training
    jobs, GPUs), and where to watch spending. The module needs the same update.
- **Evidence:** `s6_the_cloud/cloud_setup.md`, "❔ Exercises" step 1 and 3; Google Cloud, "Free Google Cloud features
    and trial offer", <https://docs.cloud.google.com/free/docs/free-cloud-features> (accessed 2026-11-22).

### CLOUD-06 · Vertex AI has been renamed

- **Category:** outdated
- **Priority:** should
- **Pages:** 12
- **Finding:** The slide and the module say "Vertex". Google's training docs are now titled "Gemini Enterprise Agent
    Platform", and the free-trial page already calls it "Gemini platform (formerly known as Gemini Enterprise Agent
    Platform, Vertex AI)". The docs still create jobs with `gcloud ai custom-jobs create`.
- **Proposal:** Use the current product name with "formerly Vertex AI" and the same wording as the module. Check the
    name again just before the course, because it changed twice in 2026.
- **Evidence:** Google Cloud, "Serverless training overview", <https://docs.cloud.google.com/vertex-ai/docs/training/overview>;
    "Configure compute resources for ... serverless training",
    <https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/configure-compute>; free
    trial page as in CLOUD-05 (all accessed 2026-11-22). `s6_the_cloud/using_the_cloud.md`, "Training".

### CLOUD-07 · Cloud Functions is now Cloud Run functions

- **Category:** outdated
- **Priority:** should
- **Pages:** 12
- **Finding:** "Functions" on the services slide uses the old Cloud Functions identity. Google renamed it Cloud Run
    functions in August 2024 and merged it into Cloud Run. `s7_deployment/cloud_deployment.md` still says Cloud
    Functions.
- **Proposal:** Rename it on the slide, or fold it into "Run", since the deployment module covers both.
- **Evidence:** Google Cloud Blog, "Cloud Functions is now Cloud Run functions", 2024-08-22,
    <https://cloud.google.com/blog/products/serverless/google-cloud-functions-is-now-cloud-run-functions> (accessed
    2026-11-22); `s7_deployment/cloud_deployment.md`, "Cloud Functions".

### CLOUD-08 · The services slide shows icons but not the workflow

- **Category:** gap
- **Priority:** should
- **Pages:** 12
- **Finding:** Page 12 is seven icons with one-word labels. The module teaches how they connect: Cloud Build builds the
    image, Artifact Registry stores it, Compute Engine or a managed training job runs it, and the bucket holds the data.
- **Proposal:** Turn the slide into a small concept diagram of that flow, with each box pointing to its module section.
- **Evidence:** `s6_the_cloud/using_the_cloud.md`, "Compute", "Data storage", "Artifact registry", "Training".

### CLOUD-09 · Secret Manager is missing

- **Category:** gap
- **Priority:** should
- **Pages:** 12
- **Finding:** The module has a whole section on Secret Manager (for example, the W&B API key for training jobs), but
    the deck never mentions secrets.
- **Proposal:** Add Secret Manager to the services overview, ideally on the CLOUD-08 diagram next to the training job.
- **Evidence:** `s6_the_cloud/using_the_cloud.md`, "Secrets management".

### CLOUD-10 · No slide on how training actually runs

- **Category:** gap
- **Priority:** should
- **Pages:** 12
- **Finding:** The module contrasts two ways to train: a VM you start and log into, and a managed custom job that
    creates and tears down the VM for you, so you can scale horizontally and vertically. The deck has only the icon.
- **Proposal:** Add one slide comparing the two paths (VM vs managed job) and when to pick each.
- **Evidence:** `s6_the_cloud/using_the_cloud.md`, "Training"; Google Cloud, "Create a serverless training job",
    <https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/create-custom-job>
    (accessed 2026-11-22).

### CLOUD-11 · GPU quotas are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 14, 16
- **Finding:** The module spends a section on GPU quotas, which start at 0 or 1 for free and teaching accounts. Students
    have to request an increase and may wait a day. The deck's warning slide doesn't mention it.
- **Proposal:** Add quotas to the warning slide or the IAM slide: what a quota is, that you request GPUs early, and
    where the request page is.
- **Evidence:** `s6_the_cloud/cloud_setup.md`, "IAM and Quotas".

### CLOUD-12 · Service accounts and keys vs Workload Identity Federation

- **Category:** gap
- **Priority:** should
- **Pages:** 14
- **Finding:** The IAM diagram shows a service account, but the deck never says what one is for. The module creates one
    with a downloaded JSON key and has to tell students to use "No organization" because many organizations block key
    creation. Google says user-managed keys "can become a security risk" and recommends keyless options.
- **Proposal:** Add a slide on service accounts with least privilege, then show JSON keys (what the course uses) next to
    Workload Identity Federation (the best practice), matching the decision for the GitHub Actions module.
- **Evidence:** `s6_the_cloud/cloud_setup.md`, "Service accounts" and the "Create project under 'No organization'"
    warning; Google Cloud, "Best practices for managing service account keys",
    <https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys>; "Configure Workload
    Identity Federation with deployment pipelines",
    <https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines> (accessed
    2026-11-22); [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650).

### CLOUD-13 · gsutil is a legacy tool

- **Category:** outdated
- **Priority:** should
- **Pages:** 15
- **Finding:** The slide lists "the bq cli and gsutil cli". Google calls gsutil "a legacy Cloud Storage CLI and minimally
    maintained" and will drop it from the gcloud CLI after March 2027. The module is switching to `gcloud storage`.
- **Proposal:** Replace gsutil with `gcloud storage` on the slide and in the new terminal screenshot or code block.
- **Evidence:** Google Cloud, "gsutil tool", <https://docs.cloud.google.com/storage/docs/gsutil> (accessed 2026-11-22);
    [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650).

### CLOUD-14 · Show the bucket as the pipeline's versioned data store

- **Category:** addition
- **Priority:** should
- **Pages:** 12
- **Finding:** After the restructure, the bucket holds append-only `raw/` and `processed/` with Object Versioning, and
    the data pipeline logs W&B reference artifacts that point at `gs://` objects. The deck shows "Bucket" as an icon
    with no link to the data pipeline.
- **Proposal:** On the services diagram (CLOUD-08), show the bucket as the place the data pipeline reads and writes,
    with versioning switched on. Leave the details to the module.
- **Evidence:** [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648);
    `s6_the_cloud/using_the_cloud.md`, "Data storage".

### CLOUD-15 · Cloud Run jobs for the data pipeline

- **Category:** addition
- **Priority:** should
- **Pages:** 12
- **Finding:** The restructure runs the Prefect pipeline on GCP as a Cloud Run job, with Cloud Scheduler optional. The
    deck's "Run" means services only, which the deployment session covers.
- **Proposal:** Mention that Cloud Run has jobs (run to completion) as well as services, and point to where the course
    uses each.
- **Evidence:** [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649); Google
    Cloud, "Execute jobs", <https://docs.cloud.google.com/run/docs/execute/jobs> (accessed 2026-11-22).

### CLOUD-16 · Link "cost management" to things students can do

- **Category:** addition
- **Priority:** could
- **Pages:** 6
- **Finding:** "Cost Management" is listed as a challenge, but the deck doesn't say how to control costs. The module
    points to billing reports and the difference between stopping and suspending a VM.
- **Proposal:** Add a line or two to the warning slide: watch billing reports, and stop or delete idle VMs and disks.
- **Evidence:** `s6_the_cloud/cloud_setup.md`, "❔ Exercises" step 3; `s6_the_cloud/using_the_cloud.md`,
    "🧠 Knowledge check" question 1.

### CLOUD-17 · Name the AWS and Azure equivalents

- **Category:** addition
- **Priority:** could
- **Pages:** 9, 12
- **Finding:** The module's knowledge check maps each GCP service to AWS and Azure so students can read blog posts from
    other ecosystems. The deck only says "We are going to take a look at GCP".
- **Proposal:** Add the GCP/AWS/Azure names next to the services, either as a small table or as a second row of labels.
- **Evidence:** `s6_the_cloud/cloud_setup.md`, "🧠 Knowledge check" question 2.

### CLOUD-18 · Grandma meme: recreate, regenerate or swap

- **Category:** image
- **Priority:** should
- **Pages:** 7
- **Finding:** The two-panel meme uses a photo of a stand-up comedian and is the slide's only content, so it can't be
    copied from the export. It also carries the deck's "the cloud is renting computers" point.
- **Proposal:** Author's call: recreate the joke as text over a licensed or generated image, or swap in another meme
    that makes the same point.
- **Evidence:** [How should images be recreated, and what may we reuse from Gamma exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705)
    (memes are decided per deck).

### CLOUD-19 · Meme of the day: four GCP accounts for $300

- **Category:** image
- **Priority:** should
- **Pages:** 18
- **Finding:** The meme is a celebrity photo grid with a watermark (@AaronNBrock), so it can't be reused as is. Its
    "$300 of free credits" premise is still current, but the trial is only for people who have never signed up before,
    and the credit now excludes the training service (see CLOUD-05).
- **Proposal:** Author's call: keep the joke and recreate or swap the image. If the credits slide from CLOUD-05 is
    added, place the meme right after it.
- **Evidence:** Free trial page as in CLOUD-05 (accessed 2026-11-22); issue #705 as in CLOUD-18.

### CLOUD-20 · Back "why cloud" with the interview study

- **Category:** addition
- **Priority:** could
- **Pages:** 4, 9
- **Finding:** The reasons on p. 4 are given without a source. In Shankar et al., the engineers interviewed mostly used
    cloud storage and GPU-backed cloud compute (AWS and GCP) as their infrastructure layer. The same paper's "keeping
    GPUs warm" theme fits the quota and cost slides.
- **Proposal:** Cite the study on p. 4 or p. 9 as evidence from practice.
- **Evidence:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 14 ("Infrastructure Layer")
    and p. 12, §5.2.2 "Keeping GPUs Warm".

### Also noticed (no id, not decided)

- p. 15 mentions Infrastructure as Code and Terraform, which no module covers (drift). A mention is probably fine.
- p. 12 labels the registry just "Registry". Writing "Artifact Registry" in full avoids confusion with Container
    Registry, which was shut down on 2025-03-18
    (<https://docs.cloud.google.com/artifact-registry/docs/transition/transition-from-gcr>, accessed 2026-11-22).
- The module's knowledge check on choosing a region (latency, GPU availability, GDPR) has no counterpart in the deck.
- p. 1 title is "Day7 - Cloud" and dated January 2026. Topic-only naming would make it "Cloud".
- p. 16 "Permissions, permissions," ends on a comma. It may be meant that way (keep list), or a third "permissions" may
    be missing.
- Module, not deck: `using_the_cloud.md` says `pytorch-latest-*` image families are deprecated (around line 116) but
    still uses `pytorch-latest-cpu` and `pytorch-latest-gpu` in the "Training" exercises, and recommends `europe-west-1`
    where the commands use `europe-west1`.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | AI-style illustration of an exploding cloud with circuitry; mood only | decorative | replaced by theme (or regenerate, author's call) | | |
| 3 | Sculley et al. Fig. 1: boxes of ML system infrastructure around a tiny "ML Code" box | chart/paper figure | re-plot/redraw + citation | | CLOUD-02 |
| 5 | Demand over time with spikes against a "Max compute" line | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 7 | Two-panel meme: grandma says GCP is "renting computers"; "You just insulted my entire industry. But yes." | meme/joke | keep / regenerate / swap? | voice | CLOUD-18 |
| 8 | Data-centre aisle with lit server racks and a presenter (stage photo) | decorative (photo; no matching type) | replaced by theme, or a licensed stand-in photo | | |
| 9 | Synergy "Cloud Provider Market Share Trend", Q4 2018 to Q2 2024 | chart/paper figure | re-plot/redraw + citation | | CLOUD-04 |
| 10 | Google Cloud products map (developer cheat sheet treemap) | tool screenshot | fresh screenshot | | |
| 12 | Seven GCP product icons: Compute Engine, Cloud Storage, Cloud Build, Artifact Registry, Vertex AI, Cloud Functions, Cloud Run | tool logo | Iconify logo (unverified that Iconify has per-product GCP icons) | | CLOUD-06, CLOUD-07 |
| 13 | Google landing-zone pyramid (workloads, technology stack, landing zone) with cake and broccoli photos | concept diagram | Mermaid/PlantUML/Excalidraw (photos as emoji) | | |
| 14 | Google IAM policy chart: who, can do what, on which resource, under which conditions | concept diagram | Mermaid/PlantUML/Excalidraw | | CLOUD-12 |
| 15 | Three screenshots: Cloud Console welcome page, terminal running `gcloud version`, Compute Engine REST reference | tool screenshot | fresh screenshot (terminal as a code block) | | CLOUD-13 |
| 17 | Console dashboard with the old "Google Cloud Platform" branding and a status notice dated 2021-12-07 | tool screenshot | fresh screenshot, or rely on the live demo | | |
| 18 | "Me signing up for my 4th GCP account to get the $300 of free credits": 2×2 grid of one celebrity in disguises, watermark @AaronNBrock | meme/joke | keep / regenerate / swap? | voice | CLOUD-19 |
| 1–18 | Dark Gamma background art, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · meme/joke ·
decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · replaced by theme.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | CLOUD-02, p. 3 claim "A fraction of real world ML systems is composed of ML code" |
| Shankar, Garcia, Hellerstein, Parameswaran, *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | CLOUD-20, p. 4 "Why use cloud?"; CLOUD-11 GPU quotas |
| Yegge 2011, *Stevey's Google Platforms Rant* | <https://courses.cs.washington.edu/courses/cse452/22wi/papers/yegge-platform-rant.html> | CLOUD-03, p. 5 "2002, Jeff Bezos" |
| Encyclopaedia Britannica, *Calculus* | <https://www.britannica.com/science/calculus-mathematics> | CLOUD-01, p. 2 "early 17th century" |
| Synergy Research Group 2026-10-29, *Cloud Market Growth Rate Surges to Well Over 50% per Year; Q3 Revenues Reach $168 Billion* | <https://www.srgresearch.com/articles/cloud-market-growth-rate-surges-to-well-over-50-per-year-q3-revenues-reach-168-billion> | CLOUD-04, p. 9 "AWS are still the largest player ... the others are growing faster" |
| Google Cloud, *Free Google Cloud features and trial offer* | <https://docs.cloud.google.com/free/docs/free-cloud-features> | CLOUD-05, CLOUD-06, CLOUD-19, p. 9 and p. 18 credits |
| Google Cloud, *Serverless training overview* | <https://docs.cloud.google.com/vertex-ai/docs/training/overview> | CLOUD-06, CLOUD-10, p. 12 "Vertex" |
| Google Cloud, *Create a serverless training job* | <https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/training/create-custom-job> | CLOUD-10 |
| Google Cloud Blog 2024-08-22, *Cloud Functions is now Cloud Run functions* | <https://cloud.google.com/blog/products/serverless/google-cloud-functions-is-now-cloud-run-functions> | CLOUD-07, p. 12 "Functions" |
| Google Cloud, *Execute jobs* (Cloud Run) | <https://docs.cloud.google.com/run/docs/execute/jobs> | CLOUD-15, p. 12 "Run" |
| Google Cloud, *gsutil tool* | <https://docs.cloud.google.com/storage/docs/gsutil> | CLOUD-13, p. 15 "gsutil cli" |
| Google Cloud, *Best practices for managing service account keys* | <https://docs.cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys> | CLOUD-12, p. 14 service account |
| Google Cloud, *Configure Workload Identity Federation with deployment pipelines* | <https://docs.cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines> | CLOUD-12 |
| Google Cloud, *Transition from Container Registry* | <https://docs.cloud.google.com/artifact-registry/docs/transition/transition-from-gcr> | Also noticed, p. 12 "Registry" |
