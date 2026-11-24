# Review brief: reproducibility_and_software (`slides/day3_reproducibility_and_software.pdf`)

- **Modules compared:** `s3_reproducibility/` (`README.md`, `docker.md`, `config_files.md`, `exercise_files/`) @
    `main` d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b, plus `s4_debugging_and_logging/logging.md`, whose experiment
    logging part moves into this session. Overlaps checked in `s1_development_environment/package_manager.md`,
    `s2_organisation_and_version_control/code_structure.md`, `pages/timeplan.md`, and by text search in the day 1, 2,
    4 and 12 PDFs.
- **Other inputs:** `literature/` (Raff 2019, read in the main sections; Sculley et al. 2015 and Shankar et al.
    searched for configuration and reproducibility; `Docker_Cookbook.pdf` (2015) only searched for VMs and
    reproducibility, `deep_learning.pdf` only its front matter; neither is cited); web sources (accessed 2026-11-24)
- **Pages:** 26 PDF pages → 20 slides after merging click builds (pages 6=7, 20=21=22=23=24=25; each page of the
    second build has its own screenshot), 19 once the closing meme is dropped. No Gamma "Error uploading image"
    placeholders. Text was extracted and every page render was checked by eye. Extraction glyphs: the stray letters D,
    Z, U, Ì and ¦ are emojis (💡, 🚩, 💭, → and ⚠️), and p. 9 "<landmark=" is "“landmark”".

## Deck outline (narrative order to keep)

1. Title and agenda: pages 1–2
2. What reproducibility is, and why research and industry need it: pages 3–5
3. Trustworthy ML/AI: pages 6–7
4. The crisis, with two examples: pages 8–10
5. What research is doing about it, and reproducibility as a spectrum: pages 11–12
6. Nicki's hierarchy of reproducibility: page 13
7. Level 1, configs and Hydra: pages 14–15
8. Level 2, dependencies: page 16
9. Level 3, VMs and Docker: pages 17–19
10. A 6-step process for reproducible software: pages 20–25
11. Meme of the day: page 26

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "We are in a crises!" (p. 8). REPRODUCIBILITY_AND_SOFTWARE-24 fixes the language.
    - "Another example: We are only humans" (p. 10)
    - "Reproducibility is not binary" (p. 12)
    - "Nicki’s hierarchy of reproducibility of ML" (p. 13)
    - "Make sure that you document everything about your experiments" (p. 13)
    - "For python: Just use a package management system" (p. 16). REPRODUCIBILITY_AND_SOFTWARE-14 changes only the
        list under it.
    - "The easiest way for someone to reproduce your work, would be to just hand over your computer." / "Instead of
        doing this, lets hand over a virtual copy of our machine" (p. 17)
    - "VM is isolation of machines, while Containers is isolation of processes" (p. 18)
    - "A 6-step process for reproducible software" (pp. 20–25)
- Voice markers:
    - The "only god and I knew how it worked" code comment opening the deck (p. 2). REPRODUCIBILITY_AND_SOFTWARE-25
        covers the image.
    - The red pyramid, with "easy to ship" traded against "highly reproducible", as the spine of the second half (pp.
        13–19). REPRODUCIBILITY_AND_SOFTWARE-01 changes only the numbers.
    - The diagnosis case with the ⚠️ (p. 7). REPRODUCIBILITY_AND_SOFTWARE-24 tightens its wording.
    - The three-column trade-off with ✅/⚠️: script parameters, argument parser, config files (p. 14)
    - Identical weights vs same mean ± std as the ML example (p. 12)
    - The 6-step table filling up one row per click, with an icon per step (pp. 20–25)
    - Emoji bullets: 💡 for ideas, 🚩 for reasons

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): W&B experiment
    logging moves into this session after Config Files, content unchanged; loguru goes to Good coding practice. → see
    REPRODUCIBILITY_AND_SOFTWARE-12 and REPRODUCIBILITY_AND_SOFTWARE-13
- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    the Docker module loses its DVC exercise and keeps a note that data is mounted or fetched at run time. → see
    REPRODUCIBILITY_AND_SOFTWARE-17 and REPRODUCIBILITY_AND_SOFTWARE-18
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): Day 4 builds on
    this session's W&B artifacts for lineage, so REPRODUCIBILITY_AND_SOFTWARE-12 should introduce artifacts but not
    data versioning.
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651):
    Rock-Paper-Scissors replaces corrupted MNIST, pending a real-photo check. The deck names no exercise dataset, so
    nothing changes.
- [Lock module numbering for the new layout](https://github.com/SkafteNicki/dtu_mlops/issues/656) is open, so the deck
    should cite modules by name.
- [Which decks does the January 2027 course need?](https://github.com/SkafteNicki/dtu_mlops/issues/723) is open and
    may decide where logging content lands. → affects REPRODUCIBILITY_AND_SOFTWARE-12
- [Agentic modules for the MLOps course](https://github.com/SkafteNicki/dtu_mlops/issues/618): this session gets an
    optional module on creating good, reproducible skills. Per [Update lecture slides for the January 2027
    course](https://github.com/SkafteNicki/dtu_mlops/issues/681) the lecture may carry agentic content but does not
    otherwise reference the A-modules. → see REPRODUCIBILITY_AND_SOFTWARE-19

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased -->

### REPRODUCIBILITY_AND_SOFTWARE-01 · The pyramid and the level slides count in opposite directions

- **Category:** wrong
- **Priority:** must
- **Pages:** 13, 14, 15, 16, 17, 18, 19
- **Finding:** The pyramid numbers Hardware 1, Software 2 and Code 3. The next slides call configs (code) "level 1"
    and VMs/Docker "level 3", so only level 2 matches.
- **Proposal:** Renumber the pyramid from the bottom (Code = 1, at the base) so that the level slides stay as they
    are. The keep-list pyramid keeps its shape and labels otherwise.
- **Evidence:** Pyramid and slide titles in the deck itself (pp. 13–19), page renders.

### REPRODUCIBILITY_AND_SOFTWARE-02 · The Hydra example doesn't run

- **Category:** wrong
- **Priority:** must
- **Pages:** 15
- **Finding:** The example uses `config_path="config.yaml"` and `cfg.pretty()`. With `hydra-core` 1.3.8 the first
    raises a `ValueError` (the name must go in `config_name`), and `pretty()` was removed from OmegaConf.
- **Proposal:** Replace it with a real code block that matches the module: `config_path`, `config_name`,
    `version_base` and `OmegaConf.to_yaml`.
- **Evidence:** Run locally with `hydra-core` 1.3.8 (current stable on PyPI, 2026-11-24): `ValueError` and
    `ConfigAttributeError: Missing key pretty`; `s3_reproducibility/config_files.md`, "Configure experiments"; Hydra,
    "A simple command-line application: config file",
    <https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/>; OmegaConf `NEWS.md`,
    <https://raw.githubusercontent.com/hydra-ecosystem/omegaconf/master/NEWS.md> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-03 · Papers with Code is gone

- **Category:** outdated
- **Priority:** must
- **Pages:** 11
- **Finding:** The slide links <https://paperswithcode.com/>, which now redirects to Hugging Face's trending papers
    page, so the link no longer shows the reproducibility work it is there for.
- **Proposal:** Drop the link and point to the ML Reproducibility Challenge instead (see
    REPRODUCIBILITY_AND_SOFTWARE-04).
- **Evidence:** HTTP fetch of <https://paperswithcode.com/>, which ends at <https://huggingface.co/papers/trending>
    (accessed 2026-11-24). That the site was shut down in July 2025 rests on secondary sources only: `unverified`.

### REPRODUCIBILITY_AND_SOFTWARE-04 · The Reproducibility Challenge screenshot is from 2021

- **Category:** outdated
- **Priority:** should
- **Pages:** 11
- **Finding:** The screenshot shows the Spring 2021 challenge. The challenge is now an official NeurIPS 2026 track
    with 16 papers, all first accepted at TMLR; its best paper award goes to a study that audits NeurIPS 2025
    contributions with agentic code auditing.
- **Proposal:** Update to MLRC 2026 and link its page. The award winner is a natural hook for
    REPRODUCIBILITY_AND_SOFTWARE-19.
- **Evidence:** MLRC 2026, <https://reproml.org/> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-05 · The checklist is now mandatory, with one name

- **Category:** outdated
- **Priority:** should
- **Pages:** 11
- **Finding:** "Checklist for conferences" shows an old text-only excerpt without a source. NeurIPS now desk-rejects
    papers that leave out its Paper Checklist.
- **Proposal:** Name it the NeurIPS Paper Checklist, say that it is mandatory, and quote two or three items as text
    rather than a tiny screenshot.
- **Evidence:** NeurIPS, "NeurIPS Paper Checklist Guidelines", <https://neurips.cc/public/guides/PaperChecklist>
    (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-06 · The crisis text is Nature's, without a citation

- **Category:** wrong
- **Priority:** must
- **Pages:** 8
- **Finding:** The sentence and the three explanations are almost word for word from Nature's "Challenges in
    irreproducible research" collection and are not credited. The chart is marked ©nature, also without a citation.
- **Proposal:** Credit the collection and cite Baker 2016 for the survey; redraw the chart.
- **Evidence:** Nature collection page, <https://www.nature.com/collections/prbfkwmwvz> ("There is growing alarm about
    results that cannot be reproduced. Explanations include increased levels of scrutiny, complexity of experiments
    and statistics, and pressures on researchers"); Baker 2016, <https://www.nature.com/articles/533452a> (accessed
    2026-11-24). The survey percentages sit behind the paywall: `unverified`.

### REPRODUCIBILITY_AND_SOFTWARE-07 · Amgen example: cite it and tighten the numbers

- **Category:** addition
- **Priority:** should
- **Pages:** 9
- **Finding:** The table and numbers are from Begley & Ellis 2012, uncited. The paper says 6 (11%) were confirmed and
    that the papers were deliberately chosen for describing something completely new. The text I could read says
    "scientists (including C.G.B.)" and has no team of "about 100": that number is `unverified`.
- **Proposal:** Cite the paper, write 11%, mention the selection, and drop or source "about 100 scientists".
- **Evidence:** Begley & Ellis 2012, <https://www.nature.com/articles/483531a> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-08 · Many-analysts example: cite it and say what happened

- **Category:** addition
- **Priority:** should
- **Pages:** 10
- **Finding:** The figure has no citation and the slide leaves out the result: 20 of 29 teams found a statistically
    significant positive effect and 9 did not, with odds ratios from 0.89 to 2.93.
- **Proposal:** Cite Silberzahn & Uhlmann 2015 (Nature) and the full 2018 study, state the split and redraw the
    figure.
- **Evidence:** Silberzahn & Uhlmann 2015, <https://www.nature.com/articles/526189a>; Silberzahn et al. 2018, abstract
    in Crossref, <https://doi.org/10.1177/2515245917747646> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-09 · The crisis section has no evidence from ML

- **Category:** gap
- **Priority:** should
- **Pages:** 8, 9, 10
- **Finding:** The evidence comes from a survey, oncology and social science. The module opens with Raff's attempt to
    reproduce 255 ML papers (162, or 63.5%, succeeded), and Kapoor & Narayanan find data leakage in 17 fields,
    affecting 329 papers (preprint).
- **Proposal:** Add one slide with Raff's numbers, and optionally the leakage survey. Keep the biology and football
    examples.
- **Evidence:** `s3_reproducibility/config_files.md`, introduction;
    `literature/A_step_toward_quantifying_independently_reproducible_machine_learning_research.pdf`, p. 2; Kapoor &
    Narayanan, <https://arxiv.org/abs/2207.07048> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-10 · Link hyperparameters to the evidence

- **Category:** gap
- **Priority:** should
- **Pages:** 14
- **Finding:** The slide calls hyperparameters the most notable subjective choice but cites nothing. In Raff,
    "Hyperparameters Specified" is significant (p = 8.45 × 10⁻⁶) while "Code Available" is not (p = 0.213).
- **Proposal:** Cite Raff on the slide. Mention the code result with care: the study deliberately never used the
    authors' code, so it doesn't show that sharing code is useless.
- **Evidence:** `s3_reproducibility/config_files.md`, introduction; Raff 2019 (file as in
    REPRODUCIBILITY_AND_SOFTWARE-09), Table 1 (p. 5) and abstract.

### REPRODUCIBILITY_AND_SOFTWARE-11 · Seeds and nondeterminism are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 12, 14
- **Finding:** The deck compares checkpoints weight by weight but never mentions seeds. The module makes the seed the
    hidden hyperparameter and has students test two runs. PyTorch adds that full reproducibility is not guaranteed
    across releases, commits or platforms, or between CPU and GPU, even with the same seed.
- **Proposal:** Add a seed slide or line next to the "same weights vs same conclusion" example, including what a seed
    does not fix.
- **Evidence:** `s3_reproducibility/config_files.md`, "❔ Exercises" steps 3 and 10; PyTorch 2.15, "Reproducibility",
    <https://docs.pytorch.org/docs/2.15/notes/randomness.html> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-12 · Experiment logging moves into this session

- **Category:** gap
- **Priority:** should
- **Pages:** 14, 15
- **Finding:** W&B experiment logging moves here, after Config Files, so that it forms one reproducibility story with
    Hydra. The deck says nothing about tracking runs, metrics or artifacts.
- **Proposal:** Add a short section after Hydra: what a tracked run records (config, git commit, command,
    environment), artifacts and the registry. The full run anatomy belongs to the module.
- **Evidence:** [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653);
    `s4_debugging_and_logging/logging.md`, "Experiment logging" and its exercises 4, 5 and 9.

### REPRODUCIBILITY_AND_SOFTWARE-13 · "Other options" mixes config tools and trackers

- **Category:** wrong
- **Priority:** must
- **Pages:** 15
- **Finding:** MLflow is listed as an alternative to Hydra, but MLflow Tracking logs parameters, code versions,
    metrics and files. It records a config; it doesn't compose one, and the module lists it as an alternative to W&B.
- **Proposal:** Move MLflow to the tracking section from REPRODUCIBILITY_AND_SOFTWARE-12. Keep or drop Sacred (latest
    release 0.8.7, November 2024; last push October 2025).
- **Evidence:** MLflow, "MLflow Tracking", <https://mlflow.org/docs/latest/ml/tracking/>;
    `s4_debugging_and_logging/logging.md`, "Experiment logging"; GitHub API for `IDSIA/sacred` (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-14 · venv and pyenv are not package managers

- **Category:** wrong
- **Priority:** must
- **Pages:** 16
- **Finding:** venv creates virtual environments and pyenv switches between Python versions. Neither manages packages,
    and neither is in the course's list of package managers.
- **Proposal:** Split the list by job (Python version, environment, packages) or cut it to the module's list, with uv
    first because the course recommends it.
- **Evidence:** `s1_development_environment/package_manager.md`, framework table and the text below it; Python docs,
    "venv", <https://docs.python.org/3/library/venv.html>; pyenv README,
    <https://raw.githubusercontent.com/pyenv/pyenv/master/README.md> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-15 · A pyproject file alone doesn't pin the software

- **Category:** gap
- **Priority:** should
- **Pages:** 16, 22
- **Finding:** Page 16 shows a `pyproject.toml` with `>=` bounds as the level 2 answer, and page 22 a hand-pinned
    Flask `requirements.txt`. The lockfile, which the module installs from in Docker, never appears.
- **Proposal:** Show `pyproject.toml` (broad requirements) next to `uv.lock` (exact resolved versions), and use the
    same pair on page 22.
- **Evidence:** uv, "Structure and files", "The lockfile", <https://docs.astral.sh/uv/concepts/projects/layout/>
    (accessed 2026-11-24); `s1_development_environment/package_manager.md`, "❔ Exercises (uv)" (`uv.lock`);
    `s3_reproducibility/docker.md`, "❔ Exercises" step 12.

### REPRODUCIBILITY_AND_SOFTWARE-16 · Containers: say that the kernel is shared

- **Category:** outdated
- **Priority:** should
- **Pages:** 17, 18, 19
- **Finding:** "Lightweight version of full VMs" and "low overhead virtual machines" blur the point made in the
    keep-list quote. Docker's docs say containers are processes sharing the host kernel and that Docker Desktop on
    macOS and Windows runs them in a Linux VM. "Any host" and "completely independent from the host" also skip CPU
    architecture, which the module warns about.
- **Proposal:** Keep "VM is isolation of machines, while Containers is isolation of processes" as the punchline,
    rewrite the two analogies, and add the arm64/amd64 caveat.
- **Evidence:** Docker, "What is a container?",
    <https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/> (accessed 2026-11-24);
    `s3_reproducibility/docker.md`, "❔ Exercises" step 13 (Apple silicon).

### REPRODUCIBILITY_AND_SOFTWARE-17 · Docker stops at build and run

- **Category:** gap
- **Priority:** should
- **Pages:** 19
- **Finding:** The module also covers sharing through a registry, layer caching, mounting data and tags. The deck has
    only Dockerfile → image → container. A tag can also be replaced, so only a digest guarantees the same base image.
- **Proposal:** Add one slide on layers and caching, and one on sharing (Dockerfile vs registry) with tag vs digest.
    Put data mounts in the caption, see REPRODUCIBILITY_AND_SOFTWARE-18.
- **Evidence:** `s3_reproducibility/docker.md`, "Docker Sharing", "❔ Exercises" steps 12 and 16, "🧠 Knowledge check"
    4; Docker, "Building best practices", <https://docs.docker.com/build/building/best-practices/>; "Docker build
    cache", <https://docs.docker.com/build/cache/> (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-18 · The Dockerfile example contradicts the module

- **Category:** outdated
- **Priority:** must
- **Pages:** 25
- **Finding:** The screenshot runs `dvc init`, `dvc remote add` and `dvc pull` during the build, copies `src/` before
    installing dependencies (which defeats the cache), and uses `setup.py`. DVC leaves the course, and data is to be
    mounted or fetched at run time.
- **Proposal:** Replace it with a short code block from the module's current uv-based Dockerfile, without data in the
    image.
- **Evidence:** [Decide what replaces the DVC content in each affected
    module](https://github.com/SkafteNicki/dtu_mlops/issues/650); `s3_reproducibility/docker.md`, "❔ Exercises" step
    12 annotation; Docker, "Docker build cache" (as in REPRODUCIBILITY_AND_SOFTWARE-17).

### REPRODUCIBILITY_AND_SOFTWARE-19 · Agents as a new source of irreproducibility

- **Category:** addition
- **Priority:** could
- **Pages:** 8, 13
- **Finding:** Nothing on LLMs or coding agents, though an agent's output can differ between identical requests:
    OpenAI calls its API non-deterministic by default, and promises only "mostly" consistent output with a seed.
    Conversely, CORE-Bench measures whether agents can reproduce papers.
- **Proposal:** Add one slide: agents as a threat (pin model and settings, commit what they produce, check it without
    the agent) and as a help. No pointer to A-modules.
- **Evidence:** OpenAI Cookbook,
    <https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter> (2023, uses 2023
    models); CORE-Bench, <https://arxiv.org/abs/2409.11363>; MLRC 2026 as in REPRODUCIBILITY_AND_SOFTWARE-04 (accessed
    2026-11-24); [Update lecture slides for the January 2027
    course](https://github.com/SkafteNicki/dtu_mlops/issues/681).

### REPRODUCIBILITY_AND_SOFTWARE-20 · Give "Regulatory Compliance" a concrete rule

- **Category:** addition
- **Priority:** could
- **Pages:** 5
- **Finding:** The row names no regulation. The EU AI Act requires high-risk systems to log events automatically (Art.
    12). Under the amended Act this applies from 2 December 2027 (Annex III) and 2 August 2028 (Annex I).
- **Proposal:** Name the AI Act's logging requirement on the row, with the dates, and cite the Act.
- **Evidence:** AI Act Explorer, "Article 12", <https://artificialintelligenceact.eu/article/12/>; European
    Commission, "AI Act", <https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai> (accessed
    2026-11-24). EUR-Lex itself could not be read: `unverified` there.

### REPRODUCIBILITY_AND_SOFTWARE-21 · Three definitions, two words

- **Category:** structure
- **Priority:** should
- **Pages:** 3
- **Finding:** The first definition includes "someone else working independently" and the second is repeatability. The
    deck then moves to "full replication" on page 12 without defining replication, while Raff needs a third term,
    independent reproducibility (without the authors' code).
- **Proposal:** Keep the three definitions, add who repeats and with what (code, data), and define replication. Say
    that fields disagree on the words. Cite the sources of the definitions, which are missing.
- **Evidence:** Raff 2019 (file as in REPRODUCIBILITY_AND_SOFTWARE-09), pp. 1–2; Peng 2011,
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC3383002/> (accessed 2026-11-24). ACM's badging terms page returned 403:
    `unverified`.

### REPRODUCIBILITY_AND_SOFTWARE-22 · Cite the spectrum and the ML table

- **Category:** addition
- **Priority:** should
- **Pages:** 12
- **Finding:** The spectrum is Fig. 1 of Peng 2011, and the table is from Raghu et al. 2019 ("Transfusion"). Neither
    is cited. The "same weights vs same mean ± std" point is only implied.
- **Proposal:** Redraw both with citations, and label the two ML ends (identical weights, same conclusion within
    error).
- **Evidence:** Peng 2011, <https://pmc.ncbi.nlm.nih.gov/articles/PMC3383002/>, Fig. 1 caption and text; Raghu et al.
    2019, <https://arxiv.org/pdf/1902.07208>, p. 3 table (accessed 2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-23 · The step screenshots show old tooling

- **Category:** outdated
- **Priority:** should
- **Pages:** 21, 23, 24
- **Finding:** Page 21 shows the old cookiecutter layout (`setup.py`, `src/models/predict_model.py`, `tox.ini`), not
    the course's MLOps template. Page 24 runs Python 3.8 on ubuntu-20.04, windows-2019 and macOS-10.15, with the macOS
    jobs failing; none of those runners is on GitHub's current list.
- **Proposal:** Use the current template tree (code block) and a fresh, passing CI run. Page 23 can become a current
    README excerpt from a course project.
- **Evidence:** `s2_organisation_and_version_control/code_structure.md`, "Cookiecutter" and exercise 2; GitHub Docs,
    "GitHub-hosted runners", <https://docs.github.com/en/actions/reference/runners/github-hosted-runners> (accessed
    2026-11-24).

### REPRODUCIBILITY_AND_SOFTWARE-24 · Language and the agent case

- **Category:** wrong
- **Priority:** must
- **Pages:** 2, 5, 7, 8, 13, 14, 18
- **Finding:** A missing apostrophe in "What's" (p. 2), "If other cannot", "Imaging an AI agent", "two persons …
    different diagnosis", "We are in a crises!", "a lot of … choices that we do", "in principal" and "an lightweight".
    The page 7 case is also looser than the module's: same agent, two hospitals, same input.
- **Proposal:** Fix the language, and state the case with the module's same-input condition. This touches two
    keep-list items (pp. 7 and 8).
- **Evidence:** Deck text; `s3_reproducibility/README.md`, "Why does reproducibility matter".

### REPRODUCIBILITY_AND_SOFTWARE-25 · Memes: "only god knows" and the closing pair

- **Category:** image
- **Priority:** should
- **Pages:** 2, 26
- **Finding:** Page 2's "Dear programmer … total_hours_wasted_here = 254" comment opens the deck as a joke about
    unreproducible code. Page 26 is a "Meme of the day" (programmers vs scientists) with the module link, and closing
    meme slides are out by rule.
- **Proposal:** Keep page 2 as a real code block (author's call). Drop page 26 as a closing slide; its
    Programmers/Scientists joke could move to pages 8–10 if the author wants it. Keep the module link elsewhere.
- **Evidence:** [How should a review brief for one deck be
    structured?](https://github.com/SkafteNicki/dtu_mlops/issues/702); [How should images be recreated, and what may
    we reuse from Gamma exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

### Also noticed (no id, not decided)

- p. 14: Sculley et al. ask that configurations be reviewed and checked into a repository (§6 "Configuration Debt", p.
    6), and name "Reproducibility Debt" (p. 8). In Shankar et al., engineers mostly changed config files instead of
    application code, and P1 warns about forgetting random seeds in parallel processes. Good backing citations.
- p. 7: the module credits its version of the figure as inspired by Strobel & Shokri 2022. Its abstract names privacy,
    fairness, robustness and explainability, not integrity or reproducibility; the deck credits nobody.
- p. 16 and p. 22: `requires-python`, `dvc` and the dependency bounds shown no longer match the repo's
    `pyproject.toml`.
- p. 1: "Day3" and "January 2026"; the new title is topic-only and dated January 2027.
- p. 4: the scientific-method wheel has no arrow from Result back to Observation; the module writes it as a loop.
- p. 13: hardware is the top level, but no module or slide says what to record about it. Raff found "Compute
    Specified" not significant (p = 0.257).
- Not covered in the deck: GPU containers (NVIDIA and AMD) and dev containers, both optional in `docker.md`. Best left
    to the module.
- Hydra 1.4 is at release candidate 3 (2026-10-25). If it ships before the lecture, recheck the code in
    REPRODUCIBILITY_AND_SOFTWARE-02.

## Module follow-ups (out of scope for the deck)

- `s3_reproducibility/config_files.md`, "Configure experiments": `class my_hp` uses annotations (`batch_size: 64`)
    instead of assignments, so `my_hp.batch_size` raises `AttributeError` (run locally); `lr: 128` is also an odd
    value.
- `s3_reproducibility/config_files.md`, "❔ Exercises": no mention of `use_deterministic_algorithms`, data loader seeds
    or the limits in PyTorch's reproducibility note.
- `s3_reproducibility/README.md`, introduction: still names `conda` as the example, while the course now recommends
    `uv`.
- `s3_reproducibility/README.md`, figure caption: "inspired by this paper" links Strobel & Shokri 2022, whose abstract
    does not list integrity or reproducibility.
- `s4_debugging_and_logging/logging.md`, "Experiment logging": `docs.wandb.ai` now redirects to `docs.coreweave.com`;
    step 5 says "yesterdays topic on reproducibility", which is wrong once the module moves to this session.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | Code-editor screenshot of the "Dear programmer … only god knows it!" comment | meme/joke | code block (keep / regenerate / swap?) | voice | REPRODUCIBILITY_AND_SOFTWARE-25 |
| 4 | "The scientific method" wheel: observation, question, hypothesis, experiment, conclusion, result | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 7 | Trustworthy Machine Learning circle overlapped by robustness, integrity, explainability, reproducibility, fairness | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 8 | Nature donut "Is there a reproducibility crisis?": 1,576 researchers; 52% significant, 38% slight, 3% no, 7% don't know | chart/paper figure | re-plot/redraw + citation | | REPRODUCIBILITY_AND_SOFTWARE-06 |
| 9 | Table from Begley & Ellis: citations of reproduced and non-reproduced articles | chart/paper figure | re-plot/redraw + citation | | REPRODUCIBILITY_AND_SOFTWARE-07 |
| 10 | "One data set, many analysts": 29 effect estimates with confidence intervals | chart/paper figure | re-plot/redraw + citation | | REPRODUCIBILITY_AND_SOFTWARE-08 |
| 11 | OpenReview page "ML Reproducibility Challenge 2021 Spring" | tool screenshot | fresh screenshot | stale | REPRODUCIBILITY_AND_SOFTWARE-04 |
| 11 | Excerpt of a conference checklist | code/config screenshot | re-plot/redraw + citation (as slide text) | stale | REPRODUCIBILITY_AND_SOFTWARE-05 |
| 12 | Peng's reproducibility spectrum, Fig. 1 | chart/paper figure | re-plot/redraw + citation | | REPRODUCIBILITY_AND_SOFTWARE-22 |
| 12 | m1.ckpt == m2.ckpt weight vectors, "VS" badge, Transfusion results table | chart/paper figure | re-plot/redraw + citation (math as LaTeX) | | REPRODUCIBILITY_AND_SOFTWARE-22 |
| 13 | Red three-level pyramid | concept diagram | Mermaid/PlantUML/Excalidraw | | REPRODUCIBILITY_AND_SOFTWARE-01 |
| 14 | Three snippets: `class hparams`, `argparse` call, `experiment1.yaml` | code/config screenshot | code block | | |
| 15 | Config tree and Hydra script | code/config screenshot | code block (from exercise files where possible) | stale | REPRODUCIBILITY_AND_SOFTWARE-02 |
| 16 | VS Code view of an old `pyproject.toml` | code/config screenshot | code block | stale | REPRODUCIBILITY_AND_SOFTWARE-15 |
| 17 | Hypervisor drawing: three laptops (Windows, macOS, Linux logos) above a host OS and its hardware | concept diagram | Mermaid/PlantUML/Excalidraw + Iconify logo | | REPRODUCIBILITY_AND_SOFTWARE-16 |
| 18 | VM stack vs container stack | concept diagram | Mermaid/PlantUML/Excalidraw | | REPRODUCIBILITY_AND_SOFTWARE-16 |
| 19 | Dockerfile → image → container, Geekflare watermark | concept diagram | Mermaid/PlantUML/Excalidraw + Iconify logo | | REPRODUCIBILITY_AND_SOFTWARE-17 |
| 20–25 | Step icons: Git, cookiecutter, Python/PyPI, document, test tube, Docker | tool logo | Iconify logo (cookiecutter icon unverified); emoji for document and test tube | | |
| 20 | GitHub commit list from August 2022 | tool screenshot | fresh screenshot | stale | |
| 21 | Old cookiecutter data science tree | code/config screenshot | code block | stale | REPRODUCIBILITY_AND_SOFTWARE-23 |
| 22 | `requirements.txt` with pinned Flask packages | code/config screenshot | code block | stale | REPRODUCIBILITY_AND_SOFTWARE-15 |
| 23 | README "Train & Test" commands of a research repo | code/config screenshot | code block | | REPRODUCIBILITY_AND_SOFTWARE-23 |
| 24 | GitHub Actions run with failing macOS jobs | tool screenshot | fresh screenshot | stale | REPRODUCIBILITY_AND_SOFTWARE-23 |
| 25 | Dockerfile with DVC steps | code/config screenshot | code block (from exercise files where possible) | stale | REPRODUCIBILITY_AND_SOFTWARE-18 |
| 26 | Two-panel meme: Programmers (crying) vs Scientists ("I know") | meme/joke | drop as closing slide; author may move it | voice | REPRODUCIBILITY_AND_SOFTWARE-25 |
| 1–26 | Dark Gamma background art, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo ·
meme/joke · decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated ·
replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Raff 2019, *A Step Toward Quantifying Independently Reproducible Machine Learning Research* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/A_step_toward_quantifying_independently_reproducible_machine_learning_research.pdf> | REPRODUCIBILITY_AND_SOFTWARE-09, REPRODUCIBILITY_AND_SOFTWARE-10, REPRODUCIBILITY_AND_SOFTWARE-21 |
| Kapoor & Narayanan 2022, *Leakage and the Reproducibility Crisis in ML-based Science* | <https://arxiv.org/abs/2207.07048> | REPRODUCIBILITY_AND_SOFTWARE-09 |
| Baker 2016, *1,500 scientists lift the lid on reproducibility* | <https://www.nature.com/articles/533452a> | REPRODUCIBILITY_AND_SOFTWARE-06, p. 8 chart |
| Nature, *Challenges in irreproducible research* | <https://www.nature.com/collections/prbfkwmwvz> | REPRODUCIBILITY_AND_SOFTWARE-06, p. 8 text |
| Begley & Ellis 2012, *Raise standards for preclinical cancer research* | <https://www.nature.com/articles/483531a> | REPRODUCIBILITY_AND_SOFTWARE-07, p. 9 |
| Silberzahn & Uhlmann 2015, *Crowdsourced research: Many hands make tight work* | <https://www.nature.com/articles/526189a> | REPRODUCIBILITY_AND_SOFTWARE-08, p. 10 figure (figure source not checked, paywall: `unverified`) |
| Silberzahn et al. 2018, *Many Analysts, One Data Set* | <https://doi.org/10.1177/2515245917747646> | REPRODUCIBILITY_AND_SOFTWARE-08 |
| Peng 2011, *Reproducible Research in Computational Science* | <https://pmc.ncbi.nlm.nih.gov/articles/PMC3383002/> | REPRODUCIBILITY_AND_SOFTWARE-22, REPRODUCIBILITY_AND_SOFTWARE-21 |
| Raghu et al. 2019, *Transfusion: Understanding Transfer Learning for Medical Imaging* | <https://arxiv.org/abs/1902.07208> | REPRODUCIBILITY_AND_SOFTWARE-22, p. 12 table |
| MLRC 2026 | <https://reproml.org/> | REPRODUCIBILITY_AND_SOFTWARE-03, REPRODUCIBILITY_AND_SOFTWARE-04, REPRODUCIBILITY_AND_SOFTWARE-19 |
| NeurIPS, *Paper Checklist Guidelines* | <https://neurips.cc/public/guides/PaperChecklist> | REPRODUCIBILITY_AND_SOFTWARE-05 |
| PyTorch, *Reproducibility* | <https://docs.pytorch.org/docs/2.15/notes/randomness.html> | REPRODUCIBILITY_AND_SOFTWARE-11 |
| Docker, *What is a container?* | <https://docs.docker.com/get-started/docker-concepts/the-basics/what-is-a-container/> | REPRODUCIBILITY_AND_SOFTWARE-16 |
| Docker, *Building best practices* | <https://docs.docker.com/build/building/best-practices/> | REPRODUCIBILITY_AND_SOFTWARE-17 |
| uv, *Structure and files* | <https://docs.astral.sh/uv/concepts/projects/layout/> | REPRODUCIBILITY_AND_SOFTWARE-15 |
| CORE-Bench (Siegel et al. 2024) | <https://arxiv.org/abs/2409.11363> | REPRODUCIBILITY_AND_SOFTWARE-19 |
| EU AI Act, Article 12 | <https://artificialintelligenceact.eu/article/12/> | REPRODUCIBILITY_AND_SOFTWARE-20 |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | p. 14 (also noticed) |
| Shankar et al., *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | p. 14 (also noticed) |
