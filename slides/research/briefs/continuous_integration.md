# Review brief: continuous_integration (`slides/day6_continuous_integration.pdf`)

- **Modules compared:** `s5_continuous_integration/` (`README.md`, `unittesting.md`, `github_actions.md`,
    `pre_commit.md`, `cml.md`, `exercise_files/`) @ `main` d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b
- **Other inputs:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`,
    `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, web sources (accessed 2026-11-24)
- **Pages:** 37 PDF pages → 37 slides. There are no click builds to merge and no Gamma "Error uploading image"
    placeholders. Pages 28 to 30 share a title but have different content. Every page render was checked by eye.
    Text-extraction glyph errors (for example p. 25 "Let9s") are not in the render; some emoji bullets on pp. 4, 7, 13
    and others render blank.

## Deck outline (narrative order to keep)

1. Title and class announcement: pages 1–2
2. Why care, Continuous X, CI/CD/CML, maturity levels: pages 3–6
3. The core task, three steps, and the TorchMetrics case study: pages 7–8
4. Step 1, version control (commits, branches, pull requests, pre-commit): pages 9–12
5. Step 2, tests (types, layout, pytest, simple and complicated): pages 13–18
6. Class exercises with solutions (data contract, smoke test, ML bug hunting): pages 19–24
7. From running locally to step 3, automation (Actions, workflow files, merge rules, bots): pages 25–33
8. Summary: page 34
9. Agents, template corrections, meme of the day: pages 35–37

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "How to automatically secure that code does not break during development?" (p. 7)
    - "Create it → Break it → Fix it →..." (p. 9)
    - "Tests are the cornerstones of continuous integration" (p. 13)
    - "Test can be simple..." / "or complicated" (pp. 17–18)
    - "Do you remember to do this before each commit?" / "Let's automate doing it instead" (p. 25)
    - "What can be automated: EVERYTHING" / "Only your imagination is the limit..." (p. 26)
    - "The agents are here" (p. 35)
- Voice markers:
    - The Fastly outage as the opening hook (p. 3). CONTINUOUS_INTEGRATION-01 corrects the facts.
    - The three-step spine (version control, tests, automate), repeated in titles and the summary (pp. 7–34)
    - TorchMetrics, the author's own project, as the running case study (pp. 8–33). CONTINUOUS_INTEGRATION-08
        refreshes it.
    - Three partner exercises with a solution slide each (pp. 19–24)
    - The arrows and brackets drawn on real screenshots (pp. 4, 11, 28, 29, 33)

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

- [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    the GitHub Actions module authenticates to GCP and pulls a pinned W&B reference artifact; CML triggers on pipeline
    changes with plain Actions steps in place of `cml`. The deck never shows DVC or `cml`, so nothing is removed. →
    see CONTINUOUS_INTEGRATION-12, -13, -18
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648): pipeline versioning
    plus a pinned dataset gives Exercise 3 its test set. → see CONTINUOUS_INTEGRATION-13
- [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649): pandera checks inside
    the pipeline arrive before this session. → see CONTINUOUS_INTEGRATION-22
- [Design the week-3 loop closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (open; not a decision yet):
    owns the new-data trigger. → see CONTINUOUS_INTEGRATION-13
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): W&B logging
    moves to Day 3, so the registry-driven CML case has its prerequisite. → see CONTINUOUS_INTEGRATION-12
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651): the module's
    MNIST tests move to Rock-Paper-Scissors, pending the real-photo check. Any course test shown must follow. → see
    CONTINUOUS_INTEGRATION-15
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621)
    (agentic map): this session gets an Agent review bot module. The lecture may cover agents but does not reference
    it. → see CONTINUOUS_INTEGRATION-19
- "Whether pipeline versioning reshapes the CML module" is still under Not yet specified on the restructure map;
    re-check before writing CONTINUOUS_INTEGRATION-12.

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased (e.g. CLOUD-01) -->

### CONTINUOUS_INTEGRATION-01 · Fastly story: wrong year, cause and duration

- **Category:** wrong
- **Priority:** must
- **Pages:** 3
- **Finding:** The slide says "3 years ago, the day before this lecture" and blames Fastly's CI. The outage was on 8
    June 2021. A valid customer configuration change triggered a latent bug that a 12 May deployment had introduced,
    and 95% of the network was normal within 49 minutes. Keep-list story.
- **Proposal:** Keep the story and its point, and date it, with the real cause: a bug passed into production and
    waited weeks for its trigger. Cite and link Fastly's post instead of its homepage. "A couple of hours" can stand
    only as start to mitigation (09:47 to 12:35).
- **Evidence:** Fastly, "Summary of June 8 outage", <https://www.fastly.com/blog/summary-of-june-8-outage> (accessed
    2026-11-24).

### CONTINUOUS_INTEGRATION-02 · "Continues" should be "Continuous"

- **Category:** wrong
- **Priority:** must
- **Pages:** 3, 4, 5, 7, 34
- **Finding:** The deck's key term is spelled "Continues" in the titles and tables of five pages. Pages 1 and 8 and
    the module have "Continuous".
- **Proposal:** Fix the spelling throughout, together with the other typos listed under Also noticed.
- **Evidence:** `s5_continuous_integration/README.md`, introduction; pages 1, 4 (figure labels) and 8 of the deck.

### CONTINUOUS_INTEGRATION-03 · Free minutes: public and private are swapped, and the unit has changed

- **Category:** wrong
- **Priority:** must
- **Pages:** 27
- **Finding:** The slide gives 2,000 free minutes per month for public repositories. Standard runners are free in
    public repositories; the included allowance applies to private ones. GitHub now states it as money ($12 per month
    on GitHub Free), converted at a per-minute rate.
- **Proposal:** Say that public is free, and private has a monthly allowance (give the dollar amount and its rough
    Linux minutes). Crop the "Private" repository header.
- **Evidence:** GitHub Docs, "GitHub Actions" billing,
    <https://docs.github.com/en/billing/concepts/product-billing/github-actions> (accessed 2026-11-24): $12 for GitHub
    Free, $0.006 per minute on a standard Linux runner; `s5_continuous_integration/github_actions.md`, "GitHub
    actions".

### CONTINUOUS_INTEGRATION-04 · The second pytest example fails

- **Category:** wrong
- **Priority:** must
- **Pages:** 16
- **Finding:** `test_mse_loss_ones` has the comment `(1 - 0)^2 = 1` but asserts that the loss `== 0`, so it fails.
- **Proposal:** Assert 1, or make it an explicit "which test fails?" question if the failure is intended (author's
    call).
- **Evidence:** Page 16, comment vs assert (arithmetic: mean of (1 - 0)^2 is 1).

### CONTINUOUS_INTEGRATION-05 · The test-layout trees contradict the src layout

- **Category:** wrong
- **Priority:** must
- **Pages:** 15
- **Finding:** Both trees put `__init__.py` directly in `src/`, which makes `src` the package, while p. 14 teaches
    `src/<project_name>`. The two alternatives also run together without a label.
- **Proposal:** Use `src/<project_name>/` in both trees and label them (separate `tests/` as the course default; tests
    beside the code as the alternative).
- **Evidence:** pytest, "Good Integration Practices",
    <https://docs.pytest.org/en/stable/explanation/goodpractices.html> (accessed 2026-11-24), which shows
    `src/mypkg/`; page 14 of the deck; `from my_project...` imports in `s5_continuous_integration/unittesting.md`, "❔
    Exercises".

### CONTINUOUS_INTEGRATION-06 · "43 checks in total" is the number that passed

- **Category:** wrong
- **Priority:** must
- **Pages:** 30, 31
- **Finding:** The TorchMetrics screenshot on p. 31 says "4 failing and 43 successful checks", i.e. 47. If p. 30 shows
    the same run (`unverified`), its 43 is the passing count. The p. 30 screenshot also shows many red crosses beside
    a ✅.
- **Proposal:** Take both numbers and both screenshots from one current PR.
- **Evidence:** Pages 30 and 31 (rendered text in the screenshots); that both show the same run is `unverified`.

### CONTINUOUS_INTEGRATION-07 · Workflow example is out of date

- **Category:** outdated
- **Priority:** must
- **Pages:** 28
- **Finding:** The annotated workflow tests Python 3.7 to 3.10 (all end-of-life), uses `actions/checkout@v3` and
    `setup-python@v4`, installs with `python setup.py install`, and lints with flake8. The course teaches uv and ruff
    and its file has current action versions.
- **Proposal:** Keep the annotated-brackets format and import the module's own `tests_uv.yaml` as a code block, so
    slide and exercise stay in sync.
- **Evidence:** `s5_continuous_integration/exercise_files/tests_uv.yaml`, `format_uv.yaml`; Python Developer's Guide,
    "Status of Python versions", <https://devguide.python.org/versions/>; PyPA, "Is setup.py deprecated?",
    <https://packaging.python.org/en/latest/discussions/setup-py-deprecated/> (accessed 2026-11-24).

### CONTINUOUS_INTEGRATION-08 · The TorchMetrics case study is from about 2022

- **Category:** outdated
- **Priority:** should
- **Pages:** 8, 9, 10, 11, 14, 15, 18, 29, 30, 31, 32
- **Finding:** The case study shows Python 3.7, ubuntu-20.04 runners, Azure Pipelines and the old file names. The
    project now requires Python 3.10+, tests PyTorch 2.6 to 2.10, and has renamed its workflows; ubuntu-20.04 runners
    went out of support in April 2025.
- **Proposal:** Keep TorchMetrics (voice marker) and retake all its screenshots in one pass, as readable crops. The
    matrix in its `ci-tests.yml` can be a code block.
- **Evidence:** TorchMetrics `ci-tests.yml`,
    <https://github.com/Lightning-AI/torchmetrics/blob/master/.github/workflows/ci-tests.yml>; PyPI metadata,
    <https://pypi.org/project/torchmetrics/>; actions/runner-images,
    <https://github.com/actions/runner-images/issues/11101> (accessed 2026-11-24).

### CONTINUOUS_INTEGRATION-09 · Branch protection: the course uses rulesets

- **Category:** outdated
- **Priority:** should
- **Pages:** 11, 31
- **Finding:** The slide names and links "Branch protection rules". The module sets up a ruleset (`Settings -> Rules
    -> Rulesets`) but calls the concept branch protection rules. Several rulesets can apply to a branch at once, and
    they can exempt chosen users.
- **Proposal:** Say "branch protection, set up as a ruleset" and update the link. Keep the three rules; they match the
    module's. Make p. 11's "No commit can be pushed" say that a ruleset enforces it.
- **Evidence:** `s5_continuous_integration/github_actions.md`, "❔ Exercises" on branch protection; GitHub Docs, "About
    rulesets",
    <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets>
    (accessed 2026-11-24).

### CONTINUOUS_INTEGRATION-10 · CD: delivery or deployment

- **Category:** outdated
- **Priority:** should
- **Pages:** 4, 5
- **Finding:** The figure on p. 4 says Continuous Delivery; the table on p. 5 expands CD as Deployment and adds
    monitoring. Fowler separates them: delivery keeps the product releasable, deployment releases every passing commit
    automatically.
- **Proposal:** Name both under CD with the one-line distinction. Leave monitoring to its own session.
- **Evidence:** Fowler, "Continuous Integration", <https://martinfowler.com/articles/continuousIntegration.html>,
    section on continuous delivery and deployment (accessed 2026-11-24); `s5_continuous_integration/README.md`
    ("continuous delivery").

### CONTINUOUS_INTEGRATION-11 · Maturity model: cite it and make it readable

- **Category:** gap
- **Priority:** should
- **Pages:** 6
- **Finding:** The slide is close to Microsoft's wording with no citation, its table is unreadable at lecture size,
    and it never says where CI and CML sit. The module walks through levels 0 to 4 and places CML at the top.
    Microsoft now presents the levels as a continuum, and says GenAIOps complements them.
- **Proposal:** Show the five levels as a ladder with the course sessions marked on it, cited. Replace the three
    consultancy bullets with the course's "where are we?" view.
- **Evidence:** `s5_continuous_integration/cml.md`, "MLOps maturity model"; Microsoft, "MLOps maturity model",
    <https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model> (accessed 2026-11-24).

### CONTINUOUS_INTEGRATION-12 · CML is named once and then dropped

- **Category:** gap
- **Priority:** should
- **Pages:** 5, 24, 34
- **Finding:** The session has a CML module, with a data-check workflow that comments on the PR and a staged-model
    workflow. The deck defines CML on p. 5 and never shows either.
- **Proposal:** Add one slide after Exercise 3 with two triggers and their checks: pipeline change → run the pipeline
    and post dataset statistics; model staged → performance test → `production` alias. Use plain Actions steps, not
    the `cml` tool.
- **Evidence:** `s5_continuous_integration/cml.md`, "❔ Exercises" 1 and 2; [Decide what replaces the DVC content in
    each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650); `iterative/cml` last release v0.20.6,
    2024-10-24, <https://github.com/iterative/cml/releases> (accessed 2026-11-24).

### CONTINUOUS_INTEGRATION-13 · Trigger on the pipeline, test on pinned data

- **Category:** addition
- **Priority:** should
- **Pages:** 5, 24
- **Finding:** "When data and code changes" (p. 5) predates the restructure: here CML triggers on pipeline code and
    params, and the new-data trigger moves to the week-3 loop. Exercise 3's "stable, versioned test dataset" now has a
    concrete mechanism, a W&B reference artifact.
- **Proposal:** Word p. 5 as code, pipeline or data changes, with a pointer to the later loop for new data. Name the
    pinned artifact in the Exercise 3 answer.
- **Evidence:** [Decide what replaces the DVC content in each affected
    module](https://github.com/SkafteNicki/dtu_mlops/issues/650); [How do modern ML pipelines version
    data?](https://github.com/SkafteNicki/dtu_mlops/issues/648); [Design the week-3 loop
    closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (open).

### CONTINUOUS_INTEGRATION-14 · Why ML systems need more than code tests

- **Category:** gap
- **Priority:** should
- **Pages:** 13, 19
- **Finding:** The deck jumps from unit tests to the data exercises without the module's reason: ML behavior also
    depends on data, so data, model and infrastructure need tests. Sculley et al. add that unit and end-to-end tests
    are not sufficient in a changing world.
- **Proposal:** Add the module's software-vs-ML testing figure as the bridge into Exercise 1, with the Sculley and ML
    Test Score citations.
- **Evidence:** `s5_continuous_integration/unittesting.md`, introduction and `system_difference` figure;
    `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, p. 7, "Monitoring and Testing"; Breck et al.,
    <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/>
    (accessed 2026-11-24) (abstract only).

### CONTINUOUS_INTEGRATION-15 · Show the course's data, model and training tests

- **Category:** gap
- **Priority:** should
- **Pages:** 16, 17, 18
- **Finding:** The examples test an MSE loss and a metrics library. The module asks for data, model and training tests
    (shapes, labels, raised errors), uses `skipif` for missing data and parametrization, and none of these is shown on
    the course's own project.
- **Proposal:** Put one data test and one parametrized model test from the module next to the simple example, and keep
    the "simple... or complicated" contrast.
- **Evidence:** `s5_continuous_integration/unittesting.md`, "❔ Exercises" 4 and 6.

### CONTINUOUS_INTEGRATION-16 · Say what coverage does and doesn't tell you

- **Category:** gap
- **Priority:** should
- **Pages:** 26
- **Finding:** "Code coverage" is one bullet. The module teaches measuring it and that 100% is no guarantee of
    bug-free code.
- **Proposal:** One line or a quick class question: does 100% coverage mean no bugs?
- **Evidence:** `s5_continuous_integration/unittesting.md`, "❔ Exercises" 7 and "🧠 Knowledge check" 1.

### CONTINUOUS_INTEGRATION-17 · Define workflow, job, step, runner

- **Category:** gap
- **Priority:** should
- **Pages:** 28, 29
- **Finding:** The module's knowledge check asks for workflow, runner, job and action; the deck defines only workflow
    files. "Hosted by Github" also skips self-hosted runners.
- **Proposal:** Add the hierarchy to the annotated example: event → workflow → jobs (parallel by default) → steps on a
    runner, plus where matrix and caching fit.
- **Evidence:** `s5_continuous_integration/github_actions.md`, "GitHub actions" and "🧠 Knowledge check" 1.

### CONTINUOUS_INTEGRATION-18 · Workflow security: permissions, pinned actions, secrets

- **Category:** addition
- **Priority:** should
- **Pages:** 26, 27
- **Finding:** "Security checks" is only a bullet. A workflow runs third-party code that can reach the repository's
    secrets and `GITHUB_TOKEN`. GitHub advises least privilege, avoiding `pull_request_target` where not needed, and
    locking actions with a lockfile or, failing that, a full commit SHA. The course's `tests_*.yaml` files use tags
    and set no `permissions`.
- **Proposal:** Add one slide: least-privilege `permissions`, secrets, pin third-party actions, and Dependabot to keep
    pins fresh. Mention Workload Identity Federation as best practice beside the JSON-key default for cloud access.
- **Evidence:** GitHub Docs, "Secure use reference",
    <https://docs.github.com/en/actions/reference/security/secure-use> (accessed 2026-11-24); [Decide what replaces
    the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650);
    `s5_continuous_integration/github_actions.md`, "Dependabot" 4.

### CONTINUOUS_INTEGRATION-19 · Agents: from an issue link to agents in CI

- **Category:** outdated
- **Priority:** should
- **Pages:** 35
- **Finding:** "The agents are here" rests on a January 2026 issue link, with no lesson. By late 2026 agents are part
    of CI: GitHub Agentic Workflows run in Actions with limited permissions, and Copilot reviews PRs. GitHub says its
    review should be supplemented by a human one.
- **Proposal:** Keep the title and move the slide after the bots (deterministic bot → review agent → agent opening
    PRs). Make the point that agent PRs pass the same required checks and a human approves. Do not reference the Agent
    review bot module.
- **Evidence:** GitHub Agentic Workflows, <https://github.github.com/gh-aw/>; GitHub Docs, "Copilot code review",
    <https://docs.github.com/en/copilot/concepts/agents/code-review> (accessed 2026-11-24); [Update lecture slides for
    the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681); [Spec: Agentic modules
    (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682).

### CONTINUOUS_INTEGRATION-20 · Agents make CI matter more

- **Category:** addition
- **Priority:** should
- **Pages:** 7, 35
- **Finding:** DORA's 2025 announcement says AI-driven change volume leads to instability without strong automated
    testing, mature version control and fast feedback, which are the deck's three steps.
- **Proposal:** Use it as a one-line cited bridge: in the opening (p. 7) or on the agents slide.
- **Evidence:** Google Cloud, "Announcing the 2025 DORA Report",
    <https://cloud.google.com/blog/products/ai-machine-learning/announcing-the-2025-dora-report> (accessed
    2026-11-24).

### CONTINUOUS_INTEGRATION-21 · Name an agent-proof test rule

- **Category:** addition
- **Priority:** could
- **Pages:** 13, 23
- **Finding:** When an agent writes both code and tests, a green run can show only that they agree; the deck has no
    mention of this. This is a teaching judgment, not a sourced fact.
- **Proposal:** A short discussion prompt after Exercise 3: which tests would you not let an agent edit?
- **Evidence:** `unverified`; related course rule: every A-module ends in a check that does not rely on the agent
    ([Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), format).

### CONTINUOUS_INTEGRATION-22 · Exercise answers: schema library and a baseline

- **Category:** addition
- **Priority:** could
- **Pages:** 20, 24
- **Finding:** Exercise 1's answer names no tool though Day 4 will teach pandera. Exercise 3's fixed `>= 90%` would
    pass a drop from 95% to 91%; the module's question is "Did a metric that I care about improve?".
- **Proposal:** Add "e.g. pandera" to answer 1, and a comparison with the current production model, with a tolerance,
    beside the fixed threshold in answer 3. Keep all three exercises.
- **Evidence:** [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649);
    `s5_continuous_integration/cml.md`, introduction (failure modes).

### CONTINUOUS_INTEGRATION-23 · Drop the course-admin slides

- **Category:** structure
- **Priority:** should
- **Pages:** 2, 36
- **Finding:** Page 2 ("50 people and 5 groups...") and p. 36 (template corrections, pointing to a PR merged on
    2026-01-12) belong to the January 2026 run. Neither is about CI.
- **Proposal:** Drop p. 36. Keep p. 2 only as a reusable attendance or group-registration slide, with new numbers or
    none (author's call, see also CONTINUOUS_INTEGRATION-24).
- **Evidence:** Pages 2 and 36; `SkafteNicki/mlops_template` PR 93 merged 2026-01-12,
    <https://github.com/SkafteNicki/mlops_template/pull/93> (accessed 2026-11-24).

### CONTINUOUS_INTEGRATION-24 · Memes and joke images: Fastly, and the lecture hall

- **Category:** image
- **Priority:** should
- **Pages:** 2, 3
- **Finding:** Page 3's meme is a celebrity photo with company logos; it carries the story's point. Page 2 is an
    AI-looking lecture hall with a swear-word slogan on the screen, and no teaching point.
- **Proposal:** Author's call. Suggested: regenerate p. 3 as a cartoon with a generic developer and captions as slide
    text (re-captioned per CONTINUOUS_INTEGRATION-01); drop or regenerate p. 2.
- **Evidence:** [How should images be recreated, and what may we reuse from Gamma
    exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705) (memes are decided per deck).

### CONTINUOUS_INTEGRATION-25 · Closing meme: bots talking to bots

- **Category:** image
- **Priority:** should
- **Pages:** 37
- **Finding:** The "Meme of the day" is a named person's tweet about a fully bot-run dependency PR. The cross-deck
    rule ends decks without a closing meme, but this one carries the bots slide's point.
- **Proposal:** Drop p. 37. If the author wants the joke, the text can move to the Dependabot slide (p. 33) as a
    credited quote, leading into agents (CONTINUOUS_INTEGRATION-19).
- **Evidence:** [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710),
    cross-deck rule 2; page 37. The tweet's original URL is `unverified`.

### Also noticed (no id, not decided)

- Typos and wording: p. 4 "make sure changed", p. 9 "User version control", p. 13 "arguable", p. 16 "Test are", p. 17
    title "tets", p. 18 title "write test", p. 27 the first word of line one should be "Built-in", and "Github" should
    be "GitHub"; stray quote marks on pp. 19 and 23.
- p. 10 recommends Git Graph: last release v1.30.0 in April 2021, last push July 2023
    (<https://github.com/mhutchie/vscode-git-graph>, accessed 2026-11-24). GitLens or the built-in graph would do.
- p. 12 pins ruff pre-commit v0.1.3 with the `ruff` hook id; the module uses v0.17.6 and `ruff-check`
    (`s5_continuous_integration/pre_commit.md`, "❔ Exercises" 7). Import the module's config.
- p. 17 uses the `tmpdir` fixture; pytest calls it deprecated in favor of `tmp_path`
    (<https://docs.pytest.org/en/stable/how-to/tmp_path.html>, accessed 2026-11-24). The current TorchMetrics test
    takes no fixture.
- p. 25 legend: the marker is `pytest.mark.skipif`, not `pytest.skipif` (`s5_continuous_integration/unittesting.md`,
    "❔ Exercises" 4). It shows Python 3.8 on Windows.
- p. 32 shows pre-commit.ci fixing commits; the module does the same with an Actions workflow, so the slide could name
    both (`s5_continuous_integration/pre_commit.md`, "❔ Exercises" 11). p. 33 could add the `uv` and `pre-commit`
    ecosystems (<https://github.blog/changelog/2026-03-10-dependabot-now-supports-pre-commit-hooks>, accessed
    2026-11-24).
- The module opens with Fowler's "doesn't get rid of bugs" quote, verified in his article. It could go on p. 13.
    Fowler also ties CI to integrating at least daily, which long-lived branches (p. 10) work against.
- p. 11 on reviews: Shankar et al. report that participants did code review to follow best practice though they felt
    it "wasn't too useful" (`literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`, p. 9). A good
    discussion point.
- p. 1: "Day6 - Continuous Integration", January 2026; topic-only naming would make it "Continuous Integration",
    January 2027.

## Module follow-ups (out of scope for the deck)

- `s5_continuous_integration/github_actions.md`, "GitHub actions": the private allowance is now stated in dollars ($12
    per month on GitHub Free), not 2,000 minutes.
- `s5_continuous_integration/github_actions.md`, "❔ Exercises" 9: calls rulesets "branch protection rules"; say that
    rulesets are the newer way to get branch protection.
- `s5_continuous_integration/github_actions.md`, "🧠 Knowledge check" 1: says an action is "the smallest unit in a
    workflow" and jobs consist of actions; jobs consist of steps, some of which use actions.
- `s5_continuous_integration/github_actions.md`, "Dependabot": version tags are explained but SHA pinning, lockfiles
    and `permissions` are not; `tests_*.yaml` set no `permissions`.
- `s5_continuous_integration/cml.md`, "❔ Exercises" 2 step 8: workflow interpolates `client_payload` values straight
    into `run:`; GitHub's secure-use page covers passing such values through environment variables.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | AI-looking lecture hall with a swear-word slogan on the screen | meme/joke | keep / regenerate / swap? | voice | CONTINUOUS_INTEGRATION-24 |
| 3 | "Dev at Fastly" meme: celebrity face over company logos | meme/joke | keep / regenerate / swap? | voice | CONTINUOUS_INTEGRATION-24 |
| 4 | Plan-to-operate pipeline with CI and CD loops and red callouts | concept diagram | Mermaid/PlantUML/Excalidraw | | CONTINUOUS_INTEGRATION-10 |
| 6 | Microsoft maturity-model table, unreadable | chart/paper figure | re-plot/redraw + citation | stale | CONTINUOUS_INTEGRATION-11 |
| 8 | TorchMetrics repository page (Python 3.7–3.10 badges) | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-08 |
| 9 | GitHub commit list, March 2021 | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-08 |
| 10 | GitLens commit graph of TorchMetrics | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-08 |
| 11 | PR review with numbered red boxes | tool screenshot | fresh screenshot, callouts as overlays | stale | CONTINUOUS_INTEGRATION-08 |
| 12 | `.pre-commit-config.yaml` and a failing hook run | code/config screenshot | code block (module config) | stale | |
| 13 | Unit tests pointing at pieces of production code | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 14, 15 | TorchMetrics source and tests folders | tool screenshot | code block (file tree) | stale | CONTINUOUS_INTEGRATION-05, CONTINUOUS_INTEGRATION-08 |
| 16 | Two MSE tests | code/config screenshot | code block | | CONTINUOUS_INTEGRATION-04 |
| 17 | NaN warning test | code/config screenshot | code block | stale | |
| 18 | Parametrized confusion-matrix test | code/config screenshot | code block (current source) | stale | CONTINUOUS_INTEGRATION-08 |
| 19 | Kaggle link card with house photo | photo | licensed stand-in or plain link | | |
| 20 | Calculator, sun and tag icons | decorative | replaced by theme | | |
| 25 | pytest run on Windows; symbol legend | code/config screenshot | code block | stale | |
| 27 | Actions tab of a private repo; starter workflows | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-03 |
| 28 | Annotated workflow file | code/config screenshot | code block (`tests_uv.yaml`) + callouts | stale | CONTINUOUS_INTEGRATION-07 |
| 29 | TorchMetrics workflows folder and steps | tool screenshot + code | fresh screenshot + code block | stale | CONTINUOUS_INTEGRATION-08 |
| 30 | TorchMetrics check lists | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-06, CONTINUOUS_INTEGRATION-08 |
| 31 | Blocked merge box with failing checks | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-06, CONTINUOUS_INTEGRATION-08, CONTINUOUS_INTEGRATION-09 |
| 32 | pre-commit.ci auto-fix commits and diff | tool screenshot | fresh screenshot | stale | CONTINUOUS_INTEGRATION-08 |
| 33 | Dependabot PR and its config | tool screenshot + code | fresh screenshot + code block | stale | |
| 34 | Git, pytest and GitHub Actions logos | tool logo | Iconify logo | | |
| 35 | GitHub card for issue 480 | tool screenshot | replace with link or current example | stale | CONTINUOUS_INTEGRATION-19 |
| 36 | GitHub card for template PR 93 | tool screenshot | dropped with the slide | stale | CONTINUOUS_INTEGRATION-23 |
| 37 | Tweet: a bot fixed, merged and celebrated a PR | meme/joke | drop, or quote as text with credit | voice | CONTINUOUS_INTEGRATION-25 |
| 1–37 | Dark Gamma background art, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo ·
meme/joke · decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated ·
replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Fastly 2021, *Summary of June 8 outage* | <https://www.fastly.com/blog/summary-of-june-8-outage> | CONTINUOUS_INTEGRATION-01, p. 3 |
| Fowler, *Continuous Integration* | <https://martinfowler.com/articles/continuousIntegration.html> | CONTINUOUS_INTEGRATION-10, p. 5; p. 13 |
| GitHub Docs, *GitHub Actions billing* | <https://docs.github.com/en/billing/concepts/product-billing/github-actions> | CONTINUOUS_INTEGRATION-03, p. 27 |
| GitHub Docs, *Secure use reference* | <https://docs.github.com/en/actions/reference/security/secure-use> | CONTINUOUS_INTEGRATION-18, p. 26 |
| GitHub Docs, *About rulesets* | <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets> | CONTINUOUS_INTEGRATION-09, p. 31 |
| pytest, *Good Integration Practices* | <https://docs.pytest.org/en/stable/explanation/goodpractices.html> | CONTINUOUS_INTEGRATION-05, pp. 14–15 |
| Python Developer's Guide, *Status of Python versions* | <https://devguide.python.org/versions/> | CONTINUOUS_INTEGRATION-07, p. 28 |
| Lightning-AI, TorchMetrics `ci-tests.yml` | <https://github.com/Lightning-AI/torchmetrics/blob/master/.github/workflows/ci-tests.yml> | CONTINUOUS_INTEGRATION-08, pp. 29–30 |
| Microsoft, *MLOps maturity model* | <https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model> | CONTINUOUS_INTEGRATION-11, p. 6 |
| Breck et al., *The ML Test Score* | <https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/> | CONTINUOUS_INTEGRATION-14, p. 13 |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | CONTINUOUS_INTEGRATION-14 |
| Shankar et al. 2022, *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | Also noticed, p. 11 |
| Google Cloud 2025, *Announcing the 2025 DORA Report* | <https://cloud.google.com/blog/products/ai-machine-learning/announcing-the-2025-dora-report> | CONTINUOUS_INTEGRATION-20, p. 7 |
| GitHub Docs, *Copilot code review* | <https://docs.github.com/en/copilot/concepts/agents/code-review> | CONTINUOUS_INTEGRATION-19, p. 35 |
| GitHub, *Agentic Workflows* | <https://github.github.com/gh-aw/> | CONTINUOUS_INTEGRATION-19, p. 35 |
