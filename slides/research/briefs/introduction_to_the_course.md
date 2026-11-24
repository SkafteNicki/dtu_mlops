# Review brief: introduction_to_the_course (`slides/day1_introduction_to_the_course.pdf`)

- **Modules compared:** `pages/` (`timeplan.md`, `projects.md`, `before.md`, `faq.md`, `overview.md`), `README.md`,
    `reports/README.md`, `s1_development_environment/README.md`, `mkdocs.yml` @ `main`
    d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b. There is no session folder for this deck. Pending material checked on
    `dev` @ 6b1b5677b2d54a0bceeb3ca3b70a0619d39bac70.
- **Other inputs:** `literature/Hidden_technical_debt_in_machine_learning_systems.pdf`, public DTU course, grade and
    evaluation pages, other web sources (accessed 2026-11-24). The other papers in `literature/` have no counterpart in
    this deck.
- **Pages:** 25 PDF pages → 25 slides. No click builds to merge and no Gamma "Error uploading image" placeholders; every
    page render was checked by eye. Glyph repairs: p. 2 "this&" and p. 25 "started&" end in an ellipsis. The stray
    letters and symbols that start bullets are emojis: 💻 and ⚡ (p. 3), ⭐ (p. 8), ⚠️ (p. 9, p. 17), 💡 (p. 10, 16, 18,
    21), ⚡ (p. 13, 23, 24), ❓ (p. 21).

## Deck outline (narrative order to keep)

1. Title: page 1
2. Who is involved (lecturer, open-source work, TAs): pages 2–4
3. Course setting and where things live (facts, webpage, Slack and DTU Learn): pages 5–7
4. What the course is, what it is not, and how it is taught: pages 8–10
5. Organization and hopes (sessions and modules, hygge): pages 11–12
6. How the days and weeks run: pages 13–14
7. Projects, passing, exam and hand-in: pages 15–18
8. The course so far (evaluation, versions, growth): pages 19–20
9. FAQ and "Can this course be vibe coded?": pages 21–22
10. Getting help (campus, Slack), then starting: pages 23–25

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "From that to this..." (p. 2)
    - "My secret identity" (p. 3)
    - "What this course is not" (p. 9)
    - "(Maybe) Learn something along the way" (p. 12)
    - "I am still learning how to do lectures" (p. 13)
    - "Lectures are not meant to give teach you anything, but provide some context to the topic of the day" (p. 13; the
        stray "give" is a typo)
    - "There has been one failure mode: group dynamics!" (p. 16). INTRODUCTION_TO_THE_COURSE-01 asks for confirmation.
    - "Some would say we are on v5.0 of the course; I would argue that we are on v0.0.5." (p. 20).
        INTRODUCTION_TO_THE_COURSE-12 proposes a change.
    - "Remember the I in LLM is for intelligence." (p. 21)
    - "Can this course be vibe coded?" (p. 22). INTRODUCTION_TO_THE_COURSE-02 changes the answer, not the question.
    - "Help us by helping each others" (p. 24)
- Voice markers:
    - Research figure next to the Sculley et al. boxes as a career change (p. 2); the boxes return on p. 9 as "what this
        course is not"
    - Exploration vs. exploitation, and great power with great responsibility, as the teaching method (p. 10)
    - Hygge as the hope for the course (p. 12)
    - Self-deprecating humour about lectures and the course version (p. 13, p. 20)
    - Passing without studying "in your imagination" (p. 16)
    - "The course in 2 statistics" (p. 19). INTRODUCTION_TO_THE_COURSE-03 updates the data only.

## Pending restructure

<!-- decisions from the restructure map that touch this deck's modules -->

-
    [Restructure the MLOps course for January 2027 to close the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/647):
    the course goal becomes closing the loop (deploy, collect, label, retrain, redeploy). The deck never mentions it.
    Which session hosts the closure is still open. → see INTRODUCTION_TO_THE_COURSE-20
- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): Day 4 becomes data
    foundations and W&B moves to Day 3. The week table on p. 14 names no topics, so it stays valid if the day pattern
    does; that map still lists the 1:1 session-to-day mapping as open.
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648),
    [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649),
    [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650),
    [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651) and
    [Find a maintained label-quality tool to replace cleanlab](https://github.com/SkafteNicki/dtu_mlops/issues/652): the
    deck names none of these tools or datasets, so no slide text changes. The p. 11 screenshot shows the "Data version
    control" module in the nav and must be retaken after it is removed.
- [Lock the M/O/A taxonomy and naming convention](https://github.com/SkafteNicki/dtu_mlops/issues/619): three module
    kinds with their own number series, so "Two types of modules" is no longer complete. Lectures do not otherwise
    reference A-modules
    ([Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681)). → see
    INTRODUCTION_TO_THE_COURSE-14
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621) and
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682): the agent use rule (skill
    pack, report answer, trace export) limits what p. 22 may promise. → see INTRODUCTION_TO_THE_COURSE-02
- [Reconcile existing agentic branch work with the A-module plan](https://github.com/SkafteNicki/dtu_mlops/issues/623):
    trace export is per student and covers the whole course, so there is more than one hand-in. → see
    INTRODUCTION_TO_THE_COURSE-09
- [Survey agent tooling options for the course](https://github.com/SkafteNicki/dtu_mlops/issues/620),
    [Define the A-module format and template](https://github.com/SkafteNicki/dtu_mlops/issues/622),
    [Decide where the A-module spec lives](https://github.com/SkafteNicki/dtu_mlops/issues/624) and
    [Assemble and publish the A-module spec](https://github.com/SkafteNicki/dtu_mlops/issues/625): no effect on this
    deck.

## Findings

<!-- max ~25; ids are stable: <DECK>-<nn>, the deck name upper-cased (e.g. CLOUD-01) -->

### INTRODUCTION_TO_THE_COURSE-01 · Pass rate does not match DTU's grade statistics

- **Category:** wrong
- **Priority:** must
- **Pages:** 16
- **Finding:** The slide says "around 99.7%". DTU's five published winter distributions (Winter 2021–2025) add up to
    1,313 passed of 1,321 who attended (99.4%), or 94.1% of 1,395 registered. All eight failures are in Winter 2024; the
    latest, Winter 2025, is 414 of 414. DTU does not publish why students failed.
- **Proposal:** Use the DTU figure, say it counts those who attended, and cite it. Confirm that "one failure mode: group
    dynamics" (keep list) still holds for the eight failures. June 2021 and the summer re-exams are not in the sum.
- **Evidence:** DTU, "Karakterfordeling" for 02476, <http://karakterer.dtu.dk/Histogram/1/02476/Winter-2021>,
    <http://karakterer.dtu.dk/Histogram/1/02476/Winter-2022>, <http://karakterer.dtu.dk/Histogram/1/02476/Winter-2023>,
    <http://karakterer.dtu.dk/Histogram/1/02476/Winter-2024>, <http://karakterer.dtu.dk/Histogram/1/02476/Winter-2025>
    (accessed 2026-11-24). Summed here: 98+192+262+347+414 of 98+192+262+355+414. The failure reason is `unverified`.

### INTRODUCTION_TO_THE_COURSE-02 · "Vibe coding is allowed" contradicts the agent use rule

- **Category:** outdated
- **Priority:** must
- **Pages:** 22
- **Finding:** The slide allows vibe coding without conditions. The spec defines vibe coding as prompting an agent
    *without* a skill pack, and any group that uses agents on the project must use one of five approved packs, commit
    its config, answer a report question about it, and hand in traces. Agents are never required.
- **Proposal:** Keep the question and the "you remain fully responsible" line. Change the answer to vibe coding versus
    agentic engineering, with the rule for the project and a link to the project page; the details belong there.
- **Evidence:** [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), "6. Agent use
    rule" and "8. Vocabulary"; Osmani, "Agentic Engineering", <https://addyosmani.com/blog/agentic-engineering/>
    (accessed 2026-11-24).

### INTRODUCTION_TO_THE_COURSE-03 · The two statistics are from January 2023

- **Category:** outdated
- **Priority:** should
- **Pages:** 19
- **Finding:** The bars (24 of 30 "completely agree"; 13/5/10 of 30 equal/more/much more time) match the January 2023
    evaluation exactly. January 2026 has 64 answers of 442 (14.48%): 58 (90.63%) completely agree they learned a lot;
    time spent is 24 equal, 23 more, 13 much more.
- **Proposal:** Re-plot both charts from the January 2026 evaluation, show the year and the response rate, and cite it.
- **Evidence:** DTU course evaluations for 02476, January 2023 <https://evaluering.dtu.dk/kursus/02476/276411> and
    January 2026 <https://evaluering.dtu.dk/kursus/02476/355600>, questions 1.1 and 2.1 (accessed 2026-11-24).

### INTRODUCTION_TO_THE_COURSE-04 · Project work is 6 of 15 days, not a third

- **Category:** wrong
- **Priority:** should
- **Pages:** 14, 15
- **Finding:** The table on p. 14 has 9 exercise days and 6 project days (60/40), but the card above it says 2/3 and 1/3
    and p. 15 says "Approximately 1/3". The time plan and project page repeat the fractions; the course description says
    "approximately 30%". Below the `must` default because the slide copies the page.
- **Proposal:** Say 9 exercise days and 6 project days and drop the fractions, once the 2027 day pattern is confirmed.
- **Evidence:** `pages/timeplan.md`, intro and week tables; `pages/projects.md`, "Project work" and "Project days"
    ("approximately be given 6 full days"); DTU course description 02476 (2026/2027),
    <https://kurser.dtu.dk/course/2026-2027/02476?lang=en-GB> (accessed 2026-11-24).

### INTRODUCTION_TO_THE_COURSE-05 · Slack channel names disagree with the pages

- **Category:** outdated
- **Priority:** must
- **Pages:** 15, 24
- **Finding:** Students are sent to `#find-a-group` (p. 15), but the project page says `#looking-for-group`. Page 24
    uses `#s1-getting-started` and `#s2-organisation-and-version-control` where the pages say `#s1` and `#s2`.
    `#zoom-online-help` and `#project` appear on no page.
- **Proposal:** Check the real channel names in the workspace, use them on the slides, and align the pages.
- **Evidence:** `pages/projects.md`, "Project work"; `pages/before.md`, "📢 Communication". The actual channel names are
    `unverified`.

### INTRODUCTION_TO_THE_COURSE-06 · TA list is for January 2026

- **Category:** outdated
- **Priority:** must
- **Pages:** 4
- **Finding:** The slide names 13 TAs and the number on duty. It is the January 2026 roster; the 2027 one is not in the
    repository.
- **Proposal:** Replace names, count and hours with the 2027 roster from the author just before the lecture.
- **Evidence:** p. 1 ("January 2026"); the 2027 roster is `unverified`.

### INTRODUCTION_TO_THE_COURSE-07 · Rooms, map and hours need the 2027 booking

- **Category:** outdated
- **Priority:** must
- **Pages:** 23
- **Finding:** "Auditorium 72", the group areas, the marked map of building 421 and the two sets of hours are for 2026.
    No page gives rooms, and the red marks sit near 035A and 027A while the map's room 072 is at the other end, so the
    map does not show what "72" means.
- **Proposal:** Replace with the 2027 booking and a map or building/room list that names the same places as the text.
- **Evidence:** p. 23 compared with its own map; the 2027 rooms and hours are `unverified`.

### INTRODUCTION_TO_THE_COURSE-08 · Slack invite must be renewed

- **Category:** outdated
- **Priority:** must
- **Pages:** 7
- **Finding:** The invite link behind the card is the same token as in `pages/before.md`, which warns that "The link may
    be expired". Joining Slack is the first thing students are asked to do.
- **Proposal:** Create a new invite, test it logged out, and use one stable link (the course page) on the slide.
- **Evidence:** `pages/before.md`, "📢 Communication"; link annotation on p. 7. Slack answered 403 to an automated fetch
    (2026-11-24), so whether the invite still works is `unverified`.

### INTRODUCTION_TO_THE_COURSE-09 · "One hand-in" misses the report and the agent traces

- **Category:** outdated
- **Priority:** must
- **Pages:** 18
- **Finding:** The slide shows one DTU Learn hand-in without a date. On `main` it is the repository link, due on the
    first project day (8 January 2027) together with a project description of at least 300 words in the repository. The
    `dev` project page adds a trace upload per student, due with the report, and the agent use rule covers every student
    who used an agent anywhere in the course.
- **Proposal:** Retitle as a hand-in overview: repository link (group, 8 January), report in the repository (group,
    final day), agent traces on DTU Learn (per student, only if agents were used).
- **Evidence:** `pages/projects.md`, "Day 1" and "Exam"; `pages/timeplan.md`, "Week 1"; `dev` `pages/projects.md`,
    "Hand-in of coding agent traces";
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), "6. Agent use rule".

### INTRODUCTION_TO_THE_COURSE-10 · The exam deadline and the automatic collection are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 17
- **Finding:** The exam slide gives no deadline. The report is due at midnight on the last day (22 January 2027 by the
    time plan), must be in `reports/README.md` on the main branch, and is collected automatically, so later changes do
    not count.
- **Proposal:** Add the deadline and the "copy the `reports` folder, it is scraped" point. The "~30 questions" still
    holds (31).
- **Evidence:** `pages/projects.md`, "Exam" and "Day 6"; `pages/timeplan.md`, "Week 3"; `reports/README.md`.

### INTRODUCTION_TO_THE_COURSE-11 · Generative AI must be declared

- **Category:** gap
- **Priority:** must
- **Pages:** 21, 22
- **Finding:** The FAQ answer is "Yes all you want". DTU permits generative AI in project reports but requires students
    to state and describe the use; not doing so counts as exam cheating. Question 31 of the report asks for it, and the
    course FAQ prefers report answers in students' own words.
- **Proposal:** Keep the LLM joke and add that use must be declared in the report (question 31) under DTU's rules.
- **Evidence:** DTU study message S1906 on generative AI in project reports,
    <https://student.dtu.dk/en/messages/studiemeddelelser/arkiv/studiemeddelelse-s1906-retningslinjer-for-brug-af-generativ-ai-i-projektrapporter-og-afsluttende-projekter?lmsgid=98eeba12-9c99-478a-aced-b0e36071efa2>
    (accessed 2026-11-24); `pages/faq.md`, "Can I use ChatGPT or similar tools"; `reports/README.md`, "Question 31".

### INTRODUCTION_TO_THE_COURSE-12 · Version joke is behind

- **Category:** outdated
- **Priority:** should
- **Pages:** 20
- **Finding:** The keep-list quote says v5.0 and v0.0.5 in the January 2026 deck. The repository has six course tags,
    June 2021 to January 2026, so January 2027 is the seventh run.
- **Proposal:** Keep the joke and bump both numbers; the author decides how runs map to versions. This changes a
    keep-list item.
- **Evidence:** Git tags `v2021june`, `v2022january` to `v2026january`, <https://github.com/SkafteNicki/dtu_mlops/tags>.

### INTRODUCTION_TO_THE_COURSE-13 · Student-number chart ends with an estimate

- **Category:** outdated
- **Priority:** should
- **Pages:** 20
- **Finding:** The 2026 point is about 450 (+20.32% on 374), made before the course. The final evaluation lists 442
    students (+18.2%); its earlier counts (196, 275, 374) match the chart. The title says "increase" but the axis is the
    number of students.
- **Proposal:** Re-plot with 442 for 2026, add 2027 from registration, and title it as number of students.
- **Evidence:** DTU course evaluations for 02476, January 2023, 2024, 2025 and 2026:
    <https://evaluering.dtu.dk/kursus/02476/276411>, <https://evaluering.dtu.dk/kursus/02476/305384>,
    <https://evaluering.dtu.dk/kursus/02476/326661>, <https://evaluering.dtu.dk/kursus/02476/355600> (accessed
    2026-11-24). The 2027 number is `unverified`.

### INTRODUCTION_TO_THE_COURSE-14 · Module kinds and labels change

- **Category:** outdated
- **Priority:** should
- **Pages:** 11
- **Finding:** "Two types of modules", all labelled M, is correct on `main`. The decided convention has three kinds with
    separate M, O and A series, and every page names its kind. "S10 is just a bunch of extra modules" still holds.
- **Proposal:** Explain the M and O labels and what core means, and let the course page explain the rest. Naming the
    third kind in one line is the author's case-by-case call under the slides rule. Use wording that does not count the
    kinds otherwise.
- **Evidence:** [Lock the M/O/A taxonomy and naming convention](https://github.com/SkafteNicki/dtu_mlops/issues/619);
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), "1. Module kinds";
    [Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681);
    `pages/before.md`, "📂 Course organization".

### INTRODUCTION_TO_THE_COURSE-15 · Course setup is skipped

- **Category:** gap
- **Priority:** should
- **Pages:** 6, 25
- **Finding:** The deck sends students straight to session 1. The Introduction page first asks them to create a course
    folder, clone the repository, and keep separate folders and environments. On `dev` the trace script collects only
    sessions started inside the course folder, so this matters from day 1.
- **Proposal:** Add one setup slide before "Lets get started" and make the final link go to the Introduction page first.
- **Evidence:** `pages/before.md`, "💻 Course setup" and "🏗️ Recommended folder structure"; `README.md`, "🔥 Where to
    start"; `dev` `pages/projects.md`, "Hand-in of coding agent traces".

### INTRODUCTION_TO_THE_COURSE-16 · Graded-pass option has a first-week deadline

- **Category:** gap
- **Priority:** should
- **Pages:** 5, 21
- **Finding:** The slide says only "Pass/not passed". Foreign students whose university needs a grade must contact the
    course responsible within the first week, and may get a short oral exam. Day 1 is when they need to hear it.
- **Proposal:** Add one FAQ item with the deadline, and a pointer for EuroTeQ students to their Slack channel.
- **Evidence:** `pages/faq.md`, "I am a foreign student and my home university doesn't accept pass/no-pass" and "I am a
    EuroTeQ student".

### INTRODUCTION_TO_THE_COURSE-17 · Streaming and recordings are not mentioned

- **Category:** gap
- **Priority:** should
- **Pages:** 10, 21
- **Finding:** The slides promise hybrid learning and allow work from home, but do not say that lectures are streamed
    and recorded or where. The FAQ says both, and the time plan links recordings from earlier years.
- **Proposal:** Add one line to the work-from-home answer, once the 2027 recording folder exists.
- **Evidence:** `pages/faq.md`, "Is it possible to attend the course fully online"; `pages/timeplan.md`, intro. The 2027
    stream link is `unverified`.

### INTRODUCTION_TO_THE_COURSE-18 · What is expected of the project

- **Category:** gap
- **Priority:** should
- **Pages:** 15, 16
- **Finding:** "Show that you can use the tools" leaves the scope open. The project page says the checklist is
    exhaustive and not all points are expected, that projects should start small, and that a bigger model is not the
    goal.
- **Proposal:** Add one line on scope to p. 16 and leave the rest to the project deck.
- **Evidence:** `pages/projects.md`, "Project days" and "Project checklist".

### INTRODUCTION_TO_THE_COURSE-19 · Expected workload is not stated

- **Category:** gap
- **Priority:** could
- **Pages:** 5, 13
- **Finding:** The deck says the workload "will depend on you". The pages say 140 hours, about 9 hours per session, with
    most students working 9–17, and 36 of 64 students in January 2026 spent more time than the 45 hours per week norm.
- **Proposal:** Put the norm next to 5 ECTS and point to the evaluation on p. 19.
- **Evidence:** `README.md`, "ℹ️ Course information"; `pages/before.md`, "📂 Course organization"; `pages/faq.md`, "I
    will be missing X days"; evaluation as in INTRODUCTION_TO_THE_COURSE-03.

### INTRODUCTION_TO_THE_COURSE-20 · Say what the course builds towards

- **Category:** addition
- **Priority:** should
- **Pages:** 8, 20
- **Finding:** The four keywords on p. 8 leave out deployment and monitoring, which its own objective text names, and
    nothing shows that the three weeks form one lifecycle. Page 20 has an uncited, low-contrast feedback-loop figure
    used for course feedback. The restructure makes closing the loop the goal.
- **Proposal:** Add the lifecycle as a small diagram on p. 8 (the course already has `figures/mlops_cycle.svg`) and one
    line on closing the loop in week 3. On p. 20 keep the course-feedback point without the uncited figure.
- **Evidence:** `pages/before.md`, "🆒 MLOps: What is it?"; `README.md` figure;
    [Restructure the MLOps course for January 2027 to close the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/647).
    The origin of the p. 20 figure is `unverified`.

### INTRODUCTION_TO_THE_COURSE-21 · Evidence for "make sure you still learn something"

- **Category:** addition
- **Priority:** could
- **Pages:** 21, 22
- **Finding:** The advice has no source. Shen and Tamkin's randomized experiments with developers learning a new library
    found that AI use impaired conceptual understanding, code reading and debugging, while three of six interaction
    patterns kept learning intact.
- **Proposal:** Cite the paper beside the advice and make the point that how AI is used matters.
- **Evidence:** Shen and Tamkin, "How AI Impacts Skill Formation", arXiv:2601.20245v3, abstract,
    <https://arxiv.org/abs/2601.20245> (accessed 2026-11-24).

### INTRODUCTION_TO_THE_COURSE-22 · The "To Vibe" definition has no source

- **Category:** outdated
- **Priority:** should
- **Pages:** 22
- **Finding:** The slide sets a definition as if quoted from a dictionary. Karpathy's post that introduced the term says
    you "forget that the code even exists", which the course's own definition now sharpens.
- **Proposal:** Replace with a cited definition: Karpathy for the origin, and the course glossary wording for how the
    term is used here.
- **Evidence:** Karpathy, 2025, <https://x.com/karpathy/status/1886192184808149383> (accessed 2026-11-24; opening only,
    the rest was truncated); [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682),
    "8. Vocabulary". The origin of the slide's wording is `unverified`.

### INTRODUCTION_TO_THE_COURSE-23 · Teaching-method images: robot cartoon and Spider-Man

- **Category:** image
- **Priority:** should
- **Pages:** 10
- **Finding:** Two images carry the point. The robot cartoon is in Berkeley's CS188 slides, which credit Klein and
    Abbeel; no reuse licence was found. The other shows a Marvel character.
- **Proposal:** Author's call: generate a new two-restaurant image for exploration vs. exploitation, and keep the
    responsibility line as text without the character.
- **Evidence:** CS188 SP25 Lecture 11, p. 1 and p. 15,
    <https://inst.eecs.berkeley.edu/~cs188/sp25/assets/lectures/cs188-sp25-lec11.pdf> (accessed 2026-11-24);
    [How should images be recreated, and what may we reuse from Gamma exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

### INTRODUCTION_TO_THE_COURSE-24 · Hopes and passing memes

- **Category:** image
- **Priority:** should
- **Pages:** 12, 16
- **Finding:** Page 12 has the "woman yelling at a cat" photo meme (cat detected as "Dog") and a hygge poster; p. 16 has
    a SpongeBob meme. All are third-party images with no provenance; the hygge and "in your imagination" jokes are on
    the keep list.
- **Proposal:** Set hygge as text, and recreate the two jokes as generated images without real people or existing
    characters. Author's call on swapping them.
- **Evidence:**
    [How should images be recreated, and what may we reuse from Gamma exports?](https://github.com/SkafteNicki/dtu_mlops/issues/705).

### INTRODUCTION_TO_THE_COURSE-25 · The deck ends on a meme

- **Category:** image
- **Priority:** should
- **Pages:** 25
- **Finding:** The last slide pairs the session link with an expanding-brain meme (learning ML/DL from university up to
    from memes). The agreed rule is that decks do not end on a meme slide. It has no "Meme of the day" heading, but it
    makes no point about starting the exercises.
- **Proposal:** Keep "Lets get started..." and the link, drop the meme. If the author wants the learning joke, it fits
    the teaching-method slide (p. 10).
- **Evidence:**
    [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710#issuecomment-7114948971),
    "Cross-deck rules".

### Also noticed (no id, not decided)

- p. 1: the title should be "Introduction to the course", January 2027.
- p. 2, p. 9: Sculley et al. Fig. 1 appears twice and is not cited; redraw once and reuse it.
- p. 3: the Lightning link ends in a zero-width space (`%E2%80%8B`) and lands on another host. Whether the part-time
    role and the list of libraries are current is `unverified`. Lightning AI merged with Voltage Park in January 2026,
    see
    <https://www.businesswire.com/news/home/20260121371691/en/Lightning-AI-and-Voltage-Park-Complete-Merger-to-Create-the-First-Cloud-Built-for-AI>
    (search excerpt only, accessed 2026-11-24).
- p. 4: the co-responsible teacher listed in `README.md` is not on "Who else to know about?".
- p. 5: prerequisites match the 2026/2027 course description; course 02456 could be named as in `README.md`.
- p. 7: Slack is a workspace with many channels, not a channel (`pages/before.md` uses the same wording).
- p. 13: "Office hour" and company talks match `pages/timeplan.md`. The time plan also has a project lecture on the
    first project day and a summary lecture on 20 January.
- p. 14: adding the dates 4–22 January 2027 and the session topics would save a click, once the layout is locked.
- p. 15: "If you Thursday still does not have a group" is 7 January 2027.
- Language: "there names" (p. 4), "principals" (p. 10), "essentially to do" (p. 11), "spend" (p. 15), "The course have"
    (p. 16), "only consist" (p. 17), "will priorities" (p. 21), "each others" (p. 24, keep list), "Lets" (p. 25).
- The restaurant analogy for MLOps and the toolbox figure in `pages/before.md` would fit p. 8 or p. 12; they may belong
    to the MLOps introduction deck instead.

## Module follow-ups (out of scope for the deck)

- `pages/projects.md`, "Exam": still says January 2026 and "the 23th"; the 2027 deadline is 22 January by the time plan.
- `pages/faq.md`, "When will the exam take place": still says January 2025, the 24th.
- `pages/timeplan.md`, intro, and `pages/projects.md`, "Project work": 2/3 and 1/3 do not match 9 and 6 days.
- `pages/projects.md`, "Project work", and `pages/before.md`, "📢 Communication": align channel names and renew the Slack
    invite (also linked from the deck).
- `pages/timeplan.md`, intro: 2027 recordings link to be added. `dev` still has the 2026 dates in its week tables, so
    the `main` update must survive the merge.
- `README.md`, "Learning objectives": the general objective differs from the official 2026/2027 course description,
    which p. 8 quotes word for word.
- `pages/faq.md`, "Can I use ChatGPT or similar tools": no mention of the declaration requirement.
- `reports/README.md`, "Overall project checklist", Week 1: profiling is cited as M12 where `pages/projects.md` says
    M13.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | Research figure: coloured branching curves with zoomed insets | chart/paper figure | author's own figure if available, or re-plot/redraw + citation (origin unverified) | | |
| 2, 9 | Sculley et al. Fig. 1, ML system boxes around a tiny "ML Code" box | chart/paper figure | re-plot/redraw + citation | | |
| 3 | GitHub profile of the lecturer (December 2023, title "Postdoc", personal email visible) | tool screenshot | fresh screenshot without the email | stale | |
| 6 | GitHub repository page, and the course site with an older navigation | tool screenshot | fresh screenshot | stale | |
| 7 | Slack logo in a link card | tool logo | Iconify logo | | INTRODUCTION_TO_THE_COURSE-08 |
| 7 | Illustration of people at a screen | decorative | replaced by theme | | |
| 10 | Robot choosing between "The Usual Place" and "Grand Opening!" | meme/joke | keep / regenerate / swap? | voice | INTRODUCTION_TO_THE_COURSE-23 |
| 10 | Spider-Man with "With great power comes great responsibility" | meme/joke | keep / regenerate / swap? | voice | INTRODUCTION_TO_THE_COURSE-23 |
| 11 | Git module page with "Core Module" box and the nav | tool screenshot | fresh screenshot (after relabelling) | stale | INTRODUCTION_TO_THE_COURSE-14 |
| 12 | Woman yelling at a cat; the cat is boxed as "Dog" | meme/joke | keep / regenerate / swap? | voice | INTRODUCTION_TO_THE_COURSE-24 |
| 12 | Hygge dictionary-entry poster | meme/joke | keep / regenerate / swap? (set as text) | voice | INTRODUCTION_TO_THE_COURSE-24 |
| 14 | Link card for the time plan | tool screenshot | plain link (no matching route) | | |
| 15 | Group of students at screens | decorative | replaced by theme | | |
| 16 | SpongeBob: "Of course you can pass your exam without studying / in your imagination" | meme/joke | keep / regenerate / swap? | voice | INTRODUCTION_TO_THE_COURSE-24 |
| 17 | Illustration of a clipboard form | decorative | replaced by theme | | |
| 18 | DTU Learn 2024 admin view with a 2022 text submission | tool screenshot | fresh screenshot in student view | stale | INTRODUCTION_TO_THE_COURSE-09 |
| 19 | Two evaluation bar charts, January 2023 | chart/paper figure | re-plot/redraw + citation | stale | INTRODUCTION_TO_THE_COURSE-03 |
| 20 | "The Feedback Loop" diagram, dark grey text on black | concept diagram | Mermaid/PlantUML/Excalidraw (origin unverified) | | INTRODUCTION_TO_THE_COURSE-20 |
| 20 | Line chart of students per year, 2021–2026 | chart/paper figure | re-plot/redraw + citation | stale | INTRODUCTION_TO_THE_COURSE-13 |
| 23 | Floor map of building 421 with three red marks | tool screenshot | fresh screenshot | stale (until 2027 rooms are confirmed) | INTRODUCTION_TO_THE_COURSE-07 |
| 25 | Expanding-brain meme, "Learning ML/DL from university" to "From memes" | meme/joke | keep / regenerate / swap? (proposal: drop) | voice | INTRODUCTION_TO_THE_COURSE-25 |
| 1–25 | Dark Gamma background art, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo ·
meme/joke · decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated ·
replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| DTU, *02476 Machine Learning Operations* (2026/2027) | <https://kurser.dtu.dk/course/2026-2027/02476?lang=en-GB> | INTRODUCTION_TO_THE_COURSE-04, p. 5 facts, p. 8 objective |
| DTU, *Karakterfordeling 02476* (Winter 2025) | <http://karakterer.dtu.dk/Histogram/1/02476/Winter-2025> | INTRODUCTION_TO_THE_COURSE-01, p. 16 pass rate |
| DTU, *Course evaluation 02476, January 2026* | <https://evaluering.dtu.dk/kursus/02476/355600> | INTRODUCTION_TO_THE_COURSE-03, -13, -19 |
| DTU, *Course evaluation 02476, January 2023* | <https://evaluering.dtu.dk/kursus/02476/276411> | INTRODUCTION_TO_THE_COURSE-03, source of the old p. 19 charts |
| DTU, *Study message S1906 on generative AI in project reports* | <https://student.dtu.dk/en/messages/studiemeddelelser/arkiv/studiemeddelelse-s1906-retningslinjer-for-brug-af-generativ-ai-i-projektrapporter-og-afsluttende-projekter?lmsgid=98eeba12-9c99-478a-aced-b0e36071efa2> | INTRODUCTION_TO_THE_COURSE-11 |
| Sculley et al. 2015, *Hidden Technical Debt in Machine Learning Systems* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Hidden_technical_debt_in_machine_learning_systems.pdf> | p. 2 and p. 9 figure (Fig. 1) |
| Shen and Tamkin 2026, *How AI Impacts Skill Formation* | <https://arxiv.org/abs/2601.20245> | INTRODUCTION_TO_THE_COURSE-21, p. 21 "make sure you still learn something" |
| Karpathy 2025, post introducing vibe coding | <https://x.com/karpathy/status/1886192184808149383> | INTRODUCTION_TO_THE_COURSE-22 |
| Osmani, *Agentic Engineering* | <https://addyosmani.com/blog/agentic-engineering/> | INTRODUCTION_TO_THE_COURSE-02 |
| Klein and Abbeel, CS188 SP25 *Reinforcement Learning II* | <https://inst.eecs.berkeley.edu/~cs188/sp25/assets/lectures/cs188-sp25-lec11.pdf> | INTRODUCTION_TO_THE_COURSE-23 |
