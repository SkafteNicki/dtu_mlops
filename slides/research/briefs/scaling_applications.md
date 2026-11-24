# Review brief: scaling_applications (`slides/day12_scaling_applications.pdf`)

- **Modules compared:** `s9_scalable_applications/` (`README.md`, `data_loading.md`, `distributed_training.md`,
    `inference.md`, `exercise_files/`) @ `main` d5d24e8c3c975acc263ef79d0e1f462a3ec4ba6b. Also
    `s4_debugging_and_logging/profiling.md` (moves here), `s4_debugging_and_logging/boilerplate.md`,
    `s7_deployment/ml_deployment.md`, `s10_extra/quantization.md`, `pages/projects.md`, `pages/timeplan.md`,
    `pages/faq.md`.
- **Other inputs:** `literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf`; the other files in
    `literature/` were keyword-searched only. Web sources (accessed 2026-11-24), including the PyTorch 2.15 docs and the
    Lightning 2.6.6 docs, the current stable releases.
- **Pages:** 23 PDF pages → 23 slides (22 after SCALING_APPLICATIONS-25). There are no click builds: pp. 20 and 21 share
    a heading but not content. Three Gamma "Error uploading image" placeholders (pp. 6, 12, 15). Text extracted with
    pdftotext and links with pypdf; every page render checked by eye. Glyph repairs: the "D" and "Ì" bullets are 💡 (p. 7
    has blanks), "¦" is ⚠️, and "don9t" on p. 14 is "don't".

## Deck outline (narrative order to keep)

1. Title: page 1
2. When to scale, what distributed means, the trade-off: pages 2–4
3. Side steps: devices, communication operations, memory: pages 5–7
4. The layers you can parallelize: page 8
5. Data loading, and finding the bottleneck: pages 9–10
6. Distributed training: DP, DDP, FSDP, in PyTorch, in a framework: pages 11–15
7. Precision, very large models, compilation, small optimizations: pages 16–19
8. Inference: batching and caching: pages 20–21
9. Course admin: page 22
10. Meme of the day: page 23 (dropped, SCALING_APPLICATIONS-25)

## Keep list

<!-- fixed by default; changing an item needs a finding that points back to it -->

- Signature quotes (verbatim, with page):
    - "We should only do it when we have a working system" (p. 2)
    - "Distributed computation is not always beneficial, its a trade-off:" (p. 4)
    - "Choose one or more to parallelize over" (p. 8)
    - "If you are using a experiment logger, good chance you are getting this for free :)" (p. 10)
    - "Just don't" on model parallelism (p. 14). SCALING_APPLICATIONS-11 qualifies it.
    - "Instead use any high level framework" (p. 15)
    - "Remember to compile your model" (p. 18)
    - "Yes, the material will stay up" (p. 22)
    - "No, there is not a specific number I look for to be fulfilled" and "Yes, you need to have some cloud in the
        project" (p. 22)
- Voice markers:
    - The Knuth quote as the opening brake (p. 2). SCALING_APPLICATIONS-16 extends it.
    - The scales of communication overhead against device utilization (p. 4)
    - "A little side step" detours (pp. 5–6)
    - The data / model / experiment pyramid (p. 8)
    - The question-style headings "How do I know if I am bottle necked?" and "What about inference?" (pp. 10, 20, 21)

## Pending restructure

<!-- decisions from the restructure and agentic maps that touch this deck's modules -->

- [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653): Profiling becomes
    the first module of this session; Lightning moves to Day 1 and is optional but preferred (not required for Hugging
    Face projects); W&B moves to Day 3, so p. 10 can assume it. → see SCALING_APPLICATIONS-01, SCALING_APPLICATIONS-10
- [Relocate the current S4 modules to their new sessions](https://github.com/SkafteNicki/dtu_mlops/issues/661) (still
    **open**): executes the move, with a brief optional mention of Lightning's profiler. Baseline `main` does not have it
    yet. → see SCALING_APPLICATIONS-01
- [Lock module numbering for the new layout](https://github.com/SkafteNicki/dtu_mlops/issues/656) (still **open**): name
    modules, never number them, on the slides.
- [Design the week-3 loop closure](https://github.com/SkafteNicki/dtu_mlops/issues/655) (still **open**): likely hosted in
    monitoring, the day before this deck. Re-check before grilling. → see SCALING_APPLICATIONS-23
- [Find the best dataset for the lifecycle loop](https://github.com/SkafteNicki/dtu_mlops/issues/651): the deck names no
    dataset, so nothing changes.
- [How do modern ML pipelines version data?](https://github.com/SkafteNicki/dtu_mlops/issues/648),
    [Choose the data-engineering tool stack](https://github.com/SkafteNicki/dtu_mlops/issues/649) and
    [Decide what replaces the DVC content in each affected module](https://github.com/SkafteNicki/dtu_mlops/issues/650):
    the deck has no DVC or data-pipeline content, so nothing changes.
- [Propose the A-module topic menu, one-or-more per session](https://github.com/SkafteNicki/dtu_mlops/issues/621) and
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682): this session gets
    Autoresearch-style optimisation, and the project page gets the agent use rule. Lectures may carry agentic content
    but do not otherwise reference A-modules
    ([Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681)). → see
    SCALING_APPLICATIONS-22, SCALING_APPLICATIONS-24

## Findings

<!-- max ~25; ids are stable: SCALING_APPLICATIONS-<nn> -->

### SCALING_APPLICATIONS-01 · No profiling, though it opens the session

- **Category:** gap
- **Priority:** should
- **Pages:** 2, 10
- **Finding:** Profiling becomes the session's first module, but the deck never mentions a profiler; p. 10 only offers
    W&B system metrics. The old Day 4 deck has no profiling slide to move over either.
- **Proposal:** Add one or two slides after p. 4 on cProfile with `tottime` vs `cumtime`, then the PyTorch profiler
    (compute vs transfer). Let p. 10 follow as the cheap first signal.
- **Evidence:** [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653);
    `s4_debugging_and_logging/profiling.md`, "Profilers" and "PyTorch profiling"; text of
    `slides/day4_debugging_ML_code.pdf`.

### SCALING_APPLICATIONS-02 · The data-loading example shows the opposite of its headline

- **Category:** wrong
- **Priority:** must
- **Pages:** 9
- **Finding:** The headline prescribes `num_workers>1`, but the plot shows more workers making small batches much slower
    and large batches roughly flat. The code indexes in-memory data, exactly the case where the module says
    multiprocessing does not pay, and it never builds a `DataLoader` or sets `num_workers`.
- **Proposal:** Use a `Dataset` that reads from disk plus the `DataLoader` call, and re-plot with error bars from the
    module's LFW exercise. Keep the bad case as the illustration of the p. 4 trade-off.
- **Evidence:** Page render; `s9_scalable_applications/data_loading.md`, "A closer look at Data loading" and exercise 5.

### SCALING_APPLICATIONS-03 · Broken image: communication operations

- **Category:** image
- **Priority:** must
- **Pages:** 6
- **Finding:** The diagram is lost, so the slide is six names without meaning. The list also lacks reduce-scatter,
    which the FSDP figure on p. 13 uses.
- **Proposal:** Redraw all operations as a small rank-by-rank diagram and add reduce-scatter.
- **Evidence:** Page renders, pp. 6 and 13; PyTorch, "Getting Started with Fully Sharded Data Parallel (FSDP2)" (names
    reduce-scatter), <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html> (accessed 2026-11-24).

### SCALING_APPLICATIONS-04 · Broken image: distributed data parallel

- **Category:** image
- **Priority:** must
- **Pages:** 12
- **Finding:** The slide's only content is a Gamma error placeholder. The module has a DDP figure and the steps behind
    it; whether the lost image was that figure is `unverified`.
- **Proposal:** Redraw the module's DDP steps (local replica, non-overlapping data, forward and backward, all-reduce,
    local update) next to a matching DP drawing.
- **Evidence:** Page render; `s9_scalable_applications/distributed_training.md`, "Distributed data parallel".

### SCALING_APPLICATIONS-05 · Broken image: high-level framework

- **Category:** image
- **Priority:** must
- **Pages:** 15
- **Finding:** A placeholder sits under the Trainer code; what it showed is unknown.
- **Proposal:** Drop it, or use the space for the framework options in SCALING_APPLICATIONS-10.
- **Evidence:** Page render.

### SCALING_APPLICATIONS-06 · AMP keeps neither gradients nor underflow safe the way the slide says

- **Category:** wrong
- **Priority:** must
- **Pages:** 16
- **Finding:** The slide says AMP keeps "gradients/loss" in FP32 to prevent underflow. With FP16 forward inputs the
    backward pass produces FP16 gradients, and `GradScaler` is what prevents underflow. FP16 is the case that pairs both;
    the docs' CPU BF16 recipe uses autocast alone.
- **Proposal:** State that range-sensitive ops run in FP32, FP16 gradients are protected by loss scaling, and separate
    the FP16 recipe from BF16. Use the `torch.amp` names, since the `torch.cuda.amp` ones are deprecated.
- **Evidence:** PyTorch, "Automatic Mixed Precision package", <https://docs.pytorch.org/docs/2.15/amp.html>
    (introduction, "Gradient Scaling") (accessed 2026-11-24).

### SCALING_APPLICATIONS-07 · Removed Lightning strategies in the code

- **Category:** outdated
- **Priority:** must
- **Pages:** 17
- **Finding:** `strategy="ddp_sharded"` with FairScale is gone: Lightning's 2.0 migration guide sends users to native
    FSDP, the alias is not in the current table, and FairScale says its FSDP was upstreamed. The code would not run.
    `precision=16` is the old spelling; the module uses `"16-mixed"`.
- **Proposal:** Replace it with `strategy="fsdp"` and one DeepSpeed line, with current precision strings, as a code block.
- **Evidence:** Lightning, "Upgrade from 1.9 to 2.0",
    <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html>; Lightning, "Strategy",
    <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html>; FairScale README,
    <https://github.com/facebookresearch/fairscale> (all accessed 2026-11-24);
    `s4_debugging_and_logging/boilerplate.md`, exercise 10.

### SCALING_APPLICATIONS-08 · IPU accelerator is gone

- **Category:** outdated
- **Priority:** must
- **Pages:** 15
- **Finding:** The code and comment offer `accelerator='ipu'`. Current Lightning lists CPU, GPU, TPU and MPS on the
    accelerator page (and `"hpu"` on the Trainer page), and the migration guide covers the dropped `ipus` flag.
- **Proposal:** Drop the IPU lines; show `cpu`, `gpu`, `tpu` and `auto`, with a comment that does not promise identical
    device counts on each.
- **Evidence:** Lightning, "Accelerator", <https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html>, and
    "Trainer", <https://lightning.ai/docs/pytorch/stable/common/trainer.html> (accessed 2026-11-24).

### SCALING_APPLICATIONS-09 · DP and FSDP advice is dated

- **Category:** outdated
- **Priority:** should
- **Pages:** 11, 13, 14
- **Finding:** The table presents `DataParallel` as a simple option, while PyTorch recommends DDP even on one node and
    the module calls DP obsolete. FSDP is shown as a figure with no API, and FSDP1 is now deprecated in favour of FSDP2
    (`fully_shard`).
- **Proposal:** Mark DP as "for understanding only", and name FSDP2's `fully_shard` on p. 14.
- **Evidence:** PyTorch, "DataParallel",
    <https://docs.pytorch.org/docs/2.15/generated/torch.nn.DataParallel.html>; FSDP2 tutorial as in
    SCALING_APPLICATIONS-03 (accessed 2026-11-24); `s9_scalable_applications/distributed_training.md`, "Data parallel".

### SCALING_APPLICATIONS-10 · "Any high level framework" shows only Lightning

- **Category:** addition
- **Priority:** should
- **Pages:** 15
- **Finding:** The heading says any framework, but the slide shows only Lightning. The restructure makes Lightning
    optional because Hugging Face projects should not need it.
- **Proposal:** Keep the Trainer code and add one line on the Hugging Face route for groups that use those libraries.
- **Evidence:** [Decide where each current Day 4 module goes](https://github.com/SkafteNicki/dtu_mlops/issues/653),
    "Lightning is optional but preferred"; `s9_scalable_applications/distributed_training.md`, "Distributed data
    parallel" exercise 3. The exact Hugging Face API to show is `unverified`.

### SCALING_APPLICATIONS-11 · "Model parallelism: just don't" needs a qualifier

- **Category:** outdated
- **Priority:** could
- **Pages:** 14, 18
- **Finding:** This is a keep-list joke, but p. 18's LLaMA 70B curve says "w/TP": its source uses tensor parallelism
    across 8 A100s, and Lightning now documents tensor parallelism. The deck warns against what its own chart relies on.
- **Proposal:** Keep the line and add "by hand": sharding and tensor parallelism are for frameworks to do.
- **Evidence:** PyTorch Blog, "PyTorch compile to speed up inference on Llama 2",
    <https://pytorch.org/blog/pytorch-compile-to-speed-up-inference/>; Lightning, "Model parallel" docs,
    <https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/> (accessed 2026-11-24).

### SCALING_APPLICATIONS-12 · Launch with torchrun, and use the module's steps

- **Category:** drift
- **Priority:** should
- **Pages:** 14
- **Finding:** The slide sets `MASTER_ADDR`/`MASTER_PORT` and uses `mp.spawn`. The module's example runs under
    `torchrun`, reads `LOCAL_RANK` and uses a `DistributedSampler`, which the slide omits.
- **Proposal:** Replace the recipe with the exercise file's: `torchrun`, `init_process_group`, `DistributedSampler`, DDP
    wrapper, imported as a code block.
- **Evidence:** `s9_scalable_applications/exercise_files/distributed_example.sh` and `distributed_example.py`; PyTorch,
    "torchrun", <https://docs.pytorch.org/docs/2.15/elastic/run.html> (accessed 2026-11-24).

### SCALING_APPLICATIONS-13 · Device memory table and the bandwidth claim

- **Category:** wrong
- **Priority:** must
- **Pages:** 7
- **Finding:** "Faster data transfer" does not follow from more memory: Google lists capacity and bandwidth as separate
    specs, and TPU v6e has about a third of v5p's capacity but well over half its bandwidth. The GPU maximum (140 GiB) is
    now 288 GB on HGX B300, and a "32 Tib" TPU cannot be one chip (192 GiB on TPU7x), so the column mixes chip and pod.
- **Proposal:** Say memory decides what fits, bandwidth how fast it moves; give per-device figures with date and unit.
    "34-64" and "higher data modalities" look like slips to fix too (`unverified` what was meant).
- **Evidence:** Google Cloud, "TPU7x", <https://docs.cloud.google.com/tpu/docs/tpu7x>; NVIDIA, "Components, HGX AI
    Factory", <https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html> (search
    excerpt only) (both accessed 2026-11-24).

### SCALING_APPLICATIONS-14 · Parallel-operation counts compare different things

- **Category:** outdated
- **Priority:** should
- **Pages:** 5
- **Finding:** The CPU range stops at 128, while a current EPYC 9965 has 192 cores and 384 threads. The TPU range matches
    matrix-unit operations per cycle, which is not the same unit as threads. "Rendering unit" undersells a GPU in a
    course where it is the training device.
- **Proposal:** State one unit per row, refresh the CPU figure and describe the GPU by what it does here.
- **Evidence:** AMD, "EPYC 9965",
    <https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9965.html>; Google Cloud, "TPU
    architecture", <https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm> (accessed 2026-11-24);
    `s9_scalable_applications/data_loading.md`, "A closer look at Data loading" (one operation per thread).

### SCALING_APPLICATIONS-15 · Xception chart is uncited and old

- **Category:** outdated
- **Priority:** could
- **Pages:** 7
- **Finding:** The chart is a pasted "Figure 3" with no source, hardware or date; one search did not find the original.
    Its out-of-memory bars are what the slide needs.
- **Proposal:** Find the source and re-plot, or replace it with a simple "does batch size N fit" illustration.
- **Evidence:** Page render; origin `unverified`.

### SCALING_APPLICATIONS-16 · Knuth's next sentence is the reason to profile

- **Category:** addition
- **Priority:** could
- **Pages:** 2
- **Finding:** The quote is accurate but stops before "Yet we should not pass up our opportunities in that critical 3%",
    which is what leads into profiling.
- **Proposal:** Cite Knuth (1974) and add that sentence as the bridge to SCALING_APPLICATIONS-01. Keep-list item.
- **Evidence:** Wikiquote, "Donald Knuth", quoting "Structured Programming with Goto Statements", doi:10.1145/356635.356640,
    <https://en.wikiquote.org/wiki/Donald_Knuth> (accessed 2026-11-24); the paper itself was not fetched.

### SCALING_APPLICATIONS-17 · Inference methods from the module are missing

- **Category:** gap
- **Priority:** should
- **Pages:** 16, 20, 21
- **Finding:** The module teaches architecture choice, quantization, pruning and knowledge distillation; only
    quantization is in the deck, and under training-time optimizations. The module also stresses that pruning does not
    give speedups by itself.
- **Proposal:** Open the inference section with a slide that lists the four, and place p. 16's quantization half there.
- **Evidence:** `s9_scalable_applications/inference.md`, "Choosing the right architecture", "Pruning", "Knowledge
    distillation"; `s9_scalable_applications/README.md`, learning objectives.

### SCALING_APPLICATIONS-18 · Batch endpoint loses the order of its answers

- **Category:** wrong
- **Priority:** must
- **Pages:** 20
- **Finding:** `list(set(items))` and the validity filter change order and length, so the caller cannot match predictions
    to inputs. A comment is cut off at the image edge. The deployment module teaches adaptive batching, where the server
    forms the batches.
- **Proposal:** Replace the screenshots with a short code block that keeps the input-output mapping, and recall adaptive
    batching as the server-side version.
- **Evidence:** Page render; Python docs, "Set types" (unordered collection),
    <https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset> (`unverified`: not fetched);
    `s7_deployment/ml_deployment.md`, adaptive batching exercise.

### SCALING_APPLICATIONS-19 · Fibonacci is not an inference cache

- **Category:** drift
- **Priority:** should
- **Pages:** 21
- **Finding:** Caching is not in the module, and the example only shows memoized recursion; the timing is mostly a cache
    hit. `lru_cache` needs hashable arguments and is bounded, so it does not transfer directly to model inputs.
- **Proposal:** Keep the "use caching if possible" point but show a prediction cache: what the key is, and why a new
    model version must invalidate it. Mention prefix caching as the LLM-serving version.
- **Evidence:** Python docs, "functools.lru_cache", <https://docs.python.org/3/library/functools.html>; vLLM docs,
    "Automatic Prefix Caching", <https://docs.vllm.ai/en/latest/> (accessed 2026-11-24); no match for caching in
    `s9_scalable_applications/`. The invalidation advice is `unverified`.

### SCALING_APPLICATIONS-20 · torch.compile without its costs, and not in the module

- **Category:** drift
- **Priority:** should
- **Pages:** 18
- **Finding:** "Remember to compile" has no caveats, and no module page covers it, although the checklist asks for
    "quantization, compilation and pruning". Failed guards trigger recompiles, with eager fallback after the limit, and
    graph breaks lose optimizations. The chart is an uncited 2023-era Llama 2 benchmark.
- **Proposal:** Keep the headline, add compile cost, recompilation and "measure after warm-up", and cite or replace the
    chart.
- **Evidence:** PyTorch, "torch.compile", <https://docs.pytorch.org/docs/2.15/generated/torch.compile.html>; "Introduction
    to torch.compile", <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>; blog as in
    SCALING_APPLICATIONS-11 (accessed 2026-11-24); `pages/projects.md`, "Week 3".

### SCALING_APPLICATIONS-21 · Model-growth chart stops at 2021

- **Category:** outdated
- **Priority:** should
- **Pages:** 17
- **Finding:** The chart ends with Megatron-Turing NLG (530B) near the end of 2021 and is uncited. DeepSpeed is still
    active (news to May 2026), so the text link holds.
- **Proposal:** Re-plot with a current cited dataset, or drop it and keep the DeepSpeed point.
- **Evidence:** Page render; DeepSpeed README, "Latest News", <https://github.com/deepspeedai/DeepSpeed> (accessed
    2026-11-24). Chart origin `unverified` (Microsoft Research page returned 403).

### SCALING_APPLICATIONS-22 · Agents as optimizers of training code

- **Category:** addition
- **Priority:** should
- **Pages:** 17, 19
- **Finding:** The deck lists manual optimizations only. The autoresearch pattern gives an agent one file, a fixed
    five-minute budget and one metric, and keeps only changes that improve it. It fits a session that measures first.
- **Proposal:** Add one slide on the loop, with the metric and budget as the guardrails, and the warning that its README
    runs the agent with permissions disabled. Do not reference the A-module.
- **Evidence:** karpathy/autoresearch README, <https://github.com/karpathy/autoresearch> (accessed 2026-11-24);
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682);
    [Update lecture slides for the January 2027 course](https://github.com/SkafteNicki/dtu_mlops/issues/681).

### SCALING_APPLICATIONS-23 · Admin slide carries 2026 logistics

- **Category:** outdated
- **Priority:** must
- **Pages:** 1, 22
- **Finding:** "Fully online as I will not be at campus" and the auditorium renovation belong to January 2026. The 2027
    plan has only a summary lecture on Wednesday 20 January and no lectures on the 21st and 22nd. The title still says
    "Day12" and January 2026.
- **Proposal:** Keep the Wednesday to Friday structure, drop the renovation and let the author confirm the format of the
    summary lecture. Title becomes topic-only, January 2027.
- **Evidence:** `pages/timeplan.md`, "Week 3"; room and format for 2027 are `unverified` (ask the author).

### SCALING_APPLICATIONS-24 · Admin slide lacks the deadline and the agent hand-in

- **Category:** gap
- **Priority:** must
- **Pages:** 22
- **Finding:** This is the last full lecture, yet the slide has no hand-in time. The report is due at midnight on the
    final day, Friday 22 January 2027. Groups that use agents must also follow the agent use rule and hand in a trace
    export. "You need to have some cloud" is stated only here, not on the project page.
- **Proposal:** Add the deadline, that the main branch is scraped, and one line on the agent hand-in. Author to confirm
    the cloud requirement; both keep-list lines stay.
- **Evidence:** `pages/projects.md`, "Day 6", "Exam" and "Project checklist"; `pages/timeplan.md`, "Week 3";
    [Spec: Agentic modules (A-modules)](https://github.com/SkafteNicki/dtu_mlops/issues/682), section 6.

### SCALING_APPLICATIONS-25 · Drop the meme, keep the exercise link

- **Category:** structure
- **Priority:** must
- **Pages:** 23
- **Finding:** The deck ends on "Meme of the day", which no deck does any more. The slide also holds the deck's only
    link to the session's exercises.
- **Proposal:** Drop the slide and move the exercise link to the end of the content. The joke (train on a planet with
    slow time) makes no point the deck needs.
- **Evidence:** [Which Day 7 Cloud review changes do we accept?](https://github.com/SkafteNicki/dtu_mlops/issues/710)
    (cross-deck rule 2); page render.

### Also noticed (no id, not decided)

- p. 8 promises an "Experiment" layer that no slide follows up. A line on sweeps would close it
    (`s9_scalable_applications/README.md` points to `s10_extra/hyperparameters.md`).
- p. 9/19: `pin_memory` is in `data_loading.md` exercise 7 but not in the deck; p. 19's `prefetch_factor` already
    defaults to 2 with workers (<https://docs.pytorch.org/docs/2.15/data.html>, accessed 2026-11-24).
- p. 16 covers INT8 only, with PTQ always calibrated; `s10_extra/quantization.md` uses INT4 and `inference.md` points to
    torchao, which no slide names.
- p. 19 (async checkpoints, DALI, data formats) is drift: none of it is in `s9_scalable_applications/`. As one overview
    slide it is probably fine.
- Literature: "Keeping GPUs Warm" (Shankar et al., §5.2.2) could back p. 4's device utilization side.
- The W&B link on p. 10 now redirects to <https://docs.coreweave.com/products/wandb/ref/python/experiments/system-metrics>
    and works (accessed 2026-11-24). <https://evaluering.dtu.dk/> redirects to the DTU login.
- Typos: "its a trade-off" (p. 4, keep list), "take away" (p. 4), "bottle necked" (p. 10), "Large models that other
    methods" (p. 14), "a experiment logger" (p. 10, keep list), "Pytorch" (pp. 9, 14, 18).

## Module follow-ups (out of scope for the deck)

- `s9_scalable_applications/data_loading.md`, "A closer look at Data loading": calls workers threads; PyTorch starts
    worker *processes* (<https://docs.pytorch.org/docs/2.15/data.html>, accessed 2026-11-24).
- `s9_scalable_applications/data_loading.md`, exercise 7: ties `pin_memory` to the dataset fitting in GPU memory; it is
    about host-to-GPU copy speed (same source).
- `s9_scalable_applications/inference.md`: no section on compilation, though the Week 3 checklist asks for it.
- `s9_scalable_applications/inference.md`, introduction: says inference can't use multiple GPUs "at least not in any
    efficient way"; the Llama 2 blog in SCALING_APPLICATIONS-11 does exactly that for one request.
- `s9_scalable_applications/distributed_training.md`, introduction: links an external sharding blog and the FSDP1 page;
    point to the FSDP2 tutorial.
- `s9_scalable_applications/README.md`: says "M14 on boilerplate" (the checklist has Lightning as M15); the reference
    also goes stale with the move.
- `pages/projects.md`, "Exam": still 23 January 2026. `pages/faq.md`, "When will the exam take place": 24 January 2025.

## Images

| Page | Shows | Type | Route | Flags | Finding |
| --- | --- | --- | --- | --- | --- |
| 2 | Portrait of Donald Knuth with the quote | photo | licensed stand-in (or quote as text) | | SCALING_APPLICATIONS-16 |
| 4 | Line drawing of a balance scale | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 5 | CPU, GPU (Titan Z) and TPU photos above scalar/vector/tensor grids | concept diagram | Mermaid/PlantUML/Excalidraw | stale | SCALING_APPLICATIONS-14 |
| 6 | Gamma "Error uploading image" | — | propose replacement: redrawn communication operations | broken | SCALING_APPLICATIONS-03 |
| 7 | Bar chart: Xception training time on CPU, GPU and TPU | chart/paper figure | re-plot/redraw + citation | stale | SCALING_APPLICATIONS-15 |
| 8 | Data / model / experiment pyramid | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 9 | Line plot: total time vs `num_workers` per batch size | chart/paper figure | re-plot/redraw (own LFW run) | | SCALING_APPLICATIONS-02 |
| 9 | Dataset class | code/config screenshot (already text) | code block from exercise files | | SCALING_APPLICATIONS-02 |
| 10 | W&B system metrics panels | tool screenshot | fresh screenshot | | |
| 11 | Data parallel, forward and backward in four steps | concept diagram | redraw (same figure as `figures/data_parallel.png`, by file name and module context) | | SCALING_APPLICATIONS-09 |
| 12 | Gamma "Error uploading image" | — | propose replacement: redrawn DDP steps | broken | SCALING_APPLICATIONS-04 |
| 13 | FSDP workflow, two instances (clipped box labels) | concept diagram | redraw + citation (origin `unverified`) | | SCALING_APPLICATIONS-09 |
| 15 | Lightning Trainer accelerators | code/config screenshot | code block | stale | SCALING_APPLICATIONS-08 |
| 15 | Gamma "Error uploading image" | — | drop or framework options | broken | SCALING_APPLICATIONS-05 |
| 17 | Model size 2018–2022, log scale | chart/paper figure | re-plot/redraw + citation, or drop | stale | SCALING_APPLICATIONS-21 |
| 17 | FairScale and DeepSpeed Trainer strategies | code/config screenshot | code block | stale | SCALING_APPLICATIONS-07 |
| 18 | Llama 2 latency: unoptimized, SDPA, compile, both | chart/paper figure | re-plot/redraw + citation | stale | SCALING_APPLICATIONS-20 |
| 18 | Operator graph before and after TorchInductor fusion | concept diagram | Mermaid/PlantUML/Excalidraw | | |
| 20 | Two FastAPI endpoints, single and batch | code/config screenshot | code block | | SCALING_APPLICATIONS-18 |
| 21 | Cached Fibonacci and its timings | code/config screenshot | code block | | SCALING_APPLICATIONS-19 |
| 23 | Film still: astronauts, "one year on this planet is 7 years on Earth" | meme/joke | swap: drop (cross-deck rule) | voice | SCALING_APPLICATIONS-25 |
| 1–23 | Dark Gamma background art, "Made with Gamma" badge | decorative | replaced by theme | | |

Types: concept diagram · code/config screenshot · tool logo · tool screenshot · chart/paper figure · photo ·
meme/joke · decorative.
Routes: Mermaid/PlantUML/Excalidraw · code block (from exercise files where possible) · Iconify logo ·
fresh screenshot · re-plot/redraw + citation · xkcd with credit link · licensed stand-in · generated ·
replaced by theme.
Flags: broken · voice · stale.

## Candidate references

| Citation | URL | Supports |
| --- | --- | --- |
| Knuth 1974, *Structured Programming with go to Statements* | <https://doi.org/10.1145/356635.356640> | SCALING_APPLICATIONS-16, p. 2 quote |
| PyTorch, *Automatic Mixed Precision package* | <https://docs.pytorch.org/docs/2.15/amp.html> | SCALING_APPLICATIONS-06, p. 16 |
| PyTorch, *DataParallel* | <https://docs.pytorch.org/docs/2.15/generated/torch.nn.DataParallel.html> | SCALING_APPLICATIONS-09, p. 14 |
| PyTorch, *Getting Started with FSDP2* | <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html> | SCALING_APPLICATIONS-03, SCALING_APPLICATIONS-09 |
| PyTorch, *torchrun* | <https://docs.pytorch.org/docs/2.15/elastic/run.html> | SCALING_APPLICATIONS-12 |
| PyTorch, *torch.utils.data* | <https://docs.pytorch.org/docs/2.15/data.html> | Also noticed, module follow-ups |
| PyTorch, *torch.compile* | <https://docs.pytorch.org/docs/2.15/generated/torch.compile.html> | SCALING_APPLICATIONS-20 |
| PyTorch, *Introduction to torch.compile* | <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html> | SCALING_APPLICATIONS-20 |
| PyTorch Blog, *PyTorch compile to speed up inference on Llama 2* | <https://pytorch.org/blog/pytorch-compile-to-speed-up-inference/> | SCALING_APPLICATIONS-11, SCALING_APPLICATIONS-20, p. 18 chart |
| PyTorch Blog, *Introducing PyTorch Fully Sharded Data Parallel (FSDP) API* | <https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/> | p. 14 claim that DDP needs more memory (full copy per GPU) |
| Lightning, *Upgrade from 1.9 to 2.0* | <https://lightning.ai/docs/pytorch/stable/upgrade/from_1_9.html> | SCALING_APPLICATIONS-07, SCALING_APPLICATIONS-08 |
| Lightning, *Strategy* | <https://lightning.ai/docs/pytorch/stable/extensions/strategy.html> | SCALING_APPLICATIONS-07 |
| Lightning, *Accelerator* | <https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html> | SCALING_APPLICATIONS-08 |
| FairScale README | <https://github.com/facebookresearch/fairscale> | SCALING_APPLICATIONS-07 |
| DeepSpeed README | <https://github.com/deepspeedai/DeepSpeed> | SCALING_APPLICATIONS-21, p. 17 |
| Google Cloud, *TPU7x* | <https://docs.cloud.google.com/tpu/docs/tpu7x> | SCALING_APPLICATIONS-13 |
| Google Cloud, *TPU architecture* | <https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm> | SCALING_APPLICATIONS-14 |
| NVIDIA, *HGX AI Factory components* | <https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html> | SCALING_APPLICATIONS-13 |
| AMD, *EPYC 9965* | <https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9965.html> | SCALING_APPLICATIONS-14 |
| Python docs, *functools* | <https://docs.python.org/3/library/functools.html> | SCALING_APPLICATIONS-19 |
| vLLM docs | <https://docs.vllm.ai/en/latest/> | SCALING_APPLICATIONS-19 |
| Karpathy, *autoresearch* | <https://github.com/karpathy/autoresearch> | SCALING_APPLICATIONS-22 |
| Shankar, Garcia, Hellerstein, Parameswaran, *Operationalizing Machine Learning: An Interview Study* | <https://github.com/SkafteNicki/dtu_mlops/blob/main/literature/Operationalizing_Machine_Learning_An_Interview_Study.pdf> | Also noticed, p. 4 device utilization |
