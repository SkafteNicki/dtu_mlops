# Research: a maintained label-quality tool for the data-engineering day

Ticket: [#652](https://github.com/SkafteNicki/dtu_mlops/issues/652) (part of map
[#647](https://github.com/SkafteNicki/dtu_mlops/issues/647)). Researched 2026-10-21.

## Answer

**Keep cleanlab (`cleanlab[datalab]`, Datalab API) for the sub-exercise, pinned to `==2.9.0`.** The upstream repo
has gone quiet: no maintainer commits or reviews for about 9 months, which matches the ticket. Still, it is the only
Apache-2.0, pip-installable, CPU-only tool here that does what the exercise needs: flag likely mislabeled samples
from out-of-sample predicted probabilities. Its last release (Jan 2026) was a dependency-modernisation release, and it
installs and runs cleanly on Python 3.13 with numpy 2.5 / scikit-learn 1.9 (smoke test below). The only maintained
alternative that does the same job is FiftyOne Brain's `compute_mistakenness`, but it brings a much heavier stack
(bundled MongoDB, web app, 50+ dependencies) and would turn a one-hour sub-exercise into a FiftyOne tutorial.

Fallback if cleanlab breaks: FiftyOne (`fiftyone` + `fiftyone-brain`, Apache-2.0, very active).

## Is cleanlab maintained?

All data comes from the GitHub API (`gh api`) and PyPI JSON, retrieved 2026-10-21.

| Signal | Value | Source |
| --- | --- | --- |
| Last commit on default branch | 2026-01-13 ("Update CI Build (#1297)") | `gh api repos/cleanlab/cleanlab/commits` |
| Last release | v2.9.0, 2026-01-13 (PyPI upload same day); v2.8.0 2026-01-08; before that v2.7.1 2025-02-27 | `gh api repos/cleanlab/cleanlab/releases`, <https://pypi.org/pypi/cleanlab/json> |
| Last merged PR | 2026-01-13 (all five most recent merges by one maintainer, `ulya-tkch`) | `gh pr list -R cleanlab/cleanlab --state merged` |
| Last maintainer (MEMBER) comment | 2026-01-08 (`jwmueller`) | `gh api repos/cleanlab/cleanlab/issues/comments` filtered on `author_association` |
| Open PRs | 56, many from Sep–Oct 2026 with only CLA-bot replies | `gh pr list --state open`; recent comments are all `author_association: NONE` |
| Open issues + PRs | 136 | `gh api repos/cleanlab/cleanlab` |
| Archived | No | same |
| Stars / license | ~11.7k / Apache-2.0 | same |
| Python | `>=3.10`; core deps only numpy, scikit-learn, tqdm, pandas, termcolor | PyPI `requires_python`, `requires_dist` |

The v2.9.0 release notes are titled "Dependency Modernization and Simplified Maintenance". They remove
TensorFlow/Keras support "reducing maintenance burden", and add "CI freshness checks" for dependencies
(<https://github.com/cleanlab/cleanlab/releases/tag/v2.9.0>). That reads like the library being put into low-maintenance
mode, not dropped: the company has moved on to other products, but they left the OSS package installable on current
dependencies.

**Verdict:** dormant, not abandoned. Outside contributors still open PRs and nobody reviews them. It is fine for a pinned
course exercise. It is a risk if we need upstream fixes.

### Smoke test (run 2026-10-21)

PEP 723 script run with `uv run --no-project --python 3.13`, deps `cleanlab[datalab]==2.9.0`, `scikit-learn`. It uses
sklearn digits with 90 labels flipped on purpose, 5-fold `cross_val_predict` logistic-regression probabilities, then
`Datalab(...).find_issues(pred_probs=..., features=X)`.

```text
cleanlab 2.9.0 numpy 2.5.4 sklearn 1.9.1
flagged 309 true flips caught 80 / 90
issue types run: null, label, outlier, near_duplicate, non_iid, class_imbalance, underperforming_group
```

It installed 40 packages in about 1 s from cache and ran on CPU in seconds. It caught 80 of the 90 injected errors.
Precision is low because a weak linear model on digits is used, which is a teaching point in itself. On Python 3.13 it
works with no workarounds.

## Candidates compared

| Tool | Finds *label* errors? | Maintained? (last commit / release) | License | Install / CPU | Fit for ~1 h sub-exercise |
| --- | --- | --- | --- | --- | --- |
| **cleanlab / Datalab** | Yes (confident learning on `pred_probs`); also outliers, near-dups, non-iid, imbalance | Dormant: 2026-01-13 / v2.9.0 2026-01-13 | Apache-2.0 | `uv add "cleanlab[datalab]"`, light, CPU | **Best fit.** Framework-agnostic, works with any model's probabilities, so it slots in after students train their model |
| cleanvision (cleanlab) | No. Image-property issues only (blur, dark, duplicates, odd aspect ratio) | 2026-01-08 / v0.3.7 2026-01-05; prior release 2024-02 | Apache-2.0 | pip, CPU | Nice optional add-on (`cleanlab[image]` pulls it in), but does not answer "which labels are wrong" |
| FiftyOne + FiftyOne Brain | Yes: `compute_mistakenness(samples, pred_field, label_field)` scores label mistakes from predictions; also `compute_near_duplicates`, `compute_uniqueness`, `compute_hardness`, `compute_leaky_splits` | Very active: commits 2026-10-17 / fiftyone v1.22.1 2026-09-29; brain commits 2026-10-01 / v0.27.0 2026-09-30 | Apache-2.0 (both) | pip, CPU OK, but heavy: requires `fiftyone-db` (bundled MongoDB), `mongoengine`, `pymongo`, `opencv`, 52 direct deps | Viable, but students have to learn the FiftyOne dataset/field model and app first. Too much for a sub-exercise |
| fastdup (Visual Layer) | Partly: duplicates, outliers, "wrong label" via similarity | Low: 2026-08-23 (notebook cleanup), last GitHub release 2024-06; PyPI 2.54 2026-08-23 | **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0** (repo LICENSE), PyPI says "Non commercial" | `requires_python <3.13`, binary wheels | **Reject**: not OSS (non-commercial, no-derivatives) and no Python 3.13 support |
| Lilac | Text-focused data curation | **Archived** (databricks/lilac, last push 2024-03) | Apache-2.0 | n/a | Reject |
| doubtlab (koaning) | Yes, simple "reasons to doubt" heuristics | Last push 2024-07 | MIT | pip, light | Reject: less maintained than cleanlab |
| Renumics Spotlight | Visual browser for data/model issues; can show cleanlab outputs | Active (commits 2026-10-21) | MIT | pip | A viewer, not a detector. Optional for visualising results |

Sources: `gh api repos/<owner>/<repo>` (`pushed_at`, `archived`, `license`), `.../commits`, `.../releases`;
PyPI JSON API (`https://pypi.org/pypi/<pkg>/json`) for latest version, upload time, `requires_python`, `requires_dist`;
fastdup license text from `gh api repos/visual-layer/fastdup/license`; FiftyOne Brain function list and
`compute_mistakenness` signature/docstring from
<https://github.com/voxel51/fiftyone-brain/blob/develop/fiftyone/brain/__init__.py>.

## Suggested shape of the sub-exercise

1. Take the course's existing image-classification dataset (e.g. corrupted MNIST) and flip around 5% of the training
   labels on purpose.
2. Get out-of-sample `pred_probs` with K-fold cross-validation. A small CNN works, or for speed a logistic regression on
   flattened pixels or frozen embeddings.
3. Run `Datalab(data, label_name=...).find_issues(pred_probs=..., features=...)` and `lab.report()`.
4. Measure precision/recall against the injected flips, look at the top-ranked images, and remove or relabel them,
   then retrain and compare accuracy.
5. Optional: `cleanlab[image]` adds cleanvision image checks (`find_issues(image_key=...)` on a HF `datasets` object).

Pin `cleanlab[datalab]==2.9.0` in the exercise requirements. Because upstream is dormant, a future numpy/sklearn/pandas
major release could break it without an upstream fix, so the pin plus the course's Python `<3.14` cap is the safeguard.
Re-check before each course run. If it breaks, switch to FiftyOne Brain `compute_mistakenness`.

## Open points / not verified

- I did not install or run FiftyOne. The heaviness claim rests on its declared dependencies, not a measured install.
- I did not check how cleanlab 2.9.0 behaves on Python 3.14. The course caps at `<3.14`, so it does not matter for now.
- The "company pivot" reading of v2.9.0 is my interpretation of the release notes and activity pattern. Cleanlab has
  not published any deprecation notice for the OSS package.
