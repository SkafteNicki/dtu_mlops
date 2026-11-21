# Research: building and hosting Slidev decks on the course site (#704)

> Research ticket for the "replace Gamma PDFs with Slidev" plan.
> All web sources checked on 2026-11-21. Current versions at time of writing:
> [Slidev **53.1.0**](https://www.npmjs.com/package/@slidev/cli) (latest npm dist-tag, docs build v53.1.0),
> `@slidev/cli@53.1.0` requires `node >=22.22.0` (npm package metadata),
> [playwright-chromium 1.65.1](https://www.npmjs.com/package/playwright-chromium),
> Node LTS is 24.x, actions/setup-node latest v7.1.0, pnpm/action-setup latest v6.1.1.

## Recommendation

**One pnpm project inside `slides/` with one folder per deck and a shared local theme; CI builds each
deck separately with its own `--base /dtu_mlops/slides/<deck>/`, `--router-mode hash` and
`--download` (PDF generated during the build); the built decks are committed to the `gh-pages`
branch under `slides/<deck>/` — next to, not inside, the mike version directories. Lecture pages link
to the decks with absolute URLs, and the link checker gets an ignore pattern for our own Pages host.**

1. **Repo layout** — one project, several entry `.md` files, one package.json for all decks:

   ```text
   slides/
     package.json            # devDependencies: @slidev/cli, playwright-chromium
     pnpm-lock.yaml          # committed
     theme/                  # shared custom theme (layouts/ components/ styles/)
     day1_introduction_to_the_course/
       slides.md             # entry; headmatter: theme: ../theme, exportFilename, download: true
       public/               # optional deck-specific static assets
     day2_introduction_to_mlops/
       ...
   ```

   Each deck's folder is its Slidev "user root": `components/`, `layouts/`, `public/`, `styles/`,
   `snippets/` etc. resolve from the deck folder, so the 11 decks cannot collide. Shared visuals live
   in `slides/theme/` and are pulled in via `theme: ../theme` — local themes are supported
   [natively](#multi-deck-patterns) by the resolver, no npm publishing needed.

2. **Build in CI, per deck.** `slidev build` accepts multiple entry files in one invocation, but the
   `--base` value is shared by all of them, and each deck needs a *different* base
   (`/dtu_mlops/slides/<deck>/`). So build one deck per invocation in a small shell loop. Add
   `--router-mode hash` (required for GitHub Pages subdirectory hosting) and `--download` (generates
   the PDF into the deck's build output during the same command).

3. **Publish on the same Pages site, outside mike.** The site is served from the `gh-pages` branch
   that mike populates ([deploy step](../../.github/workflows/deploy_docs.yaml): `uv run mike deploy
   --push ...`). mike's own README documents cohabiting non-versioned content on that branch —
   ["CNAME (and Other Special Files)"](https://github.com/jimporter/mike/blob/master/README.md):
   *check out your `gh-pages` branch ... and commit the necessary files to the root directory*. We do
   exactly that with a `slides/` directory, after the existing mike steps. Decks end up at
   `https://skaftenicki.github.io/dtu_mlops/slides/<deck>/`, stable across all doc versions.

4. **PDFs**: `download: true` (headmatter) or `--download` (CLI) makes the build itself render a PDF
   into the deck output and adds a download button inside the deck SPA. With `exportFilename` set per
   deck the PDF also gets a stable direct URL, e.g.
   `https://skaftenicki.github.io/dtu_mlops/slides/day1_introduction_to_the_course/day1_introduction_to_the_course.pdf`.
   No separate `slidev export` step is needed, but the same Playwright requirements apply
   (see [PDF export in CI](#pdf-export-in-ci)).

### Workflow sketch

A sketch of the deploy job (for this document only — not a workflow file). Existing steps from
`.github/workflows/deploy_docs.yaml` are kept verbatim; new steps are marked.

```yaml
name: Deploy docs
on:
  push:
    branches: [main, dev]

permissions:
  contents: write

concurrency:                      # NEW: main and dev both push gh-pages; serialize
  group: deploy-docs
  cancel-in-progress: false

jobs:
  deploy:
    name: Deploy docs
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v6
        with:
          fetch-depth: 0

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true

      - name: Sync dependencies
        run: uv sync

      # ---------- NEW: build Slidev decks ----------
      - name: Setup pnpm
        uses: pnpm/action-setup@v6        # version from "packageManager" in slides/package.json

      - name: Setup Node
        uses: actions/setup-node@v7
        with:
          node-version: 'lts/*'           # Node 24; @slidev/cli@53 needs >=22.22.0
          cache: 'pnpm'
          cache-dependency-path: slides/pnpm-lock.yaml

      - name: Install Node dependencies (Chromium downloads here)
        working-directory: slides
        run: pnpm install --frozen-lockfile

      - name: Set deck base URL
        run: echo "SLIDES_BASE=/${{ github.event.repository.name }}/slides" >> "$GITHUB_ENV"

      - name: Build decks (SPA + PDF each)
        working-directory: slides
        run: |
          for deck in */; do
            name="${deck%/}"
            [ -f "$deck/slides.md" ] || continue
            pnpm exec slidev build "$deck/slides.md" \
              --base "$SLIDES_BASE/$name/" \
              --router-mode hash \
              --download \
              --out "dist/$name"
          done
      # -------------------------------------------

      - name: Configure git credentials
        run: |
          git config user.name github-actions[bot]
          git config user.email 41898282+github-actions[bot]@users.noreply.github.com

      - name: Deploy latest version (main branch)      # unchanged
        if: github.ref == 'refs/heads/main'
        run: |
          uv run mike deploy --push --update-aliases latest stable
          uv run mike set-default --push latest

      - name: Deploy dev version (dev branch)           # unchanged
        if: github.ref == 'refs/heads/dev'
        run: uv run mike deploy --push dev experimental

      - name: Publish decks to gh-pages under /slides/   # NEW
        run: |
          git fetch origin gh-pages
          git worktree add "$RUNNER_TEMP/gh-pages" gh-pages
          mkdir -p "$RUNNER_TEMP/gh-pages/slides"
          for deck in slides/dist/*/; do
            rsync -a --delete "$deck" "$RUNNER_TEMP/gh-pages/slides/$(basename "$deck")/"
          done
          git -C "$RUNNER_TEMP/gh-pages" add slides/
          git -C "$RUNNER_TEMP/gh-pages" commit -m "slides: rebuild slidev decks" || echo "no deck changes"
          git -C "$RUNNER_TEMP/gh-pages" push origin gh-pages
```

Notes on the sketch:

- `--base` must begin **and** end with a slash ([hosting docs](https://sli.dev/guide/hosting); the CLI
  exits with an error otherwise).
- The mike steps keep working unchanged; the decks step runs *after* them and pushes a second commit
  to `gh-pages`. Both branches run the same loop; `--base` does not depend on the mike version name.
- Deck folders must keep stable names, because the folder name is both the URL segment and the
  `--out` directory. Reuse the current names: `day1_introduction_to_the_course` →
  `/dtu_mlops/slides/day1_introduction_to_the_course/` (the only PDF name needing cleanup is
  `DeepLearningSoftware.pdf` → e.g. `deep_learning_software`).
- If only some decks changed, only those subfolders get re-synced. With a global
  `rsync --delete slides/dist/ .../slides/` it would be safer but rebuilds-force; either is fine, the
  per-deck loop is the incremental-friendly variant (see
  [CI time and incremental builds](#expected-ci-time-and-changed-only-rebuilds)).

## Details and evidence

### Multi-deck patterns in one repo

- **Several entries in one project is supported and documented**: `slidev build slides1.md slides2.md`
  (or a glob) builds each deck into its own folder named after the entry file ("Multiple Builds",
  [Building and Hosting](https://sli.dev/guide/hosting#multiple-builds)).
- **But multi-entry builds share one `--base`**: the CLI's `build [entry..]` command loops over the
  entry files and calls `resolveOptions({ entry, ..., base })` with the same `base` for every deck,
  writing to `out/<basename>/` (source:
  [`packages/slidev/node/cli.ts`](https://github.com/slidevjs/slidev/blob/main/packages/slidev/node/cli.ts)).
  Multiple entries are therefore for same-base deployments. Hosting each deck under a different
  subpath (`/slides/<deck>/`) requires one invocation per deck — a shell loop in CI.
- **Per-deck user root**: Slidev sets `userRoot = dirname(entry)` (source:
  [`resolver.ts`, `getRoots()`](https://github.com/slidevjs/slidev/blob/main/packages/slidev/node/resolver.ts)),
  and per the [directory-structure conventions](https://sli.dev/custom/directory-structure) the
  `components/`, `layouts/`, `public/`, `snippets/`, `styles/`, `setup/` folders are looked up from
  the user root. One folder per deck scales cleanly to 11 decks; there is no cross-deck confusion.
- **Shared theme**: theme resolution handles local paths — a theme value beginning with `.` (or
  containing `/` while not being an npm scope) is resolved relative to the entry file (source:
  `resolver.ts`, `createResolver()`). The `slidev theme eject` command likewise writes a local
  `theme: './theme'` value into the deck headmatter. So every deck sets `theme: ../theme` and the
  shared theme directory contributes layouts/components/styles to all decks (theme roots are merged
  into the deck roots in `options.ts`, `resolveOptions()`).
- **Shared public assets**: `public/` is per user root, so deck-specific assets go in
  `slides/<deck>/public/`. Images shared by several decks can be referenced with relative paths from
  the deck markdown (Vite bundles them) or duplicated. If a future need arises for *one* shared
  `public/`, a pnpm workspace with per-deck packages is the way out — not needed for 11 decks that
  mostly share the theme.
- **Precedent**: [antfu/talks](https://github.com/antfu/talks) (the Slidev author's talk archive, 60+
  decks in one repo) is a pnpm workspace ('packages: */src', `packageManager: pnpm@11.13.0`) with one
  package per deck, a single shared dependency set pinned via the pnpm `catalog:`, and
  `playwright-chromium` as a root dev dependency (source: repo's `package.json`,
  `pnpm-workspace.yaml`). That repo *commits* exported PDFs; our plan generates them in CI instead.
  We also don't need workspaces: 11 decks can share one `package.json` + `node_modules` at
  `slides/`.

### Package manager next to uv

- **Slidev is package-manager agnostic**: its export docs give pnpm/npm/yarn/bun/deno variants for
  installing `playwright-chromium` ([Exporting](https://sli.dev/guide/exporting#cli)). **pnpm** is the
  recommended choice here: committed lockfile, fast content-addressed store, first-class CI caching
  via `actions/setup-node` `cache: 'pnpm'`, and it's what upstream Slidev developers use.
- **Coexistence with uv is trivial**: the Node project lives under `slides/` with its own
  `package.json` + `pnpm-lock.yaml`; `uv sync` ignores them. Follow-up needed: add
  `slides/node_modules/` and build output (`slides/dist/`) to `.gitignore`.
- **Node version**: `@slidev/cli@53.1.0` declares `engines: { node: ">=22.22.0" }` (npm registry
  metadata). The Slidev docs' own deploy example uses `node-version: 'lts/*'`
  ([hosting docs deploy.yml](https://sli.dev/guide/hosting#github-pages)), which resolves to Node 24
  (LTS since Oct 2025, [Node release schedule](https://github.com/nodejs/release)). Either pin
  `'lts/*'` or a concrete Node 24 version.
- **pnpm ≥10 blocks dependency install scripts by default** — including `playwright-chromium`'s
  browser-download postinstall and `esbuild`'s binary install. Approval is stored via
  [`pnpm approve-builds`](https://pnpm.io/cli/approve-builds) (current pnpm writes an `allowBuilds`
  map, older versions write `onlyBuiltDependencies`, in `pnpm-workspace.yaml` or `package.json`).
  antfu/talks pins exactly: `onlyBuiltDependencies: [esbuild, playwright-chromium, unrs-resolver]`.
  **If you skip this, `--download` fails in CI with a missing-Chromium error that looks like a
  Playwright bug.** Run `pnpm approve-builds` once locally and commit the result.

### Publishing next to mkdocs and base-path mechanics

- **Where the site lives**: GitHub Pages project sites are served at
  `http(s)://<owner>.github.io/<repositoryname>`
  ([What is GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)),
  i.e. `https://skaftenicki.github.io/dtu_mlops/`. Hence per-deck base
  `/dtu_mlops/slides/<deck>/`. (Assumption: no custom domain — there is no `CNAME` in the repo.)
- **Do not merge the decks into the mkdocs `site/` dir.** The deploy job never runs `mkdocs build`
  itself — `mike deploy` builds and copies in one command, and there is no supported point to inject
  extra files between mike's internal build and its copy into a version directory. Merging before
  `mike deploy` would also require parameterizing `--base` per mike version (`/dtu_mlops/latest/...`
  vs `/dtu_mlops/dev/...`) and would duplicate every deck into every version directory.
- **The mike-documented alternative is branch cohabitation**: [mike README, "CNAME (and Other
  Special Files)"](https://github.com/jimporter/mike/blob/master/README.md): *"Some special files
  that you'd like to deploy along with your documentation ... need to go in the root directory of
  your site. There's no special handling for this in mike ... check out your gh-pages branch ... and
  commit the necessary files to the root directory."* Our `slides/` directory fits this pattern
  exactly; `mike set-default` only writes an HTML redirect at the site root and does not touch it.
  Bonus: deck URLs are version-agnostic — `latest`, `dev` and old snapshots all point at the same,
  always-fresh decks.
- **Router mode: use `hash`.** Slidev's default `routerMode` is `'history'` (source: parser config
  defaults), which needs a server-side SPA fallback for deep links such as `/slides/day1/12`:
  `slidev build` mitigates by copying `index.html` → `404.html` and writing a Netlify-style
  `_redirects` (source: `build.ts`, with the comment *"copy index.html to 404.html for GitHub
  Pages"*). Neither works on GitHub Pages: `_redirects` is Netlify-only, and a 404 page is served from
  the **site root** only ([custom 404 page docs](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-custom-404-page)),
  so a nested deck's 404.html never triggers — a deep-link refresh would land on mike's root-level
  404 instead. The CLI option exists for exactly this case: `--router-mode` — *"override routerMode
  in the built output (**hash for subdirectory deploys like GitHub Pages**; ...)"* (source: `cli.ts`,
  `build` command). With hash mode every navigation stays in the URL fragment and only
  `/slides/<deck>/` (a plain `index.html` with HTTP 200) is ever fetched. Deck links from course
  pages should therefore link to the folder URL, not to individual slides.
- **Slidev's official GitHub Pages guide** ([hosting docs](https://sli.dev/guide/hosting#github-pages))
  assumes the "GitHub Actions" Pages mode (`configure-pages` + `upload-pages-artifact` +
  `deploy-pages`, one artifact per site, and warns against branch deploys). We deliberately do *not*
  follow that guide literally: this repo uses branch-based mike deploys, and switching the whole site
  to the artifact flow is a much bigger change (see open risks).

### PDF export in CI

- **Prerequisite**: PDF/PPTX/PNG export relies on Playwright, and the
  [`playwright-chromium`](https://www.npmjs.com/package/playwright-chromium) package must be a dev
  dependency of the project ([Exporting docs](https://sli.dev/guide/exporting#cli)). Its Chromium
  browser is downloaded during `pnpm install` *iff* pnpm is allowed to run the package's postinstall
  (see the pnpm section above). Slidev exports use a local dev server under the hood (`cli.ts`
  `export` command spins one up per entry; the `download` path in `build.ts` does the same on port
  12445).
- **`--download` replaces a separate export step**: with `download: true` in the headmatter (or
  `slidev build --download`), the build renders `slidev-exported.pdf` — named
  `<exportFilename>.pdf` if configured — into the deck's build output and enables a download button
  in the SPA ([Generate PDF when Building](https://sli.dev/features/build-with-pdf),
  [CLI `--download`](https://sli.dev/builtin/cli#build); source: `build.ts` calls `exportSlides` with
  `getExportOptions(...)`). So PDFs are attached to the site for download automatically, inside the
  deck folder: yes, do it.
- **Export tuning**: `--timeout` (default 30000 ms), `--wait`, `--wait-until` and `--with-toc` apply
  to this export as well ([exporting docs](https://sli.dev/guide/exporting#options)). Bigger decks
  may need a higher timeout; the docs' troubleshooting section also covers missing emoji fonts in
  headless Linux (fix = install Google's Noto Color Emoji into `/usr/local/share/fonts` before
  exporting).
- **Caching**: cache the **pnpm store** (`actions/setup-node` with `cache: 'pnpm'` +
  `cache-dependency-path: slides/pnpm-lock.yaml`). Do **not** bother caching Playwright browsers:
  Playwright's CI guide explicitly recommends against it — *"the amount of time it takes to restore
  the cache is comparable to the time it takes to download the binaries"* and OS dependencies are not
  cacheable ([Playwright CI docs — Caching browsers](https://playwright.dev/docs/ci#caching-browsers)).
  The Chromium download (~150–200 MB) costs roughly a minute on GH runners. If a run fails to launch
  Chromium with missing-lib errors, add `pnpm exec playwright install-deps chromium`
  ([Playwright docs](https://playwright.dev/docs/browsers)); on `ubuntu-latest` it is usually not
  needed — this needs confirming on our first run.
- **Time per deck (estimate, not yet measured)**: SPA build ≈ 5–20 s per deck (Vite); PDF export
  ≈ 20–90 s per deck (headless Chromium prints each slide). For 11 decks sequentially that is
  roughly **5–15 minutes added** to the deploy job, plus ~1–3 min dependency install cold / <1 min
  warm with the pnpm cache.

### Expected CI time and changed-only rebuilds

- Full rebuild = ~5–15 min (estimate above). Two cheap levels of incrementality:
  1. **Skip everything when no deck changed.** Use
     [dorny/paths-filter](https://github.com/dorny/paths-filter) (supports `push` events on
     long-lived branches by diffing against the previous commit on that branch) on `slides/**` +
     delete the old `54 MB` of PDFs and stop shipping them: on docs-only pushes the slidev steps are
     skipped and the decks already on `gh-pages` are left untouched (mike never deletes non-version
     content; it manages only its version directories per the [mike README](https://github.com/jimporter/mike/blob/master/README.md)).
  2. **Rebuild only changed decks.** `dorny/paths-filter`'s `list-files: shell` output (or plain
     `git diff --name-only ${{ github.event.before }} ${{ github.sha }} -- slides/`) feeds the build
     loop, and the per-deck `rsync --delete` in the publish step syncs only rebuilt deck folders.
- If deck builds ever dominate, parallelize with a job matrix (one job per deck, artifact upload),
  but that is likely over-engineering for 11 decks a few times a week.

### Linking from mkdocs pages, and the link checkers

- **mkdocs build**: `check_docs.yaml` runs `uv run mkdocs build -d build --strict` on every PR to
  main, so links in course pages must not raise mkdocs warnings.
  - Relative links such as the current `[Slides](../slides/day6_continuous_integration.pdf)` only
    work today because `mkdocs.yml` uses `docs_dir: .` (same-dir plugin) and mkdocs copies
    **every non-excluded file** in the docs dir into the site ([docs: files that are not
    Markdown are copied unaltered](https://www.mkdocs.org/user-guide/writing-your-docs/#linking-to-images-and-media)) —
    including today's PDFs. Once PDFs are deleted, any leftover relative link raises a "link points
    to a file not found in the docs directory" style warning → strict PR build fails. All page
    links must be updated atomically with the PDF removal.
  - Absolute root-relative links (`/slides/...`) are a trap twice over: mkdocs docs state
    *"Using absolute paths with links is not officially supported. ... Absolute paths are not
    modified at all."* ([User Guide, Internal links](https://www.mkdocs.org/user-guide/writing-your-docs/#linking-to-pages)),
    and the browser would resolve `/slides/...` against the host root
    (`https://skaftenicki.github.io/slides/...`) — missing the `/dtu_mlops/` site prefix anyway.
  - **Recommended: full absolute URLs** in the page markdown, e.g.
    `[Slides](https://skaftenicki.github.io/dtu_mlops/slides/day6_continuous_integration/){ .md-button }`.
    mkdocs treats full http(s) URLs as external and performs none of the internal-link validation on
    them. A helper include/macro could avoid repeating the host across 21 pages.
- **markdown-link-check** (`.github/workflows/linkcheck.yaml` via
  [gaurav-nelson/github-action-markdown-link-check](https://github.com/gaurav-nelson/github-action-markdown-link-check))
  — yes, it will choke unless we act:
  - Relative links are checked **on the filesystem**: for local files the checker sets
    `baseUrl = 'file://' + dirname(file)` and the `file` protocol handler does
    `fs.access(target)` → 200 if readable, **400 if missing**; with
    `aliveStatusCodes: [0, 200]` a missing file is *dead* (sources:
    [`markdown-link-check` main](https://github.com/tcort/markdown-link-check/blob/master/markdown-link-check)
    file-input wiring, and
    [`tcort/link-check` `lib/proto/file.js`](https://github.com/tcort/link-check/blob/master/lib/proto/file.js)).
    So links to the removed PDFs fail. Interestingly, a relative link to a deck *folder* would pass
    if the folder exists in the repo (our layout has `slides/<deck>/` in-repo), but that still collides
    with mkdocs' own link handling above — don't use relative links.
  - Absolute URLs are checked **live over HTTP** and must return 200. Deck URLs exist only after the
    first post-merge deploy, so the PR-time run would 404 links to *new* decks. Add to
    `.github/linkcheck_config.json` `ignorePatterns`:
    `{ "pattern": "^https://skaftenicki\\.github\\.io.*" }` (the config already uses
    `ignorePatterns` for other hosts). Cost: links to our own site are no longer check-checked —
    acceptable, since they're checked implicitly after every deploy anyway.
- **Watch out for `docs_dir: .`**: the deck source `.md` files under `slides/` would be picked up by
  mkdocs as publishable pages ("All Markdown files included in your documentation directory will be
  rendered", [User Guide](https://www.mkdocs.org/user-guide/writing-your-docs/#file-layout)). The
  repo already has the `exclude` plugin; when the deck sources land, add `slides/*` to its globs
  (this research note included).

## Open risks and follow-ups (repo changes needed when executing the plan)

1. `mkdocs.yml`: add `slides/*` to the `exclude` plugin globs so deck sources (and this note) are not
   rendered/published as course pages.
2. `.github/linkcheck_config.json`: add the ignore pattern for our own Pages host (see above).
3. `.gitignore`: `slides/node_modules/`, `slides/dist/`.
4. `.github/workflows/deploy_docs.yaml`: extend per the sketch; add a `concurrency` group — the mike
   README warns against concurrent deploys of a branch, and both `main` and `dev` push to
   `gh-pages`.
5. Update all 21 slide links across `pages/timeplan.md`, `pages/projects.md` and the eleven
   session `README.md` files atomically with deleting the PDFs — otherwise `check_docs` (mkdocs
   `--strict`) and `linkcheck.yaml` both go red on the PR.
6. Devon/main both publish `/slides/` — last push wins. Acceptable while deck content only changes on
   `main`; if dev decks diverge, gate dev push.
7. **Timing estimates are unverified** — no build of actual course decks exists yet. Measure the
   first run; raise `--timeout` if large decks (fonts/mermaid/iframes) need it. Confirm whether
   `playwright install-deps` is needed on `ubuntu-latest`.
8. Page-parity risk: the Gamma PDFs embed the course fonts and exact layouts; Slidev renders with web
   fonts — expect visual drift, budget review time.
9. Site size: main repo loses 54 MB of PDFs, but `gh-pages` gains 11 SPAs + 11 CI-rendered PDFs
   (current PDFs are ~3–7 MB each; Slidev PDFs should be smaller — unverified). Browsers of the
   whole `gh-pages` history live with every `mike` deploy anyway.
10. Switching Pages to the "GitHub Actions" source (Slidev's documented flow) would combine mkdocs +
    decks into one artifact — bigger migration, explicitly out of scope here but the natural
    long-term endpoint if mike versioning is ever dropped.
11. pnpm version pinning: set `"packageManager": "pnpm@<version>"` in `slides/package.json` so
    `pnpm/action-setup` and local installs agree; run `pnpm approve-builds` and commit the
    generated build-script approvals.

## Sources

All URLs retrieved 2026-11-21.

Slidev (docs + source, v53.x):

- https://sli.dev/guide/hosting — build, `--base`, `--out`, multiple builds, GitHub Pages/Netlify/Vercel/Zephyr
- https://sli.dev/guide/exporting — export CLI, playwright-chromium requirement, `--wait`/`--timeout`/`--wait-until`, fonts troubleshooting
- https://sli.dev/features/build-with-pdf — `download: true` / `--download` during build
- https://sli.dev/builtin/cli — CLI commands and options (`build`, `export`, `--download`)
- https://sli.dev/custom/directory-structure — per-deck `components/`, `public/`, ... conventions
- https://github.com/slidevjs/slidev/blob/main/packages/slidev/node/cli.ts — build/export loops, shared `--base`, `--router-mode` help text, base validation
- https://github.com/slidevjs/slidev/blob/main/packages/slidev/node/options.ts — `routerMode` override, `__SLIDEV_HASH_ROUTE__`, download config handling
- https://github.com/slidevjs/slidev/blob/main/packages/slidev/node/resolver.ts — `userRoot = dirname(entry)`, local theme resolution
- https://github.com/slidevjs/slidev/blob/main/packages/slidev/node/commands/build.ts — 404.html/_redirects generation, `download` → in-build PDF export
- https://github.com/slidevjs/slidev/blob/main/packages/parser/src/config.ts (main branch, accessed via raw.githubusercontent.com) — `routerMode` default `'history'`, `download` default `false`
- https://registry.npmjs.org/-/package/@slidev/cli/dist-tags — `latest: 53.1.0`; package metadata `engines: node >=22.22.0`

GitHub Actions & Pages:

- https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages — project site at `<owner>.github.io/<repositoryname>`
- https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-custom-404-page — root 404.html semantics
- https://github.com/actions/setup-node — Node setup, `cache: pnpm` (see README §Caching packages data)
- https://github.com/pnpm/action-setup — pnpm bootstrap
- https://github.com/dorny/paths-filter — changed-file filtering incl. push events, `list-files` outputs

Playwright / pnpm:

- https://playwright.dev/docs/ci — CI patterns, "Caching browsers" is not recommended
- https://playwright.dev/docs/browsers — `playwright install --with-deps` / `install-deps`
- https://pnpm.io/cli/approve-builds — build-script approval mechanism (`allowBuilds`, legacy `onlyBuiltDependencies`)
- https://registry.npmjs.org/-/package/playwright-chromium/dist-tags — `latest: 1.65.1`

mkdocs / mike / link checking:

- https://www.mkdocs.org/user-guide/writing-your-docs/ — file layout, non-md files copied as-is, "absolute paths ... not officially supported"
- https://www.mkdocs.org/user-guide/cli/ — `--strict`
- https://github.com/jimporter/mike (README) — version directories, "CNAME (and Other Special Files)", CI guidance
- https://github.com/tcort/markdown-link-check/blob/master/markdown-link-check — local-file `baseUrl: file://<dirname>` wiring
- https://github.com/tcort/link-check/blob/master/lib/proto/file.js — filesystem link semantics (200/400, `fs.access`)
- https://github.com/gaurav-nelson/github-action-markdown-link-check — workflow action used by this repo
- Repo-internal: `.github/workflows/deploy_docs.yaml`, `.github/workflows/linkcheck.yaml`, `.github/workflows/check_docs.yaml`, `.github/linkcheck_config.json`, `mkdocs.yml`, `tasks.py`, `pyproject.toml`, `slides/` (11 Gamma PDFs, 54 MB)

Precedent:

- https://github.com/antfu/talks — multi-deck Slidev repo (pnpm workspace, catalog-pinned `@slidev/cli`, `playwright-chromium` devDep, `onlyBuiltDependencies: [esbuild, playwright-chromium, unrs-resolver]`)
- https://github.com/nodejs/release — Node release/LTS schedule
