# PROTOTYPE: Slidev theme variants (throwaway)

Prototype for the wayfinder ticket
[What should the Slidev theme look like?](https://github.com/SkafteNicki/dtu_mlops/issues/708).
Three structurally different variants of one theme in `slides/theme/`, shown on the Day 7 pilot deck (moved to
`slides/cloud/` per the layout decision). This branch is never merged; only the chosen direction is rebuilt properly.

```bash
cd slides
npx pnpm@10.34.9 install
npx pnpm@10.34.9 dev:cloud           # http://localhost:3030/?variant=D  (or A, B, C)
```

Switch variants with the yellow pill at the bottom, or the `[` and `]` keys (arrow keys belong to Slidev). The pill only
shows on the dev server. Headless export of one variant:
`VITE_MLOPS_VARIANT=B npx pnpm@10.34.9 exec slidev export cloud/slides.md --format png`.

| Key | Name                 | Look                                                                                   | Structure                                                                                                         |
| --- | -------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| D   | **Chosen**           | A's look and Barlow; accent DTU red `#990000` (light) / `#e83f48` (dark, WCAG AA)       | Follows Slidev's dark/light toggle; DTU logo top-right on every slide; page number; no section numbers           |
| A   | Gamma dark           | `#161415` background, `#e5e0df` text, coral `#f65f62` accent, Barlow (all from the PDFs) | No chrome except a small page number; left-aligned cover; red-bar quote                                           |
| B   | DTU corporate        | White, DTU red `#990000`, Arial (DTU's digital fallback for Neo Sans)                    | Red rule + logo header, footer with course, deck and page; split cover with red panel; full-red section slides     |
| C   | Engineering notebook | Warm paper `#f6f3ec`, ink, blue `#1f5fd1`, JetBrains Mono headings + IBM Plex Sans body | Left rail with deck name and counter; `# ` prompt titles; dark terminal cover; grid-paper sections; boxed quote |

What every variant shares: theme tokens `text-accent`, `border-accent`, `bg-surface`, `text-muted` (decks use these
instead of raw Tailwind colours), a `.cite` class for inline citations, a Mermaid setup per variant so diagrams need no
per-deck colours, and `layouts/` for `cover`, `section` (frontmatter `kicker`), `quote` (frontmatter `author`), `center`
and `default`. The DTU logo is a text stand-in (`components/DtuLogo.vue`); the real asset must come from DTU.

The last four slides of the deck are theme test slides (section, quote, code with click highlighting, Mermaid).
