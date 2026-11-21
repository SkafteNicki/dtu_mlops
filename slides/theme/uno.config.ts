import { defineConfig } from 'unocss'

// Theme tokens decks can use instead of raw Tailwind colours (text-accent, border-accent, bg-accent, ...).
export default defineConfig({
  theme: {
    colors: {
      accent: 'var(--mlops-accent)',
      ink: 'var(--mlops-ink)',
      muted: 'var(--mlops-muted)',
      surface: 'var(--mlops-surface)',
    },
  },
})
