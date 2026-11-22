// PROTOTYPE: theme variant switch for the wayfinder ticket
// "What should the Slidev theme look like?". Throwaway, never ships.
// The variant is read once at startup from ?variant= (or localStorage) and
// switching reloads the page, so Mermaid, Shiki and dark mode stay consistent.
export const VARIANTS = [
  { key: 'D', name: 'Chosen: A + DTU red, dark/light toggle', dark: null },
  { key: 'A', name: 'Gamma dark', dark: true },
  { key: 'B', name: 'DTU corporate', dark: false },
  { key: 'C', name: 'Engineering notebook', dark: false },
] as { key: string; name: string; dark: boolean | null }[]

export type VariantKey = 'A' | 'B' | 'C' | 'D'

const STORAGE_KEY = 'mlops-theme-variant'

export function currentVariant(): VariantKey {
  if (typeof window === 'undefined') return 'D'
  const fromUrl = new URLSearchParams(window.location.search).get('variant')?.toUpperCase()
  const valid = (k?: string | null) => VARIANTS.some(v => v.key === k)
  if (valid(fromUrl)) {
    localStorage.setItem(STORAGE_KEY, fromUrl!)
    return fromUrl as VariantKey
  }
  const stored = localStorage.getItem(STORAGE_KEY)
  if (valid(stored)) return stored as VariantKey
  // Fallback for headless exports: VITE_MLOPS_VARIANT=B pnpm slidev export ...
  const fromEnv = (import.meta.env.VITE_MLOPS_VARIANT as string | undefined)?.toUpperCase()
  return (valid(fromEnv) ? fromEnv : 'D') as VariantKey
}

export function setVariant(key: VariantKey) {
  localStorage.setItem(STORAGE_KEY, key)
  const url = new URL(window.location.href)
  url.searchParams.set('variant', key)
  window.location.href = url.toString()
}

export function variantInfo(key: VariantKey = currentVariant()) {
  return VARIANTS.find(v => v.key === key)!
}
