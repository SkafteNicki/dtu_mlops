import { currentVariant, variantInfo } from '../variant'

export default (() => {
  if (typeof document === 'undefined') return
  const key = currentVariant()
  const html = document.documentElement
  html.dataset.variant = key
  // D follows Slidev's dark/light toggle; A-C fix their own colour scheme.
  if (variantInfo(key).dark === null) return
  localStorage.setItem('slidev-color-schema', variantInfo(key).dark ? 'dark' : 'light')
  html.classList.toggle('dark', variantInfo(key).dark)
})
