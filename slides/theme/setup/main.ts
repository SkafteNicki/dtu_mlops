import { currentVariant, variantInfo } from '../variant'

export default (() => {
  if (typeof document === 'undefined') return
  const key = currentVariant()
  const html = document.documentElement
  html.dataset.variant = key
  // Each variant fixes its own colour scheme; Slidev stores the scheme here.
  localStorage.setItem('slidev-color-schema', variantInfo(key).dark ? 'dark' : 'light')
  html.classList.toggle('dark', variantInfo(key).dark)
})
