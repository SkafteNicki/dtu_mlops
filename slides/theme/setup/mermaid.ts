import { currentVariant } from '../variant'

export default (() => {
  const v = currentVariant()
  if (v === 'D') {
    const dark = typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
    return {
      theme: 'base',
      themeVariables: dark
        ? { darkMode: true, background: '#161415', fontFamily: 'Barlow', primaryColor: '#262223', primaryBorderColor: '#e83f48', primaryTextColor: '#e5e0df', lineColor: '#a8a2a0', secondaryColor: '#3a2526', tertiaryColor: '#1f1b1c' }
        : { fontFamily: 'Barlow', primaryColor: '#ffffff', primaryBorderColor: '#990000', primaryTextColor: '#1a1718', lineColor: '#5c5657', secondaryColor: '#f5e6e6', tertiaryColor: '#f3f1f0' },
    }
  }
  if (v === 'A')
    return {
      theme: 'base',
      themeVariables: {
        darkMode: true, background: '#161415', fontFamily: 'Barlow',
        primaryColor: '#262223', primaryBorderColor: '#f65f62', primaryTextColor: '#e5e0df',
        lineColor: '#a8a2a0', secondaryColor: '#3a2526', tertiaryColor: '#1f1b1c',
      },
    }
  if (v === 'B')
    return {
      theme: 'base',
      themeVariables: {
        fontFamily: 'Arial', primaryColor: '#ffffff', primaryBorderColor: '#990000',
        primaryTextColor: '#000000', lineColor: '#990000', secondaryColor: '#f5e6e6', tertiaryColor: '#f2f2f2',
      },
    }
  return {
    theme: 'base',
    themeVariables: {
      fontFamily: 'JetBrains Mono', primaryColor: '#fffdf7', primaryBorderColor: '#1d1f24',
      primaryTextColor: '#1d1f24', lineColor: '#1f5fd1', secondaryColor: '#e7eefc', tertiaryColor: '#f6f3ec',
    },
  }
})
