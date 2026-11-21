<script setup lang="ts">
// PROTOTYPE variant switcher. Dev server only, never in builds or exports.
// Keys: [ and ] cycle (arrow keys belong to Slidev navigation).
import { onMounted, onUnmounted } from 'vue'
import { VARIANTS, currentVariant, setVariant, variantInfo } from './variant'

const show = import.meta.env.DEV && typeof window !== 'undefined' && !/\/(export|print|presenter)/.test(window.location.pathname)
const cur = currentVariant()
const idx = VARIANTS.findIndex(v => v.key === cur)
const step = (d: number) => setVariant(VARIANTS[(idx + d + VARIANTS.length) % VARIANTS.length].key)
const onKey = (e: KeyboardEvent) => {
  const t = e.target as HTMLElement
  if (t?.closest('input, textarea, [contenteditable]')) return
  if (e.key === ']') step(1)
  if (e.key === '[') step(-1)
}
onMounted(() => show && window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <div v-if="show" class="proto-switcher">
    <button @click="step(-1)">‹</button>
    <span>Theme prototype · {{ cur }} ({{ variantInfo(cur).name }})</span>
    <button @click="step(1)">›</button>
  </div>
</template>

<style>
.proto-switcher { position: fixed; bottom: 14px; left: 50%; transform: translateX(-50%); z-index: 9999; display: flex; gap: 10px; align-items: center; background: #ff0; color: #000; font: 600 12px/1 ui-monospace, monospace; padding: 6px 12px; border-radius: 999px; box-shadow: 0 2px 10px rgba(0,0,0,.4); border: 2px solid #000; }
.proto-switcher button { font-size: 16px; padding: 0 6px; cursor: pointer; }
</style>
