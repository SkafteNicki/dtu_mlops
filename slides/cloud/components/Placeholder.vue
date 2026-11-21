<script setup lang="ts">
// Stand-in for a source image that was not recreated in the rough draft.
// `kind` follows the image routes from the image-recreation research
// (screenshot, photo, chart, meme, decorative, logo); `page` is the page in
// the Gamma PDF the image came from. The slot describes what the image shows.
defineProps<{
  kind: string
  page: number
}>()
</script>

<template>
  <div class="placeholder">
    <div class="placeholder-label">Image placeholder · {{ kind }} · PDF page {{ page }}</div>
    <div class="placeholder-body"><slot /></div>
  </div>
</template>

<style scoped>
.placeholder {
  border: 2px dashed currentColor;
  border-radius: 8px;
  opacity: 0.75;
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  line-height: 1.4;
}
.placeholder-label {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-size: 0.7rem;
  margin-bottom: 0.4rem;
}
</style>
