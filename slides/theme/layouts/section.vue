<script setup lang="ts">
// Frontmatter: `kicker` (e.g. "02") shown above the title.
defineProps<{ kicker?: string }>()
</script>

<template>
  <div class="slidev-layout mlops-section">
    <div v-if="kicker" class="kicker">{{ kicker }}</div>
    <slot />
  </div>
</template>
