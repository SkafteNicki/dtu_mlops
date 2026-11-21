<script setup lang="ts">
// Frontmatter: `author` attribution line. Slot: the quote itself (a markdown blockquote).
import { currentVariant } from '../variant'
defineProps<{ author?: string }>()
const variant = currentVariant()
</script>

<template>
  <div class="slidev-layout mlops-quote">
    <div v-if="variant === 'B'" class="mark">“</div>
    <slot />
    <div v-if="author" class="who">{{ author }}</div>
  </div>
</template>
