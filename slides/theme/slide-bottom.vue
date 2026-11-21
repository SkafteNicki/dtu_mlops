<script setup lang="ts">
// Per-slide chrome. A: page number only. B: red rule + logo + footer. C: left rail.
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'
import { currentVariant } from './variant'

const variant = currentVariant()
const { $page, $frontmatter, $slidev } = useSlideContext()
const layout = computed(() => $frontmatter.layout ?? 'default')
const bare = computed(() => ['cover', 'section'].includes(layout.value))
const total = computed(() => $slidev.nav.total)
const title = computed(() => String($slidev.configs.title ?? ''))
</script>

<template>
  <template v-if="!bare">
    <div v-if="variant === 'A'" class="mlops-pageno">{{ $page }}</div>
    <div v-else-if="variant === 'B'" class="mlops-chrome-b">
      <div class="rule" />
      <div class="logo"><DtuLogo /></div>
      <div class="footer"><span>02476 Machine Learning Operations · {{ title }}</span><span>{{ $page }} / {{ total }}</span></div>
    </div>
    <div v-else class="mlops-chrome-c">
      <span class="count">{{ String($page).padStart(2, '0') }}</span>
      <span class="deck">02476 · {{ title }}</span>
      <span>{{ total }}</span>
    </div>
  </template>
</template>
