<script setup lang="ts">
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'
import { currentVariant } from '../variant'

const variant = currentVariant()
const { $slidev } = useSlideContext()
const deck = computed(() => String($slidev.configs.title ?? '').toLowerCase().replace(/^day\s*\d+\s*-\s*/, ''))
</script>

<template>
  <div v-if="variant === 'B'" class="slidev-layout mlops-cover">
    <div class="panel">
      <DtuLogo color="#ffffff" />
      <div class="text-sm opacity-90">02476 Machine Learning Operations<br>DTU Compute</div>
    </div>
    <div class="body"><slot /></div>
  </div>
  <div v-else-if="variant === 'C'" class="slidev-layout mlops-cover">
    <div class="prompt">$ mlops lecture --topic {{ deck }}</div>
    <div><slot /></div>
  </div>
  <div v-else class="slidev-layout mlops-cover"><slot /></div>
</template>
