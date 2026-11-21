<script setup lang="ts">
// PROTOTYPE stand-in for the official DTU logo (the real asset comes from DTU's design guide).
defineProps<{ color?: string }>()
</script>

<template>
  <div class="dtu-logo" :style="{ color: color ?? '#990000' }" title="Stand-in for the official DTU logo">
    <span>DTU</span><small>stand-in</small>
  </div>
</template>

<style scoped>
.dtu-logo { display: inline-flex; flex-direction: column; align-items: flex-start; line-height: 1; font-family: Arial, sans-serif; }
.dtu-logo span { font-weight: 900; font-size: 1.4rem; letter-spacing: 0.04em; }
.dtu-logo small { font-size: 0.45rem; opacity: 0.6; text-transform: uppercase; }
</style>
