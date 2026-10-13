<script setup lang="ts">
import type { HealthResponse } from '../types/rag'

defineProps<{
  health: HealthResponse | null
  loading: boolean
  error: string
}>()

defineEmits<{
  refresh: []
}>()
</script>

<template>
  <section class="panel side-panel" aria-labelledby="health-title">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Runtime</p>
        <h2 id="health-title">服务状态</h2>
      </div>
      <span
        class="status-pill"
        :class="health?.status === 'ok' ? 'status-ok' : 'status-degraded'"
      >
        {{ health?.status ?? 'unknown' }}
      </span>
    </div>

    <p v-if="error" class="alert error-alert" role="alert">{{ error }}</p>

    <div v-if="health" class="health-grid">
      <div class="metric">
        <span>文档</span>
        <strong>{{ health.documents }}</strong>
      </div>
      <div class="metric">
        <span>Chunks</span>
        <strong>{{ health.chunks }}</strong>
      </div>
    </div>

    <ul v-if="health" class="check-list">
      <li>
        <span>Embedding</span>
        <strong :class="health.embedding_loaded ? 'ready' : 'not-ready'">
          {{ health.embedding_loaded ? '已加载' : '未加载' }}
        </strong>
      </li>
      <li>
        <span>Index</span>
        <strong :class="health.index_loaded ? 'ready' : 'not-ready'">
          {{ health.index_loaded ? '已加载' : '未加载' }}
        </strong>
      </li>
    </ul>

    <p v-if="health?.detail" class="detail">{{ health.detail }}</p>

    <button class="secondary-button full-width" type="button" :disabled="loading" @click="$emit('refresh')">
      <span v-if="loading" class="spinner spinner-dark" aria-hidden="true" />
      {{ loading ? '正在刷新…' : '刷新状态' }}
    </button>
  </section>
</template>

<style scoped>
.panel-heading {
  align-items: center;
  display: flex;
  gap: 1rem;
  justify-content: space-between;
}

h2 {
  font-size: 1.2rem;
  margin: 0.2rem 0 0;
}

.eyebrow {
  color: var(--muted);
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  margin: 0;
  text-transform: uppercase;
}

.status-pill {
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 800;
  padding: 0.35rem 0.65rem;
  text-transform: uppercase;
}

.status-ok {
  background: var(--success-soft);
  color: var(--success);
}

.status-degraded {
  background: var(--warning-soft);
  color: var(--warning);
}

.health-grid {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: 1fr 1fr;
  margin-top: 1.25rem;
}

.metric {
  background: var(--surface-soft);
  border: 1px solid var(--border);
  border-radius: 12px;
  display: grid;
  gap: 0.25rem;
  padding: 0.8rem;
}

.metric span {
  color: var(--muted);
  font-size: 0.76rem;
}

.metric strong {
  font-size: 1.35rem;
}

.check-list {
  display: grid;
  gap: 0.7rem;
  list-style: none;
  margin: 1rem 0;
  padding: 0;
}

.check-list li {
  align-items: center;
  display: flex;
  font-size: 0.86rem;
  justify-content: space-between;
}

.ready {
  color: var(--success);
}

.not-ready {
  color: var(--warning);
}

.detail {
  background: var(--warning-soft);
  border-radius: 10px;
  color: var(--warning);
  font-size: 0.82rem;
  line-height: 1.5;
  padding: 0.75rem;
}
</style>
