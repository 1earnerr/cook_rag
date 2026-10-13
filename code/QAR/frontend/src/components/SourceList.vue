<script setup lang="ts">
import type { SourceResponse } from '../types/rag'

defineProps<{
  sources: SourceResponse[]
}>()
</script>

<template>
  <section class="sources" aria-labelledby="sources-title">
    <div class="section-heading">
      <div>
        <p class="eyebrow">Evidence</p>
        <h3 id="sources-title">引用来源</h3>
      </div>
      <span class="count-badge">{{ sources.length }}</span>
    </div>

    <p v-if="sources.length === 0" class="empty-state">
      本次回答没有引用来源。
    </p>

    <ol v-else class="source-list">
      <li v-for="source in sources" :key="source.chunk_id" class="source-item">
        <div class="source-header">
          <span class="source-id">{{ source.source_id }}</span>
          <strong>{{ source.title }}</strong>
        </div>
        <dl>
          <div>
            <dt>章节</dt>
            <dd>{{ source.section }}</dd>
          </div>
          <div>
            <dt>文件</dt>
            <dd class="source-path">{{ source.source }}</dd>
          </div>
        </dl>
      </li>
    </ol>
  </section>
</template>

<style scoped>
.sources {
  border-top: 1px solid var(--border);
  margin-top: 1.5rem;
  padding-top: 1.35rem;
}

.section-heading,
.source-header {
  align-items: center;
  display: flex;
  gap: 0.75rem;
  justify-content: space-between;
}

.section-heading h3 {
  font-size: 1.05rem;
  margin: 0.2rem 0 0;
}

.eyebrow {
  color: var(--muted);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  margin: 0;
  text-transform: uppercase;
}

.count-badge,
.source-id {
  background: var(--accent-soft);
  border: 1px solid #b8d1ff;
  border-radius: 999px;
  color: var(--accent-strong);
  font-size: 0.78rem;
  font-weight: 800;
  padding: 0.2rem 0.55rem;
}

.empty-state {
  background: var(--surface-soft);
  border-radius: 10px;
  color: var(--muted);
  margin: 1rem 0 0;
  padding: 1rem;
}

.source-list {
  display: grid;
  gap: 0.75rem;
  list-style: none;
  margin: 1rem 0 0;
  padding: 0;
}

.source-item {
  background: var(--surface-soft);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.9rem 1rem;
}

.source-header {
  justify-content: flex-start;
}

.source-header strong {
  font-size: 0.94rem;
}

dl {
  display: grid;
  gap: 0.45rem;
  margin: 0.75rem 0 0;
}

dl div {
  display: grid;
  gap: 0.5rem;
  grid-template-columns: 3rem minmax(0, 1fr);
}

dt {
  color: var(--muted);
  font-size: 0.8rem;
}

dd {
  font-size: 0.84rem;
  margin: 0;
  overflow-wrap: anywhere;
}

.source-path {
  color: var(--muted-strong);
  font-family: var(--mono);
}
</style>
