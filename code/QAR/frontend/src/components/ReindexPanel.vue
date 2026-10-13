<script setup lang="ts">
import { ref } from 'vue'
import { getApiErrorMessage, reindex } from '../api/rag'
import type { ReindexResponse } from '../types/rag'

const emit = defineEmits<{
  completed: []
}>()

const loading = ref(false)
const error = ref('')
const result = ref<ReindexResponse | null>(null)

async function rebuild(): Promise<void> {
  if (loading.value) return
  const confirmed = window.confirm('索引重建会同步处理全部 Markdown。确认现在执行吗？')
  if (!confirmed) return

  loading.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await reindex()
    emit('completed')
  } catch (requestError) {
    error.value = getApiErrorMessage(requestError)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="panel side-panel" aria-labelledby="reindex-title">
    <div>
      <p class="eyebrow">Maintenance</p>
      <h2 id="reindex-title">索引重建</h2>
      <p class="description">同步重建 Markdown、Embedding、FAISS、BM25 和索引元数据。</p>
    </div>

    <p v-if="error" class="alert error-alert" role="alert">{{ error }}</p>

    <dl v-if="result" class="result-list" aria-live="polite">
      <div><dt>结果</dt><dd class="success-text">{{ result.success ? '成功' : '失败' }}</dd></div>
      <div><dt>文档</dt><dd>{{ result.documents }}</dd></div>
      <div><dt>Chunks</dt><dd>{{ result.chunks }}</dd></div>
      <div><dt>Schema</dt><dd>v{{ result.index_schema_version }}</dd></div>
      <div><dt>耗时</dt><dd>{{ result.elapsed_seconds.toFixed(2) }} s</dd></div>
    </dl>

    <button class="danger-button full-width" type="button" :disabled="loading" @click="rebuild">
      <span v-if="loading" class="spinner" aria-hidden="true" />
      {{ loading ? '正在重建索引…' : '重建索引' }}
    </button>
  </section>
</template>

<style scoped>
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

.description {
  color: var(--muted);
  font-size: 0.84rem;
  line-height: 1.55;
  margin: 0.75rem 0 1rem;
}

.result-list {
  background: var(--success-soft);
  border: 1px solid #bce8d4;
  border-radius: 12px;
  display: grid;
  gap: 0.55rem;
  margin: 0 0 1rem;
  padding: 0.8rem;
}

.result-list div {
  display: flex;
  justify-content: space-between;
}

dt {
  color: var(--muted);
  font-size: 0.8rem;
}

dd {
  font-size: 0.84rem;
  font-weight: 700;
  margin: 0;
}

.success-text {
  color: var(--success);
}
</style>
