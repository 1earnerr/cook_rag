<script setup lang="ts">
import { ref } from 'vue'
import { chat, getApiErrorMessage } from '../api/rag'
import type { ChatResponse } from '../types/rag'
import SourceList from './SourceList.vue'

const query = ref('糖醋鲤鱼需要哪些原料？')
const loading = ref(false)
const error = ref('')
const result = ref<ChatResponse | null>(null)

async function submit(): Promise<void> {
  const normalized = query.value.trim()
  if (!normalized || loading.value) {
    if (!normalized) error.value = '请输入问题。'
    return
  }

  loading.value = true
  error.value = ''
  result.value = null
  try {
    result.value = await chat({ query: normalized, filters: {} })
  } catch (requestError) {
    error.value = getApiErrorMessage(requestError)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <section class="panel chat-panel" aria-labelledby="chat-title">
    <div class="panel-heading">
      <div>
        <p class="eyebrow">Ask the knowledge base</p>
        <h2 id="chat-title">知识库问答</h2>
      </div>
      <span class="mode-chip">Hybrid Retrieval</span>
    </div>

    <form class="question-form" @submit.prevent="submit">
      <label for="question">问题</label>
      <textarea
        id="question"
        v-model="query"
        rows="4"
        maxlength="2000"
        placeholder="请输入要查询的知识库问题"
        :disabled="loading"
        @keydown.ctrl.enter.prevent="submit"
      />
      <div class="form-footer">
        <span>Ctrl + Enter 提交</span>
        <button class="primary-button" type="submit" :disabled="loading || !query.trim()">
          <span v-if="loading" class="spinner" aria-hidden="true" />
          {{ loading ? '正在检索并生成…' : '提交问题' }}
        </button>
      </div>
    </form>

    <p v-if="error" class="alert error-alert" role="alert">{{ error }}</p>

    <div v-if="result" class="answer-block" aria-live="polite">
      <div class="answer-meta">
        <span>模式：{{ result.retrieval.mode }}</span>
        <span>证据：{{ result.retrieval.count }}</span>
      </div>
      <p class="answer-text">{{ result.answer }}</p>
      <SourceList :sources="result.sources" />
    </div>

    <div v-else-if="!loading && !error" class="initial-state">
      <span class="initial-icon">Q</span>
      <p>提交一个问题，页面只保留当前一次回答及其实际证据来源。</p>
    </div>
  </section>
</template>

<style scoped>
.chat-panel {
  min-height: 33rem;
}

.panel-heading,
.form-footer,
.answer-meta {
  align-items: center;
  display: flex;
  justify-content: space-between;
}

.panel-heading {
  gap: 1rem;
  margin-bottom: 1.5rem;
}

h2 {
  font-size: 1.45rem;
  margin: 0.25rem 0 0;
}

.eyebrow {
  color: var(--accent-strong);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  margin: 0;
  text-transform: uppercase;
}

.mode-chip {
  background: var(--surface-soft);
  border: 1px solid var(--border);
  border-radius: 999px;
  color: var(--muted-strong);
  font-size: 0.76rem;
  font-weight: 700;
  padding: 0.45rem 0.7rem;
  white-space: nowrap;
}

.question-form {
  display: grid;
  gap: 0.6rem;
}

label {
  font-size: 0.88rem;
  font-weight: 700;
}

textarea {
  background: #fbfcff;
  border: 1px solid var(--border-strong);
  border-radius: 12px;
  color: var(--text);
  font: inherit;
  line-height: 1.65;
  padding: 0.9rem 1rem;
  resize: vertical;
  transition: border-color 150ms ease, box-shadow 150ms ease;
}

textarea:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
  outline: none;
}

.form-footer {
  gap: 1rem;
}

.form-footer > span {
  color: var(--muted);
  font-size: 0.78rem;
}

.answer-block {
  border-top: 1px solid var(--border);
  margin-top: 1.5rem;
  padding-top: 1.35rem;
}

.answer-meta {
  color: var(--muted);
  font-size: 0.8rem;
  justify-content: flex-start;
  gap: 1rem;
}

.answer-text {
  font-size: 1rem;
  line-height: 1.8;
  margin: 0.9rem 0 0;
  white-space: pre-wrap;
}

.initial-state {
  align-items: center;
  color: var(--muted);
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  justify-content: center;
  min-height: 15rem;
  text-align: center;
}

.initial-state p {
  margin: 0;
  max-width: 26rem;
}

.initial-icon {
  align-items: center;
  background: var(--accent-soft);
  border: 1px solid #b8d1ff;
  border-radius: 14px;
  color: var(--accent-strong);
  display: flex;
  font-size: 1.2rem;
  font-weight: 900;
  height: 3rem;
  justify-content: center;
  width: 3rem;
}

@media (max-width: 620px) {
  .panel-heading,
  .form-footer {
    align-items: stretch;
    flex-direction: column;
  }

  .primary-button {
    justify-content: center;
    width: 100%;
  }
}
</style>
