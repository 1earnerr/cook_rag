<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { getApiErrorMessage, getHealth } from './api/rag'
import ChatPanel from './components/ChatPanel.vue'
import HealthCard from './components/HealthCard.vue'
import ReindexPanel from './components/ReindexPanel.vue'
import type { HealthResponse } from './types/rag'

const health = ref<HealthResponse | null>(null)
const healthLoading = ref(false)
const healthError = ref('')

async function refreshHealth(): Promise<void> {
  if (healthLoading.value) return
  healthLoading.value = true
  healthError.value = ''
  try {
    health.value = await getHealth()
  } catch (error) {
    healthError.value = getApiErrorMessage(error)
  } finally {
    healthLoading.value = false
  }
}

onMounted(refreshHealth)
</script>

<template>
  <div class="app-shell">
    <header class="site-header">
      <div class="brand-mark">Q</div>
      <div>
        <p class="product-label">QAR · Markdown Knowledge Base</p>
        <h1>可追踪证据的混合检索问答</h1>
      </div>
    </header>

    <main class="dashboard">
      <ChatPanel />
      <aside class="side-column">
        <HealthCard
          :health="health"
          :loading="healthLoading"
          :error="healthError"
          @refresh="refreshHealth"
        />
        <ReindexPanel @completed="refreshHealth" />
      </aside>
    </main>

    <footer>
      Vue 3 → FastAPI → RAGService → Hybrid RAG Core
    </footer>
  </div>
</template>

<style scoped>
.app-shell {
  margin: 0 auto;
  max-width: 1180px;
  padding: 2rem 1.25rem 1.5rem;
}

.site-header {
  align-items: center;
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.brand-mark {
  align-items: center;
  background: var(--text);
  border-radius: 14px;
  color: white;
  display: flex;
  flex: 0 0 auto;
  font-size: 1.3rem;
  font-weight: 900;
  height: 3.2rem;
  justify-content: center;
  width: 3.2rem;
}

.product-label {
  color: var(--accent-strong);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.08em;
  margin: 0 0 0.2rem;
  text-transform: uppercase;
}

h1 {
  font-size: clamp(1.45rem, 3vw, 2.1rem);
  letter-spacing: -0.03em;
  margin: 0;
}

.dashboard {
  align-items: start;
  display: grid;
  gap: 1.25rem;
  grid-template-columns: minmax(0, 2fr) minmax(17rem, 0.85fr);
}

.side-column {
  display: grid;
  gap: 1.25rem;
}

footer {
  color: var(--muted);
  font-family: var(--mono);
  font-size: 0.74rem;
  padding: 1.5rem 0 0;
  text-align: center;
}

@media (max-width: 850px) {
  .dashboard {
    grid-template-columns: 1fr;
  }

  .side-column {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 620px) {
  .app-shell {
    padding: 1rem 0.8rem;
  }

  .side-column {
    grid-template-columns: 1fr;
  }
}
</style>
