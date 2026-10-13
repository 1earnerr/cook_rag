export type FilterScalar = string | number | boolean
export type FilterValue = FilterScalar | FilterScalar[]

export interface ChatRequest {
  query: string
  filters: Record<string, FilterValue>
}

export interface SourceResponse {
  source_id: string
  chunk_id: string
  document_id: string
  title: string
  section: string
  source: string
}

export interface RetrievalInfo {
  mode: 'hybrid'
  count: number
}

export interface ChatResponse {
  answer: string
  sources: SourceResponse[]
  retrieval: RetrievalInfo
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  embedding_loaded: boolean
  index_loaded: boolean
  documents: number
  chunks: number
  detail: string | null
}

export interface ReindexResponse {
  success: boolean
  documents: number
  chunks: number
  index_schema_version: number
  elapsed_seconds: number
}
