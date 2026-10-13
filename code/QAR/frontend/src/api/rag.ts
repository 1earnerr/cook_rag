import axios from 'axios'
import type {
  ChatRequest,
  ChatResponse,
  HealthResponse,
  ReindexResponse,
} from '../types/rag'

interface ErrorResponse {
  detail?: string
}

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim()

const apiClient = axios.create({
  baseURL: (configuredBaseUrl || 'http://localhost:8000').replace(/\/+$/, ''),
  timeout: 300_000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export async function getHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/api/v1/health')
  return response.data
}

export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const response = await apiClient.post<ChatResponse>('/api/v1/chat', request)
  return response.data
}

export async function reindex(): Promise<ReindexResponse> {
  const response = await apiClient.post<ReindexResponse>('/api/v1/reindex')
  return response.data
}

export function getApiErrorMessage(error: unknown): string {
  if (!axios.isAxiosError<ErrorResponse>(error)) {
    return '发生未知错误，请稍后重试。'
  }
  if (!error.response) {
    return '无法连接后端服务，请确认 FastAPI 已启动且地址配置正确。'
  }

  switch (error.response.status) {
    case 422:
      return '请求参数不正确，请检查问题内容后重试。'
    case 500:
      return '后端处理失败，请查看服务日志后重试。'
    case 502:
      return '外部大模型服务不可用，请检查 LLM 配置或稍后重试。'
    case 503:
      return 'RAG 服务尚未初始化完成，请先查看服务状态。'
    default:
      return error.response.data?.detail || `请求失败（HTTP ${error.response.status}）。`
  }
}
