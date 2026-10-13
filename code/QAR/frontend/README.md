# QAR Vue Frontend

QAR 的最小 Web 演示界面，只调用现有 FastAPI `/chat`、`/health`、`/reindex`，不保存聊天记录，不增加后端业务能力。

## 环境

- Node.js 20+（本项目使用 Node.js 22.17.0 完成构建验证）
- npm 10+
- 已安装并可启动的 QAR FastAPI 后端

## 安装与配置

在 `code/QAR/frontend` 目录执行：

```powershell
npm install
Copy-Item .env.example .env.local
```

`.env.local`：

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

API 地址只在环境变量和 `src/api/rag.ts` 的单一 fallback 中定义，组件不拼接后端 URL。

## 启动

先在项目根目录启动后端：

```powershell
.venv-lc1/Scripts/python.exe -m uvicorn api:app --app-dir code/QAR --host 127.0.0.1 --port 8001
```

再启动前端：

```powershell
Set-Location code/QAR/frontend
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。后端 CORS 只允许本地 Vite 地址 `http://localhost:5173` 和 `http://127.0.0.1:5173`。

## 构建

```powershell
npm run build
```

构建脚本先执行 `vue-tsc --noEmit`，再生成 `dist/`。

## 页面功能

1. **知识库问答**：提交当前问题，展示 Answer、Hybrid mode、Evidence 数量以及实际 Sources。安全拒答是正常 200 响应，不显示为错误。
2. **服务状态**：页面加载时自动请求 `/health`，展示真实 Embedding、Index、文档和 Chunk 状态，并可手动刷新。
3. **索引重建**：确认后同步调用 `/reindex`；执行期间禁用按钮，成功后展示计数、schema 和耗时并刷新健康状态。

Axios 请求集中在 `src/api/rag.ts`，类型集中在 `src/types/rag.ts`。前端分别处理 422、500、502、503 和网络错误，不展示后端 traceback。

## 已知限制

- 页面只保留当前一次回答，刷新后清空。
- 不提供登录、权限、多租户、文件上传或知识库管理。
- `/chat` 是普通 HTTP 请求，不使用 WebSocket 或流式输出。
- `/reindex` 是后端现有的同步全量重建，执行时需要等待。
- 缺少 `LLM_API_KEY`、`LLM_BASE_URL` 或 `LLM_MODEL`，或外部 OpenAI-compatible LLM 调用失败时，需要 LLM 的问答会显示 502 友好提示；被 Relevance Gate 拒绝的问题仍可正常返回安全拒答。
- 本前端只用于本地演示，没有生产部署、安全加固或浏览器兼容矩阵验证。

## 人工浏览器验收

按上面的命令先启动 FastAPI，再启动 Vite。若 8000 端口被占用，改用可用端口并同步修改 `.env.local` 中的 `VITE_API_BASE_URL`，随后重新启动 Vite。浏览器打开 `http://localhost:5173`（必要时使用 `http://127.0.0.1:5173`），提交“糖醋鲤鱼需要哪些原料？”并检查 Answer、retrieval mode/count 和 Sources；再提交“如何在三台服务器上搭建 Kubernetes 高可用集群？”并检查固定安全拒答、空 Sources 和 count 0。当前真实 LLM HTTP E2E 使用 DeepSeek `deepseek-chat` OpenAI-compatible API 完成；浏览器视觉验收尚待人工执行。
