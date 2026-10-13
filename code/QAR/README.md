# QAR：通用 Markdown 知识库 RAG API

基于本地 Markdown 的检索问答服务，同时提供 FastAPI HTTP API 和命令行入口。项目代码位于 `code/QAR`，公开食谱语料位于 `data/QAR/cook`；这些文件是测试语料，不是企业内部数据。语料中的模板注明来自 HowToCook，本项目不将食谱内容作为原创成果；分发前应核对上游来源和许可，当前副本未提供独立许可文件。

## 安装、运行与测试

建议 Python 3.12。在项目根目录执行（PowerShell）：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r code/QAR/requirements.txt
Copy-Item code/QAR/.env.example code/QAR/.env
# 编辑 .env，填写 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
.venv/Scripts/python.exe code/QAR/main.py
```

HTTP 服务在项目根目录启动：

```powershell
.venv/Scripts/python.exe -m uvicorn api:app --app-dir code/QAR --host 127.0.0.1 --port 8000
```

启动期间会同步加载 BGE，并根据 corpus/index metadata 复用或重建索引。初始化失败时服务仍保留 `/api/v1/health`，状态为 `degraded`；此时 `/chat` 返回 503。

首次真实运行需要下载 `BAAI/bge-small-zh-v1.5`（CPU 推理），生成答案需要可用的 OpenAI-compatible LLM API。模型名由 `.env` 中的 `LLM_MODEL` 指定；项目不内置特定提供商或默认模型。默认数据、索引及 `.env` 路径相对于本文件目录定位，不依赖启动目录。

项目使用 LangChain 1.x 生态的直接依赖：`langchain-core==1.6.3`、`langchain-huggingface==1.2.2`、`langchain-text-splitters==1.1.2` 和 `langchain-openai==1.6.2`。源码不使用顶层 `langchain` 或 `langchain-classic`。Phase 2.7 已将原 community FAISS/BM25 wrapper 替换为直接调用 `faiss-cpu` 与 `rank-bm25` 的小型项目内适配器，因此不再安装已 sunset 的 `langchain-community`，也不再传递安装 `langchain-classic`。

HTTP 层直接依赖 `fastapi==0.141.1`、`pydantic==2.13.5` 和 `uvicorn==0.53.0`。API 只通过 `RAGService` 调用现有 `MarkdownRAGSystem`，不在路由层操作 FAISS、BM25、RRF、Embedding 或 Prompt。

原有检索架构保持不变：native `IndexFlatL2` 保存 BGE 向量并维护稳定 chunk ID 与 Document 映射；`BM25Okapi` 继续使用项目的中文 tokenizer；统一 `retrieve()`、三种 mode、RRF、metadata filter、Evidence 和索引生命周期接口不变。

OpenAI-compatible LLM 通过 `langchain-openai` 的 `ChatOpenAI` 调用 Chat Completions 接口，并显式设置 `use_responses_api=False`。`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 均从环境读取，temperature 0.1 和 max tokens 2048 保持不变。当前项目只使用标准文本生成/流式输出，不依赖特定提供商的扩展字段。

不下载 BGE、不调用外部 LLM 的回归测试：

```powershell
.venv/Scripts/python.exe -m pip install -r code/QAR/requirements-test.txt
.venv/Scripts/python.exe -m pytest -c code/QAR/pytest.ini code/QAR/tests -q
```

测试使用真实 LangChain 分割器、真实 FAISS/BM25；通过依赖注入提供确定性测试向量和模拟 LLM。它验证程序行为与持久化一致性，**不验证 BGE 语义质量或真实模型 API 可用性**。缺依赖时测试报错，不静默跳过。

Embedding 分层诊断和真实 BGE + FAISS 冒烟可以单独运行：

```powershell
$env:HF_HOME = "$PWD/.hf-cache"
.venv/Scripts/python.exe code/QAR/tools/diagnose_embedding.py
.venv/Scripts/python.exe code/QAR/tools/smoke_real_bge_faiss.py
```

诊断脚本依次验证 import、AutoTokenizer、AutoModel、SentenceTransformer、LangChain HuggingFaceEmbeddings、`embed_query` 和 `embed_documents`，每一步在独立子进程中记录时间、缓存目录、CPU/CUDA 环境和异常。若模型已缓存，可额外设置 `$env:HF_HUB_OFFLINE = "1"`，以区分本地模型加载和 Hugging Face 网络访问问题。冒烟脚本验证当前公开语料的真实向量建库、保存、重载、FAISS/BM25/Hybrid、RRF metadata 及 Evidence/Context 一致性；它不调用外部 LLM，也不代表检索质量评估。

## Docker 启动

项目根目录提供最小 Docker Compose 配置，后端和 Vue production 静态页面分别运行在两个容器中：

```bash
docker compose build
docker compose up
```

访问：

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
```

停止：

```bash
docker compose down
```

Compose 从 `code/QAR/.env` 注入 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`；真实 `.env` 不进入 Docker build context，也不会 COPY 到镜像。首次启动可能下载 `BAAI/bge-small-zh-v1.5`，模型缓存使用 `huggingface_cache` named volume；索引文件使用 `qar_index` named volume。该配置仅用于本地演示，不包含生产级高可用、Kubernetes 或 CI/CD。

前端镜像在构建阶段执行 `npm ci` 和 `npm run build`，运行阶段由 Nginx 托管 `dist`。浏览器请求的 API 地址保留为 `http://localhost:8000`，因为请求由宿主机浏览器发出，而不是 frontend 容器内部发出。

## HTTP API

API 调用链保持为：

```text
HTTP Request → FastAPI → RAGService → MarkdownRAGSystem
             → Hybrid Retrieval / Evidence / LLM → HTTP Response
```

查看实际加载状态（文档数和 Chunk 数来自内存中的当前语料）：

```powershell
curl.exe http://127.0.0.1:8000/api/v1/health
```

执行混合检索与问答：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d '{"query":"糖醋鲤鱼需要哪些原料？","filters":{}}'
```

`sources` 是实际进入 Prompt 的 Evidence，并返回 `source_id`、`chunk_id`、`document_id`、`title`、`section` 和 `source`。无有效 Evidence 时返回 HTTP 200、固定拒答文本、空 `sources` 及 `count=0`，不调用回答生成模型。

同步强制重建当前语料的 FAISS、BM25 和 index metadata：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/reindex
```

参数校验失败返回 422，未就绪返回 503，外部 LLM 调用失败返回 502，检索/重建失败返回脱敏的 500。完整 traceback 只记录在服务日志中。

## Web Frontend

`frontend/` 提供独立的 Vue 3 + Vite + TypeScript + Axios 演示页面，包含当前单次问答与 Sources、真实服务状态和同步索引重建。前端只调用上述三个既有 API，不保存聊天历史，也不增加新的 RAG 功能。

```powershell
Set-Location code/QAR/frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

默认通过 `VITE_API_BASE_URL=http://localhost:8000` 访问后端。FastAPI 仅增加本地开发 CORS，允许 `http://localhost:5173` 和 `http://127.0.0.1:5173`。完整运行与构建说明见 `frontend/README.md`。

## Retrieval Evaluation

`evaluation/questions.jsonl` 包含 39 条人工标注问题（27 条有答案、12 条无答案），使用当前公开 Markdown 语料做文档级 relevance 评估。脚本只调用冻结的统一 `retrieve()`，比较 Vector、BM25 和 Hybrid：

```powershell
$env:HF_HOME = "$PWD/.hf-cache"
$env:HF_HUB_OFFLINE = "1"
.venv-lc1/Scripts/python.exe code/QAR/evaluation/evaluate_retrieval.py
```

当前一次固定 baseline 运行结果（无答案问题不计入指标）：

| Method | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|
| Vector | 0.8889 | 0.9630 | 0.8420 |
| BM25 | 1.0000 | 1.0000 | 0.9568 |
| Hybrid | 1.0000 | 1.0000 | 0.9815 |

逐问题排名、按 Query Type 统计、无答案返回及失败分析见 `evaluation/results/`。这是小规模诊断集，结果不证明对企业知识库的泛化能力，也不是生成答案质量评估。

当前 Relevance Gate 使用最终 Hybrid Chunk 与 Query 的最大 BGE 余弦相似度，阈值为 0.45。在当前 27/12 分布中它不误拒已标注答案，但仍会误接受 3 条无答案 Query，因此只是轻量保守防线，不是可靠的通用 OOD 分类器。

## 使用其他 Markdown 语料

在 `code/QAR` 下使用 Python：

```python
from config import RAGConfig
from main import MarkdownRAGSystem

config = RAGConfig(
    data_path="D:/knowledge/markdown",
    index_save_path="D:/knowledge/index",
    query_rewrite_enabled=False,
    recipe_metadata_enabled=False,
    context_max_chars=2000,
)
app = MarkdownRAGSystem(config)
app.initialize_system()
app.build_knowledge_base()
result = app.ask_question("如何恢复服务？")
print(result.to_dict())  # answer + sources
```

命令行与 HTTP API 共用同一 RAG Core。当前是单进程同步服务，没有对话记忆。显式 `filters={"source": "manual.md"}` 可调用通用 metadata 后过滤；不会根据问题自动猜测食谱分类/难度。

## 数据与索引约定

- 加载 `.md`/`.MD`；跳过 `template`/`templates` 目录及名为 `template`、`示例菜` 的模板文件，跳过空白/纯注释文件。读取失败终止本次建库，避免静默丢文档。
- 父文档保留原文。检索文本去除 HTML 注释、将 Markdown 图片替换为替代文本；不读取图片内容。清洗是轻量规则，不是完整 Markdown AST 解析器。
- 通用 metadata：`document_id`、`parent_id`、`title`、`source`、`section`；子块另有 `chunk_id`、`chunk_index` 等。
- `source` 是相对语料根目录的路径。`document_id` 是该路径的 SHA-256；内容修改不改变 document_id，重命名会改变它。
- `chunk_id` 基于 document_id、章节/子块位置和正文；无关文档增删不会改变其 ID。改变当前文档中的章节顺序可能改变后续块 ID。
- 分块先保留 Markdown 标题结构，再用 BGE tokenizer 计算输入 token 数。超过 `embedding_chunk_max_tokens=450` 的 Section 才进入 `RecursiveCharacterTextSplitter` 二次切分；最终 Chunk 会再次检查 token 数。450 是低于模型 512 上限的安全配置，不是经过效果评估得到的最优值；overlap 仍按字符配置。
- `recipe_metadata_enabled=True` 时才提取食谱 `category`、`difficulty`、`dish_name`。默认关闭，通用流程不依赖这些字段。
- `index_metadata.json` 记录原文指纹、模型名、分块版本、分块内容/metadata 指纹、可选元数据开关及 FAISS 文件校验值。元数据与当前内容一致、文件完整时才加载索引，否则全量重建。
- 每次建库只读取一次原文：当前父文档、当前 chunks、BM25 与新 FAISS 使用同一批内存数据；复用旧 FAISS 前验证该批数据的指纹。
- 保存前使旧 manifest 失效，索引文件完成后才写成功标记；失败后下次重建。它不是多进程原子快照发布，也不支持并发重建。
- Windows 下 FAISS 原生文件 API 无法可靠处理中文路径；索引模块改用 FAISS 内存序列化，再由 Python 文件 API 保存/读取 `index.faiss`，docstore 仍保存在本地可信的 `index.pkl`。
- 文档改动在下一次 `build_knowledge_base()` 或重启时生效，不做文件监控。空语料重建报错，不替换当前内存知识库；不会声称已经切换到空库。
- 只加载本机可信索引目录。FAISS 文档存储需要 pickle 反序列化；哈希用于检测意外损坏，不是针对恶意篡改的安全签名。

## 证据与回答

```text
Markdown → 标题分块 → BGE/FAISS + BM25 → RRF
→ 命中子块 → 有界证据上下文 → 通用 Prompt → LLM
```

系统不再把命中子块替换成整篇父文档，避免长文档前部挤掉真正命中的后部章节。保留父文档及回溯方法用于查阅原文，暂不自动扩展相邻章节。

每条 Sources 包含 `evidence_id`（S1/S2）、`chunk_id`、`document_id`、`title`、`section`、`source`、`content`。其中 content 是**实际进入 Prompt 的原始子块片段**，截断时不伪造完整正文。预算包含证据标题和分隔符，但不包括整个系统提示词及问题；这是字符预算，不是 token 预算。长块按剩余候选分享预算，尽量为后续证据留空间。

普通回答返回 `RAGAnswer(answer, sources)`，可调用 `to_dict()`；流式返回 `StreamEvent`：`sources` → `token` → `done`。不使用共享 `last_sources` 状态。无有效证据时不调用回答生成模型，返回明确说明；若显式开启 Query Rewrite，检索前仍可能发生一次改写模型调用。

Prompt 要求使用编号引用，但尚无事实蕴含检查或模型引用 ID 校验，因此 Sources 代表提供给模型的证据，不保证每条事实都被正确引用。当前 0.45 相关性阈值是基于小规模数据集选择的保守 Gate，仍会误接受部分知识库外问题，不能视为通用 OOD 分类器。

Query Rewrite 默认关闭。开启后，异常、空结果或非字符串结果回退原问题；最终回答始终使用原始问题。没有 list/detail/general 路由或食谱专用生成模板。

## 已知边界

- 当前 `vector_fetch_k=20`、`bm25_fetch_k=20`、`final_k=5`、`rrf_k=60` 是可配置默认值，尚未通过评估调优。不要把默认参数写成效果结论。
- BM25 使用确定性的中文单字/双字词与 ASCII/数字 tokenizer；它不是中文分词模型，也没有引入新的搜索服务。
- RRF 保留 `vector_rank`、`bm25_rank`、`rrf_score`，没有加入模型 Reranker。
- 开发 Evaluation 有 39 条人工问题，另有 22 条独立 Holdout；仍没有 Chunk-level relevance、生成答案评估或统计显著性结论。
- FastAPI 只提供 `/chat`、`/health`、`/reindex`；Vue 前端只展示这三个接口，无数据库、用户/权限、对话历史、多租户和后台任务。
- `/reindex` 是同步全量重建，适用于当前小型公开语料；没有多进程锁或零停机索引切换，不应同时发起问答和重建。
- API 未实现认证和限流，应只在可信本地/内网环境中暴露，尤其是可触发重建的 `/reindex`。
- `max_section_chars=1200` 和 `section_chunk_overlap=100` 只控制递归切分器的字符窗口；最终安全检查以 BGE tokenizer 的 token 数为准。这些参数尚未通过 Evaluation 调优。
- 上层 `code/requirements.txt` 和 Milvus `code/docker-compose.yml` 是遗留文件，QAR 不使用；请按本目录依赖安装，不启动该 Compose。

基础测试覆盖：稳定身份、模板清理、可选领域 metadata、token 上限与 overlap、上下文精确预算与证据一致性、空证据、重写回退、真实 FAISS/BM25 保存与重载、中文索引路径、语料变更、模型/分块配置变更、损坏索引及保存失败恢复、公开语料长文档。

2026-09-21 的 LangChain 1.x 迁移环境冒烟记录：Python 3.12.14、CPU 版 Torch 2.14.0、Transformers 5.17.0、Sentence Transformers 5.7.0 环境下，真实 BGE 的 query/document embedding 均为 512 维；公开语料加载 322 篇文档并生成 1804 个最终 Chunk，最大 446 tokens；FAISS 成功建库、保存、重载并返回真实 Chunk，BM25、Hybrid/RRF 与 Evidence/Context 同步通过。选择 Sentence Transformers 5.7.0 是为了满足 `langchain-huggingface==1.2.2` 的官方 `full` extra 所声明的 `<6.0` 兼容范围。该历史记录仅证明当时的检索链路可用，不证明检索效果；当时未完成外部 LLM E2E。

### 真实 LLM HTTP 验证与人工浏览器验收

当前真实 LLM E2E 使用 DeepSeek `deepseek-chat` OpenAI-compatible API 完成验证，提供商和模型仍由本地 `.env` 决定。真实 FastAPI `/api/v1/chat` 已对糖醋鲤鱼原料、宫保鸡丁做法、老式锅包肉炸制、戚风蛋糕长章节、带 `source` 过滤的空气炸锅羊排以及 Kubernetes 无答案问题分别得到 HTTP 200。前五条返回 Answer + Sources；无答案请求返回固定安全拒答和空 Sources。三条抽查请求的 API Sources 与实际进入 Prompt Context 的 Evidence 一致。锅包肉案例存在已知漏答：原文有炸制次数和油温，但该次进入 Prompt 的 Chunk 只包含复炸后段，模型据此回答证据不足。此验证不是生成答案准确率评估，也不是浏览器视觉验收。

浏览器视觉验收请手动执行：

1. 在项目根目录启动 FastAPI：`.venv-lc1/Scripts/python.exe -m uvicorn api:app --app-dir code/QAR --host 127.0.0.1 --port 8000`。
2. 在 `code/QAR/frontend` 中设置 `.env.local` 的 `VITE_API_BASE_URL=http://localhost:8000`，执行 `npm run dev`。
3. 浏览器打开 `http://localhost:5173`（若本机 localhost 解析异常，改用 `http://127.0.0.1:5173`）。
4. 提交“糖醋鲤鱼需要哪些原料？”，检查 Answer、hybrid mode、count 与 Sources。
5. 提交“如何在三台服务器上搭建 Kubernetes 高可用集群？”，检查固定安全拒答、`sources=[]`、`count=0`。

若本机 8000 端口已被占用，可将启动命令中的端口改为可用端口，并同步修改前端 `.env.local` 的 `VITE_API_BASE_URL`；修改后重新启动 Vite。

本轮未自动操作浏览器，页面显示效果仍需人工确认。无效 LLM Key 的真实调用返回 HTTP 502；前端错误映射会显示“外部大模型服务不可用”的友好提示。
