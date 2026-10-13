# QAR 通用 Markdown 知识库 RAG：项目技术说明书

> 本说明书对应当前冻结版本。事实基线：322 篇公开 Markdown、1804 个 Chunk、BGE 512 维向量、79 项 pytest；开发 Evaluation 为 27 条有答案 + 12 条无答案，独立 Holdout 为 10 条有答案 + 12 条无答案。真实 LLM HTTP E2E 已使用 DeepSeek `deepseek-chat` OpenAI-compatible API 验证；浏览器视觉验收仍需人工执行。

## 一、3 分钟项目介绍

下面这段可以直接作为面试口述稿，正常语速约 3 分钟。

> 我做的是一个面向通用 Markdown 文档的轻量知识库 RAG。项目最早是食谱问答原型，公开食谱现在只作为可复现测试语料；后来我把数据模型、检索和生成链路改成不依赖菜名、分类或难度字段的通用实现。
>
> 我主要想解决三个问题。第一，Markdown 文档既有标题结构，也可能有很长的章节，简单按固定字符切分会破坏章节语义，而只按标题切分又可能超过 Embedding 的输入上限。第二，Dense Retrieval 对自然语言语义比较好，但专有词和精确关键词可能漏检；BM25 擅长词面匹配，但不理解同义表达，所以需要把两条路线稳定地融合。第三，检索返回 Top-K 并不表示知识库真的能回答，如果直接把无关内容交给大模型，会产生看似合理的回答和错误来源。
>
> 离线建库时，我先读取 Markdown，排除模板和空文件，按标题切 Section；超过 BGE 安全 token 阈值的 Section 再递归切分并保留 overlap。`document_id` 由相对路径做 SHA-256，`chunk_id` 再结合文档 ID、章节位置和内容生成，保证索引重建后的身份稳定。Chunk 一路使用 `BAAI/bge-small-zh-v1.5` 生成归一化的 512 维向量，存入原生 FAISS `IndexFlatL2`；另一路用 `rank_bm25` 和自己实现的中文单字、双字 tokenizer 建 BM25。查询时两路各取 20 个候选，用 RRF 按排名融合，最终返回 5 个 Chunk。
>
> 我重点处理了几个容易被 Demo 忽略的问题。索引 metadata 会记录语料 hash、Chunk hash、Embedding 模型、分块版本和索引文件 hash，只有全部一致才复用 FAISS；BM25始终从本次内存中的相同 Chunk 构建，避免原文、向量和稀疏索引版本错位。Metadata Filter 也不是在全局 Top-K 之后再删除，而是先形成允许的 `chunk_id` 集合，让 Vector、BM25 和 Hybrid 都在过滤后的候选空间里检索。最终进入 Prompt 的内容会转换成带 `S1`、`S2` 编号的 Evidence，返回的 Sources 只来自实际装入 Context 的 Evidence。
>
> 在效果验证上，我人工标注了 27 条有答案问题和 12 条无答案问题。文档级结果是：Vector 的 Recall@3、Recall@5、MRR 分别是 0.8889、0.9630、0.8420；BM25 是 1.0、1.0、0.9568；Hybrid 是 1.0、1.0、0.9815。Evaluation 还暴露了 Metadata Filter 漏召回和知识库外问题仍会返回食谱 Top-K，我分别修复了候选生成，并基于开发集设置了 0.45 的轻量相关性 Gate。之后用完全独立的 10 条答案、12 条无答案 Holdout 固定验证，答案误拒是 0，无答案拒绝 9 条、误接受 3 条。
>
> 系统通过 `RAGService` 封装成 FastAPI，只提供 `/chat`、`/health` 和同步 `/reindex`。它是一个完整、可测试的个人项目，不是生产级平台。当前限制包括数据集较小、语料只来自公开食谱、相关性是文档级标注、Gate 仍有 25% 的 Holdout 无答案误接受，而且 Sources 只能证明证据进入 Prompt，不能证明模型每句话都被严格支撑。真实 OpenAI-compatible LLM HTTP E2E 已验证，浏览器视觉验收仍待人工执行。

## 二、完整架构

### 2.1 离线建库与在线问答

```mermaid
flowchart TD
    subgraph Build[离线或同步重建]
        M[Markdown 文件] --> L[加载、清洗、排除模板]
        L --> H[Markdown Header 结构切分]
        H --> T{token 数 <= 450?}
        T -- 是 --> C[保留完整 Section]
        T -- 否 --> R[RecursiveCharacterTextSplitter 二次切分 + overlap]
        C --> ID[Stable document_id / chunk_id]
        R --> ID
        ID --> E[BGE 归一化 Embedding]
        E --> F[Native FAISS IndexFlatL2]
        ID --> B[中文 unigram + bigram]
        B --> BM[BM25Okapi]
        ID --> IM[Index metadata / corpus hash / chunk hash]
        F --> IM
    end

    subgraph Online[在线查询]
        HTTP[POST /api/v1/chat] --> S[RAGService]
        S --> Q[MarkdownRAGSystem.ask_question]
        Q --> MF[Metadata Candidate Filtering]
        MF --> VF[BGE Query Embedding + FAISS Top-20]
        MF --> BF[BM25 Top-20]
        VF --> RF[RRF by stable chunk_id]
        BF --> RF
        RF --> K[Final Top-5 Retrieval Documents]
        K --> G{max BGE cosine >= 0.45?}
        G -- 否 --> N[安全拒答，不调用 LLM，sources=[]]
        G -- 是 --> EV[Evidence S1...Sn]
        EV --> CX[2000 字符有界 Context]
        CX --> P[通用知识库 Prompt]
        P --> LLM[OpenAI-compatible LLM]
        LLM --> AR[Answer + Sources]
        AR --> RESP[ChatResponse]
    end
```

注意实际代码顺序：Relevance Gate 检查的是 RRF 返回的 `Document` Chunk，Gate 通过后才由 `GenerationIntegrationModule` 创建正式的 `Evidence` 对象。图中的 Final Top-5 可以理解为“候选证据 Chunk”。

### 2.2 模块逐项说明

#### 1. Markdown 文档解析

1. **解决什么问题**：把目录中的公开 Markdown 变成统一的 LangChain `Document`，并保留来源和标题。
2. **为什么需要**：检索必须有稳定文本和 metadata；模板、空文件、注释和图片链接会制造无效 Chunk。
3. **当前实现**：`Path.rglob` 按路径排序，只接受 `.md`；跳过 `template/templates/示例菜`；检索文本去除 HTML 注释和图片链接，父文档仍保留原始正文。
4. **输入**：配置中的 `data_path` 目录。
5. **输出**：父级 `Document` 列表，metadata 至少包含 `source/document_id/parent_id/title/doc_type`。
6. **删除后的影响**：无法建立统一语料、来源追踪和后续稳定 ID；模板内容也会进入知识库。
7. **代码证据**：`rag_modules/data_preparation.py` 的 `DataPreparationModule.load_documents()`、`_clean_content()`、`_extract_title()`。

#### 2. Markdown Header + Token-aware Chunking

1. **解决什么问题**：同时保留标题语义边界并避免超长章节被 BGE 截断。
2. **为什么需要**：只按标题切分时，一个 Section 仍可能远超模型长度；纯固定字符切分又会把标题和正文关系切散。
3. **当前实现**：先用 `MarkdownHeaderTextSplitter` 按 `# / ## / ###` 切分并保留标题；token 数超过 450 时，用 `RecursiveCharacterTextSplitter` 按段落、换行、中文标点、空格逐级切分，必要时继续缩小字符窗口，直到所有块满足 token 预算。
4. **输入**：父文档正文、Embedding tokenizer、`max_section_chars=1200`、`section_chunk_overlap=100`、`embedding_chunk_max_tokens=450`。
5. **输出**：子级 `Document`，保留 `title/section/source/document_id/parent_id/section_index/subchunk_index`。
6. **删除后的影响**：删除 Header 切分会损失章节上下文；删除 token 兜底会让超长输入被模型截断；删除 overlap 会增加答案跨边界丢失风险。
7. **代码证据**：`DataPreparationModule._markdown_header_split()`、`_split_section()`、`_token_count()`；测试见 `tests/test_phase2_retrieval.py` 的 token budget 和 overlap 用例。

#### 3. Stable ID 与索引生命周期

1. **解决什么问题**：让同一文档和 Chunk 在重建后保持可追踪身份，并阻止旧 FAISS 与新语料或 BM25 混用。
2. **为什么需要**：RRF 去重、Evidence 来源、持久化位置映射和索引一致性都依赖稳定身份。
3. **当前实现**：`document_id=SHA256(相对路径)`；`chunk_id=SHA256(document_id + section/subchunk 位置 + content)`。索引 metadata 保存 schema、语料 hash、Chunk hash、模型、分块版本、数量和索引文件 hash；只有全部匹配才复用 FAISS。BM25 每次由当前内存 Chunk 构建。
4. **输入**：相对路径、Chunk 内容与位置、当前配置、索引文件。
5. **输出**：稳定 ID、`index_metadata.json`、可复用或需重建的判断。
6. **删除后的影响**：同一 Chunk 可能被 RRF 当成两个结果；Sources 无法稳定回溯；语料更新后可能继续使用旧向量。
7. **代码证据**：`DataPreparationModule._stable_chunk_id()`；`rag_modules/index_metadata.py`；`MarkdownRAGSystem.build_knowledge_base()`；`tests/test_data_preparation.py` 和 `tests/test_index_lifecycle.py`。

#### 4. BGE Embedding

1. **解决什么问题**：把 Query 和 Chunk 映射到同一语义向量空间，支持不依赖完全相同关键词的检索。
2. **为什么需要**：用户表达可能与文档原词不同，单纯 BM25 会漏掉语义相近内容。
3. **当前实现**：`HuggingFaceEmbeddings` 加载 `BAAI/bge-small-zh-v1.5`，CPU 推理，`normalize_embeddings=True`；真实输出为 512 维。
4. **输入**：Query 字符串或 Chunk 文本列表。
5. **输出**：Query 的 512 维向量或文档向量矩阵。
6. **删除后的影响**：只剩词面检索，对自然语言改写和语义表达更敏感。
7. **代码证据**：`rag_modules/index_construction.py` 的 `setup_embeddings()`、`build_vector_index()`；真实冒烟脚本 `tools/smoke_real_bge_faiss.py`。

#### 5. Native FAISS

1. **解决什么问题**：在本地对 1804 个归一化向量做精确近邻检索，并保存/加载索引。
2. **为什么需要**：避免每次查询都对全部向量用 Python 手工比较，同时保持本项目规模下的简单性。
3. **当前实现**：`faiss.IndexFlatL2`；`position -> chunk_id -> Document` 两级映射；普通查询使用 `index.search`，带 metadata filter 时只重构允许位置的向量并计算 L2；索引通过内存序列化支持 Windows 中文路径。
4. **输入**：512 维 Query 向量、`k`、可选允许 `chunk_id` 集合。
5. **输出**：按距离排序的 Chunk `Document`。
6. **删除后的影响**：没有 Dense Retrieval；只靠 BM25 难以覆盖语义查询。
7. **代码证据**：`rag_modules/native_faiss.py` 的 `NativeFAISSIndex`；`rag_modules/index_construction.py`；中文路径测试在 `tests/test_index_lifecycle.py`。

#### 6. 中文 BM25

1. **解决什么问题**：补充专有名词、菜名、原料、数字和精确词面匹配。
2. **为什么需要**：Dense Retrieval 会把语义相近内容放在前面，但可能漏掉用户明确写出的稀有词或精确实体。
3. **当前实现**：直接使用 `rank_bm25.BM25Okapi`；中文连续片段生成单字和相邻双字，英文、数字和下划线片段转小写后作为 token。
4. **输入**：所有 Chunk 文本和 Query。
5. **输出**：BM25 分数及按分数排序的 Chunk，并写入 `bm25_rank`。
6. **删除后的影响**：q024 这类 Dense 失败但关键词明确的问题可能无法由 Hybrid 修复。
7. **代码证据**：`rag_modules/retrieval_optimization.py` 的 `NativeBM25Retriever`、`RetrievalOptimizationModule.tokenize()`。

#### 7. RRF 融合

1. **解决什么问题**：融合 FAISS 和 BM25 两套不可直接比较的分数。
2. **为什么需要**：L2 距离和 BM25 分数尺度、方向、分布都不同，直接相加没有稳定含义。
3. **当前实现**：每一路按排名对稳定 `chunk_id` 累加 `1/(rrf_k + rank)`；`rrf_k=60`；保留 `vector_rank/bm25_rank/rrf_score`，按总分降序、`chunk_id` 作为确定性 tie-break 排序。
4. **输入**：Vector Top-20 与 BM25 Top-20。
5. **输出**：融合后的候选 Chunk，最终截取 5 个。
6. **删除后的影响**：必须选择单路检索或人为归一化两种分数；q024 的单路失败不能由当前 Hybrid 方式修复。
7. **代码证据**：`RetrievalOptimizationModule._rrf_rerank()`、`retrieve()`。

#### 8. Metadata Candidate Filtering

1. **解决什么问题**：让 `filters={"source": ...}` 的含义成为“在满足条件的 Chunk 集合内检索”。
2. **为什么需要**：如果先全局取 Top-K 再过滤，目标文档可能根本没进入 Top-K，最终会错误返回空结果。
3. **当前实现**：先扫描 Chunk metadata 得到允许的 `chunk_id` 集合；Vector、BM25、Hybrid 都使用该集合生成候选；末端仍做一次防御性过滤。字段没有写死，支持任意 metadata 的标量精确匹配或集合匹配。
4. **输入**：Query、通用 `filters` 字典、全部 Chunk metadata。
5. **输出**：只来自允许集合的 Vector/BM25/Hybrid 结果。
6. **删除后的影响**：q027 会重现“明确指定 source 仍返回空”的漏召回。
7. **代码证据**：`RetrievalOptimizationModule._matching_chunk_ids()`、`_metadata_matches()`、`retrieve()`；测试 `test_metadata_filter_limits_candidate_generation_before_top_k`。

#### 9. Relevance Gate

1. **解决什么问题**：区分“检索器总能返回 Top-K”和“这些候选是否至少与 Query 足够相关”。
2. **为什么需要**：FAISS 和 BM25 对知识库外问题也会返回排序靠前的食谱，直接交给 LLM 会增加错误回答风险。
3. **当前实现**：对最终 Hybrid Chunk 复用当前 BGE Query 向量和 FAISS 中已存向量计算余弦相似度，取最大值；低于固定 `0.45` 时清空 Chunk，生成模块返回安全拒答且不调用 LLM。
4. **输入**：检索 Query 和最终 Top-5 Chunk。
5. **输出**：保留 Chunk，或空列表。
6. **删除后的影响**：明显知识库外问题也会进入 Prompt，并返回无关 Sources。
7. **代码证据**：`NativeFAISSIndex.semantic_similarities()`；`MarkdownRAGSystem.ask_question()`；`tests/test_system.py` 的 Gate 用例。

#### 10. Evidence、Context 与 Prompt

1. **解决什么问题**：保证对外 Sources 与真正进入 Prompt 的内容一一对应，并控制长 Context。
2. **为什么需要**：直接返回 Retrieval Result 可能包含未装入 Prompt 的 Chunk；长首块也可能挤掉后续关键证据。
3. **当前实现**：检索 Chunk 转为不可变 `Evidence`，包含 `evidence_id/chunk_id/document_id/title/section/source/content`；`_build_context()` 在 2000 字符预算内为剩余 Evidence 分享空间，重新连续编号 S1…Sn；Prompt 要求只根据证据回答并引用编号。
4. **输入**：Gate 通过的 Chunk `Document` 列表。
5. **输出**：渲染后的 Context 字符串和实际选中的 Evidence 列表。
6. **删除后的影响**：来源可能无法追溯，API Sources 可能与 Prompt 不一致，长文档可能再次造成后续证据丢失。
7. **代码证据**：`rag_modules/models.py` 的 `Evidence/RAGAnswer`；`generation_integration.py` 的 `build_evidence()`、`_build_context()`、`_chain()`；`tests/test_context_builder.py`。

#### 11. OpenAI-compatible LLM 集成

1. **解决什么问题**：把结构化 Evidence 综合成自然语言答案。
2. **为什么需要**：检索只返回资料片段，不能自动组织成对用户友好的回答。
3. **当前实现**：通过 `langchain_openai.ChatOpenAI` 调用由 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 配置的 OpenAI-compatible Chat Completions，temperature 0.1、max tokens 2048。无证据时不会初始化或调用 LLM。
4. **输入**：系统 Prompt、Evidence Context、原始用户问题。
5. **输出**：字符串答案；内部 `RAGAnswer` 同时携带 Sources。
6. **删除后的影响**：系统只能返回检索片段，不能生成回答。
7. **代码证据**：`GenerationIntegrationModule.setup_llm()`、`_chain()`、`generate_basic_answer()`。真实 HTTP E2E 已用 DeepSeek `deepseek-chat` 验证；pytest 仍使用注入的模拟 Runnable。

#### 12. RAGService 与 FastAPI

1. **解决什么问题**：提供稳定 HTTP 边界，隔离 API Schema 与底层 RAG 生命周期。
2. **为什么需要**：如果路由直接操作 FAISS、BM25 和 Prompt，异常映射、初始化和测试会与算法代码耦合。
3. **当前实现**：FastAPI lifespan 调用 `RAGService.initialize()`；Service 暴露 `ask/health/reindex`，把内部 `RAGAnswer` 转换为 Pydantic Response；API 映射 422、503、502、500 等主要错误。
4. **输入**：`ChatRequest`、health GET 或 reindex POST。
5. **输出**：`ChatResponse/HealthResponse/ReindexResponse`。
6. **删除后的影响**：项目只能通过 Python/CLI 调用，缺少可测试的服务接口；直接耦合路由会让职责混乱。
7. **代码证据**：`api.py`、`services/rag_service.py`、`schemas.py`、`tests/test_api.py`。

## 三、一次 Query 的真实调用链

示例请求：`糖醋鲤鱼需要哪些原料？`

### 3.1 HTTP 与 Schema

客户端发送：

```http
POST /api/v1/chat
Content-Type: application/json

{"query":"糖醋鲤鱼需要哪些原料？","filters":{}}
```

`ChatRequest` 把 `query` 去除首尾空白，验证非空；`filters` 被验证为 `dict[str, scalar | list[scalar]]`。非法字段会因 `extra="forbid"` 被拒绝。

### 3.2 Service 到 Core

`api.chat()` 只调用：

```python
rag_service.ask(request.query, request.filters)
```

`RAGService.ask()` 检查 Retrieval 和 Generation 是否已初始化，再调用：

```python
MarkdownRAGSystem.ask_question("糖醋鲤鱼需要哪些原料？", filters=None)
```

Query Rewrite 默认关闭，所以检索 Query 与原始问题相同。

### 3.3 Hybrid Retrieval

`retrieve()` 收到：

```python
query = "糖醋鲤鱼需要哪些原料？"
filters = None
mode = "hybrid"
vector_fetch_k = 20
bm25_fetch_k = 20
final_k = 5
```

无过滤条件时 `allowed_chunk_ids=None`。

Dense 路线把字符串传给 `embed_query()`，得到 shape 为 `(512,)` 的归一化向量；Native FAISS 将其包装为 `(1, 512)` 的 `float32` 数组，对 `IndexFlatL2` 执行 Top-20 搜索。每个命中从 position 映射回稳定 `chunk_id`，再映射成 `Document`，并写入 `vector_rank`。

Sparse 路线把 Query 切成中文单字和相邻双字，例如包含“糖、醋、鲤、鱼、糖醋、醋鲤、鲤鱼……”；`BM25Okapi.get_scores()` 对 1804 个 Chunk 计算分数，取 Top-20，并写入 `bm25_rank`。

RRF 用稳定 `chunk_id` 合并两路。此次冻结代码实际运行得到的前 5 个 Chunk 为：

| Final rank | Source | Section | Vector rank | BM25 rank | RRF score | BGE cosine |
|---:|---|---|---:|---:|---:|---:|
| 1 | `dishes/aquatic/糖醋鲤鱼/糖醋鲤鱼.md` | 主标题 | 1 | 1 | 0.032787 | 0.7854 |
| 2 | 同上 | 操作 | 2 | 2 | 0.032258 | 0.7556 |
| 3 | 同上 | 必备原料和工具 | 3 | 4 | 0.031498 | 0.7264 |
| 4 | `dishes/aquatic/糖醋鱼.md` | 必备原料和工具 | 4 | 3 | 0.031498 | 0.7078 |
| 5 | 同上 | 主标题 | 9 | 8 | 0.029199 | 0.6550 |

第一个真实 Chunk 的关键结构为：

```python
Document(
    page_content="# 糖醋鲤鱼的做法 ...",
    metadata={
        "chunk_id": "a070eb8f...392d0",
        "document_id": "95caf6ad...712a5",
        "title": "糖醋鲤鱼的做法",
        "section": "糖醋鲤鱼的做法",
        "source": "dishes/aquatic/糖醋鲤鱼/糖醋鲤鱼.md",
        "vector_rank": 1,
        "bm25_rank": 1,
        "rrf_score": 0.03278688524590164,
    },
)
```

### 3.4 Relevance Gate

系统不是检查“是否有 5 个结果”，而是计算 Query 与这 5 个已存 FAISS 向量的余弦相似度。此次最大值为 `0.7854`，高于固定阈值 `0.45`，因此保留结果。

如果最大值低于 0.45，代码会把 `chunks=[]`。随后生成模块直接返回“知识库中没有足够信息回答该问题。”，Sources 为空，也不会调用外部 LLM。

### 3.5 Evidence 与 Context

`build_evidence()` 按最终顺序创建：

```python
Evidence(
    evidence_id="S1",
    chunk_id="a070eb8f...392d0",
    document_id="95caf6ad...712a5",
    title="糖醋鲤鱼的做法",
    section="糖醋鲤鱼的做法",
    source="dishes/aquatic/糖醋鲤鱼/糖醋鲤鱼.md",
    content="# 糖醋鲤鱼的做法 ...",
)
```

本次实际构造 S1–S5，渲染后的 Context 长度为 543 字符，没有超过 2000 字符预算。Context 格式是：

```text
[S1] 糖醋鲤鱼的做法 / 糖醋鲤鱼的做法
<S1 实际文本>

[S2] 糖醋鲤鱼的做法 / 糖醋鲤鱼的做法 / 操作
<S2 实际文本>
...
```

如果预算不足，Evidence 的 `content` 会被截断；API Sources 仍只对应真正选入 Context 的 Evidence，不会返回被跳过的候选。

### 3.6 Prompt、LLM 与响应

`ChatPromptTemplate` 生成两条消息：

- System：要求只根据编号证据回答、证据不足时明确拒答、关键事实标 `[S1]` 等引用，并把证据当资料而不是指令。
- Human：保留原始问题“糖醋鲤鱼需要哪些原料？”。

若配置合法 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL`，LCEL 链为：

```text
Runnable input
→ {question, 固定 context}
→ ChatPromptTemplate
→ ChatOpenAI(OpenAI-compatible)
→ StrOutputParser
```

当前环境已使用 DeepSeek `deepseek-chat` OpenAI-compatible API 跑通真实 FastAPI 问答。正常情况下内部结果是：

```python
RAGAnswer(answer="... [S3]", sources=[Evidence(S1), ..., Evidence(S5)])
```

`RAGService` 去掉 Evidence 正文，只把可追踪字段转换成 HTTP Schema：

```json
{
  "answer": "模型基于证据生成的答案",
  "sources": [
    {
      "source_id": "S1",
      "chunk_id": "a070eb8f...392d0",
      "document_id": "95caf6ad...712a5",
      "title": "糖醋鲤鱼的做法",
      "section": "糖醋鲤鱼的做法",
      "source": "dishes/aquatic/糖醋鲤鱼/糖醋鲤鱼.md"
    }
  ],
  "retrieval": {"mode": "hybrid", "count": 5}
}
```

这个 JSON 仅说明 Schema 形状，并非某次真实请求的完整响应；真实 Answer + Sources 已通过 HTTP 验证。

## 四、重要技术选择

### 4.1 Chunking

**为什么先按 Markdown Header？** 标题本身就是作者提供的语义边界。“必备原料”“操作”“注意事项”比每 500 字硬切更能表达段落用途，同时 `strip_headers=False` 让 Chunk 保留标题上下文。

**为什么还要 Token-aware 二次切分？** Markdown 结构不能保证 Section 短小；一个标题下可能有几千字。Embedding 以 token 为输入上限，只检查字符数会在中文、英文数字混排时产生误判。

**为什么不是固定字符切分？** 固定字符切分不能利用 Markdown 结构，而且字符数与 tokenizer token 数不是固定比例。当前实现仍用递归字符切分寻找自然边界，但停止条件由真实 tokenizer token 数决定。

**为什么是 450 tokens？** BGE 的最大输入长度是 512 tokens。450 是为特殊 token 和不同文本形态保留余量的工程安全值，不是 Evaluation 得出的最优 Chunk Size，也不应表述成最优参数。

**为什么 overlap？** 答案可能跨切分边界，100 字符 overlap 可以让相邻块共享边界上下文。这里配置的是字符 overlap，不是 100 tokens；测试只证明 overlap 存在和 token 上限满足，没有证明 100 是最优值。

### 4.2 Embedding

**为什么用 `BAAI/bge-small-zh-v1.5`？** 当前语料和问题以中文为主，该模型可以本地运行，规模适合 CPU 冒烟与个人项目，且已通过真实加载、Query/Document Embedding 和 FAISS 建库验证。选择理由是适配与成本平衡，不是声称它优于所有模型。

**维度是多少？** 512 维，真实 FAISS 索引也是 `d=512`。

**最大输入是多少？** tokenizer 最大序列长度 512 tokens；项目使用 450 的安全阈值。

**为什么没有换 Qwen3 Embedding？** 项目目标是验证完整工程闭环，而不是持续做模型排行榜。现有 BGE 已形成固定 Evaluation 和 Holdout 基线；没有在相同数据上进行受控对比，就不能声称换模型会提升，也不应为了简历堆模型名破坏冻结基线。

### 4.3 FAISS

**为什么使用 Native FAISS？** 项目只有 1804 个向量，需要的是精确检索、持久化和稳定身份映射。直接使用 `faiss-cpu` 可以去掉已停止维护的 `langchain-community` wrapper，同时保持上层 `retrieve()` 不变。

**为什么不用 Milvus、Elasticsearch 或 Chroma？** 当前是单机、小语料、同步重建，没有分布式扩展、服务化向量库或复杂倒排需求。引入外部服务会增加部署和运维成本，不能自动提高检索质量。

**为什么 `IndexFlatL2`？** 它不需要训练、是精确搜索，对 1804 个向量计算成本可接受，行为也容易测试和解释。项目没有证明 HNSW、IVF 等近似索引能带来必要收益。

**归一化向量 + L2 与 Cosine 的关系？** 对单位向量 `x,y`：

```text
||x-y||² = ||x||² + ||y||² - 2 x·y = 2 - 2 cos(x,y)
```

因此按平方 L2 从小到大与按 cosine 从大到小排序一致。项目建库和 Query 都通过相同 BGE 配置生成归一化向量；Gate 仍显式计算 cosine，含范数除法，避免只靠假设读取分数。

**stable chunk_id 如何对应 position？** `index_to_docstore_id: dict[int,str]` 保存 FAISS position 到 `chunk_id`，`documents_by_id: dict[str,Document]` 保存 ID 到 Chunk。加载时 `_validate()` 检查 position 是否覆盖 `0..ntotal-1`、ID 是否唯一、两个映射集合是否一致。

### 4.4 BM25

**原理是什么？** BM25 对 Query 中每个 token 计算 IDF，并用词频饱和与文档长度归一化调整贡献。常见单词项形式为：

```text
IDF(q) × f(q,D) × (k1+1)
---------------------------
f(q,D) + k1 × (1-b+b×|D|/avgdl)
```

项目直接使用 `rank_bm25.BM25Okapi` 的默认参数，没有自研 BM25 数学实现或调参。

**为什么 Dense 之外还需要 BM25？** 精确菜名、原料、数字和步骤关键词往往适合词面检索。q024 中 Vector 在 Top-5 没有找到老式锅包肉，BM25 Rank 1 命中，Hybrid 因而恢复命中。

**为什么中文需要 tokenizer？** 英文可以按空格切词，中文通常没有空格。若把整段中文当一个 token，Query 与 Chunk 很难共享词项。

**当前 tokenizer 怎么工作？** 正则提取连续中文或 `[A-Za-z0-9_]` 片段；中文片段生成每个单字和所有相邻双字，ASCII/数字转小写作为整体 token。例如“宫保鸡丁 restore-123”包含“宫”“宫保”“鸡丁”“restore”“123”。

**缺陷是什么？** 它不是真正中文分词，没有词性、词典、停用词、同义词和领域实体识别；长中文片段会生成很多 token；单字可能带来噪声；BM25 IDF 也仍基于全语料。它的优势是确定、无新增服务、容易复现。

### 4.5 RRF

**RRF 是什么？** Reciprocal Rank Fusion 只使用各检索器的排名，不直接比较原始分数。

**公式是什么？** 对 Chunk `d`：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

当前只有 Vector 和 BM25 两路，`k=60`。

**为什么不直接相加？** FAISS 的 L2 越小越好，BM25 越大越好，量纲和分布不同；未经校准的加法权重没有依据。RRF 把两路都转换成排名贡献。

**`rrf_k=60` 的意义？** 它使前几名之间的贡献差异更平滑，避免 Rank 1 完全压倒后续结果。当前值是冻结默认值，没有通过参数搜索证明为最优。

### 4.6 Metadata Filter

**最初为什么失败？** 原流程是全局检索并截断 Top-K，然后删除不符合 source 的结果。q027 的目标 Chunk 没进入全局候选，所以过滤后为空。

**为什么 Top-K 后过滤会漏召回？** 过滤条件限定了用户的搜索空间。先截断相当于先在错误空间里做选择，后续删除无法找回已经被截掉的目标。

**现在怎样工作？** `_matching_chunk_ids()` 先扫描全部 Chunk，找出 metadata 满足条件的 ID。FAISS 只比较这些位置的向量；BM25 只对这些位置排序；Hybrid 融合两路受限候选。

**q027 如何修复？** 对 `filters={"source":"dishes/semi-finished/空气炸锅羊排/空气炸锅羊排.md"}`，三种 mode 都在该 source 的 Chunk 中检索，目标文档从空结果变为 Rank 1。

### 4.7 Relevance Gate

**为什么 Top-K 非空不代表能回答？** 排序算法对任何 Query 都能给出“相对最接近”的文档，即使所有文档都绝对不相关。开发集最初的 Kubernetes、量子物理等问题仍会收到食谱 Top-K。

**为什么使用 BGE cosine？** Query 与 Chunk 的 BGE 向量已经存在，无需增加模型；cosine 直观表达方向相似性。实现复用 FAISS 中的文档向量，只额外生成 Query 向量。

**0.45 从哪里来？** 开发集 27 个答案 Query 的最低最大相似度为 0.4999；12 个无答案的最高值为 0.5165，尾部有重叠。0.45 在开发集上误拒 0/27、拒绝无答案 9/12、误接受 3/12，属于保护答案召回的保守阈值。

**Holdout 怎么样？** 固定 0.45 后，10 个答案误拒 0；12 个无答案拒绝 9、误接受 3，结果没有用于再次调阈值。

**为什么不是可靠 OOD Detector？** 开发集两类尾部重叠，Holdout 仍有 25% 无答案误接受；相似度受措辞和语料表面词影响。它没有经过多领域大样本校准，也没有概率意义。

**h007 说明什么？** h007 是拌饭酱汁问题。BM25 找到标注文档，Hybrid Final-5 却没有目标文档，但其他候选的最大相似度为 0.7347，Gate 仍放行。Gate 检查“是否看起来相关”，不能检查“最终证据是否真的包含答案”。

### 4.8 Evidence / Sources

**Evidence 与 Retrieval Result 有什么区别？** Retrieval Result 是带排名 metadata 的 LangChain `Document`；Evidence 是面向生成和引用的不可变业务对象，具有 S 编号和完整追踪字段，并且可能因 Context 预算截断正文。

**为什么 Sources 必须来自实际 Evidence？** 用户看到的来源必须与模型看到的材料一致。项目先打包 Context，再返回同一批 `selected Evidence`，避免把检索到但未进入 Prompt 的文档当成回答来源。

**Sources 能证明答案正确吗？** 不能。Sources 只能证明这些片段被提供给模型。当前没有逐句事实蕴含检查、引用编号校验或 LLM-as-a-Judge，模型仍可能误读、漏引或生成证据外内容。

### 4.9 FastAPI

**为什么增加 Service 层？** `RAGService` 管理初始化、调用和内部模型到 API Schema 的转换；API 路由只负责 HTTP，便于 Mock 和异常映射。

**为什么 API 不直接操作 FAISS/BM25？** 检索策略已经集中在 Core；路由直接访问底层会重复逻辑，让 HTTP 与算法耦合，并可能绕过 Gate、Evidence 和索引生命周期。

**为什么只有三个接口？** `/chat` 覆盖核心使用，`/health` 暴露真实加载状态，`/reindex` 复用完整重建链路。个人项目没有聊天历史、文档上传、用户和任务管理需求。

**为什么没有 Redis、数据库和异步任务？** 当前 322 文档、1804 Chunk，重建允许同步完成；没有用户状态、分布式节点或高并发指标来证明这些基础设施必要。同步 reindex 是已知限制，不包装成生产能力。

## 五、Retrieval Evaluation

### 5.1 数据集与标注

开发 Evaluation 共 39 条：

- 有答案 27 条：实体 5、原料/关键词 6、自然语义 5、模糊表达 4、长章节 4、Metadata 3。
- 无答案 12 条：不计入 Recall/MRR，只观察错误候选和相似度。

采用 document-level annotation。`relevant_documents` 保存相对 source 路径，运行时解析成稳定 `document_id`。同一文档命中多个 Chunk 时，Top-K 文档列表按第一次出现去重。它没有标注“哪个 Chunk 包含完整答案”。

### 5.2 三个冻结 Baseline

- **Vector**：BGE + Native FAISS。
- **BM25**：中文 unigram/bigram + `BM25Okapi`。
- **Hybrid**：上述两路各取 20，RRF 后取 5。

三者都调用同一个 `retrieve()`，没有为 Evaluation 写另一套检索，也没有在第一轮中调 K、tokenizer 或 RRF。

### 5.3 指标

**Recall@3**：相关文档中有多少比例出现在前三个去重文档。当前每题通常标一个相关文档，所以 q024 的 Vector 目标未进入前三，Recall@3=0；BM25 Rank 1，Recall@3=1。

**Recall@5**：同理检查前五，反映扩大结果窗口后能否找回目标。Vector 总体从 Recall@3 0.8889 升到 Recall@5 0.9630，说明部分目标排在第 4–5 位或仍有一次未命中。

**MRR**：第一个相关文档排名的倒数。Rank 1 得 1，Rank 2 得 0.5，未命中得 0，再对问题求平均。q022 中 Vector Rank 1 的 MRR=1，Hybrid Rank 2 的 MRR=0.5，所以虽然都命中，Hybrid 排名更差。

### 5.4 最终结果

| Method | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|
| Vector | 0.8889 | 0.9630 | 0.8420 |
| BM25 | 1.0000 | 1.0000 | 0.9568 |
| Hybrid | 1.0000 | 1.0000 | 0.9815 |

这些数字只说明当前公开食谱语料和 27 个有答案问题上的文档级表现，不能推广成企业知识库准确率，也不是统计显著的通用结论。

### 5.5 三个 Case Study

**q022：Hybrid 可能降低单题排名。** 问题是戚风蛋糕三次加糖状态。Vector 把目标排 Rank 1；BM25 先排了魔芋蛋糕，目标 Rank 2；RRF 后目标仍为 Rank 2。因此 Hybrid Recall 仍为 1，但 MRR 从 Vector 的 1 降到 0.5。融合不是每题都优于最佳单路。

**q024：BM25 修复 Dense 漏检。** 问题是老式锅包肉炸几次和油温。Vector Top-5 没有目标；BM25 依靠精确关键词把目标排 Rank 1；Hybrid 也恢复 Rank 1。这是保留稀疏检索的直接证据。

**q027：Evaluation 发现 Filter 语义错误。** 问题本身只有“翻面以后还需要加热多久”，但 filter 指定空气炸锅羊排 source。修复前全局 Top-K 后过滤导致三种模式全空；修复候选生成后，三种模式都在指定 source 内把目标排 Rank 1。整体指标的增长全部由这个 Bug 修复解释，不是参数调优。

## 六、独立 Holdout Validation

### 6.1 为什么需要 Holdout

0.45 是观察开发 Evaluation 分布后选出的。如果仍用同一批数据宣称它能泛化，会把“选阈值”和“验证阈值”混为一谈。因此阈值冻结后另建 22 条问题：10 条全新的答案问题来自此前未用于答案评估的文档，12 条无答案覆盖天文、地质、网络安全、土木、语言学、经济、心理、电子、文学、生态、体育和艺术。

Holdout 只能验证一次既定规则，不能再用于把阈值调到更好看，否则它会变成新的开发集。

### 6.2 固定 0.45 的结果

| 指标 | 数量 | 比例 |
|---|---:|---:|
| Answer false reject | 0/10 | 0% |
| No-answer reject | 9/12 | 75% |
| No-answer false accept | 3/12 | 25% |

Answer 最大相似度分布：min 0.6693、median 0.7578、mean 0.7614、max 0.8322。

No-answer 最大相似度分布：min 0.3744、median 0.4209、mean 0.4339、max 0.5116。

### 6.3 边界失败

- **钢筋混凝土养护 h014，0.5116**：与食谱中“操作、养护、方式”等一般性过程表达产生语义接近，Gate 误接受。
- **MOSFET 开关损耗 h018，0.4734**：技术 Query 与语料中的功率、加热、时间等词形成弱表面或语义关联，超过阈值。
- **珊瑚白化 h020，0.4956**：水温、海水、水产类食谱造成领域词重叠，Gate 误接受。
- **h007**：它不是 false reject，而是 Retrieval 质量与 Gate 目标不一致。Hybrid 漏掉正确文档，但错误候选仍有高相似度，说明 Gate 不验证答案覆盖性。

结论只能表述为：0.45 是一个“轻量、保守的相关性拒答机制”，不能表述为“可靠 OOD 检测”。

## 七、项目真实演进过程

### 7.1 食谱问答原型 → 通用 Markdown 模型

最初代码围绕食谱的 list/detail/general 路由和 category、difficulty 等字段组织。问题是核心链路依赖领域字段，换成运维手册或制度文档就不成立。改造后删除领域路由，`source/title/section/document_id/chunk_id` 成为通用字段；食谱 metadata 只作为默认关闭的可选适配器。测试验证普通 Markdown 不出现菜品字段。

### 7.2 随机身份 → Stable ID

早期不稳定 Chunk 身份会让索引重建、RRF 去重和来源映射失去依据。现在文档路径决定 `document_id`，内容和结构位置决定 `chunk_id`。测试覆盖增加无关文件后 ID 不变、内容修改后文档 ID 不变但 Chunk ID 改变、相同文本不同 source 不合并。

### 7.3 长文档父文档替换 → 命中 Chunk 进入 Context

早期如果命中子块后把整篇父文档塞进 Context，长文档前部会耗尽预算，真正命中的后部章节反而丢失。现在保留实际命中 Chunk，并用共享预算避免第一个长 Chunk 挤掉后续 Evidence。父子映射仍用于追溯，不自动扩展整篇父文档。

### 7.4 各索引独立变化 → 同一语料版本生命周期

FAISS、BM25 和原文如果在不同时间构建，会出现 position、ID 和内容错位。现在一次 `build_knowledge_base()` 只读取一次语料；当前 Chunk 同时用于 FAISS 和 BM25；metadata 和文件 hash 不匹配就整体重建；禁止单独 append 向量。

### 7.5 简单向量检索 → 中文 BM25 + RRF

Dense 对语义表达有效，但 q024 等精确实体问题说明词面检索仍有价值。项目加入确定性的中文单字/双字 tokenizer 和 `rank_bm25`，再用 RRF 融合排名，保留每一路 rank 便于分析。没有引入 Elasticsearch 或 Reranker。

### 7.6 Header 分块 → Token-aware 兜底

最初只用 Markdown Header，后来发现 `1200` 字符不能保证低于 BGE 512-token 上限。改成复用真实 tokenizer 计数，超过 450 token 才二次递归切分，并测试中文、英文数字混排、metadata、稳定 ID 和 overlap。

### 7.7 LangChain 0.3 → 拆包后的 1.x 生态

项目曾使用 LangChain 0.3.x。迁移时没有重写 RAG，而是把实际依赖收敛到 `langchain-core`、`langchain-huggingface`、`langchain-text-splitters`、`langchain-openai`。顶层 `langchain`、`langchain-classic` 均不需要。

### 7.8 去除 `langchain-community` → Native FAISS/BM25

审计发现 community 只承担 FAISS wrapper 和 BM25Retriever，而项目已经直接依赖 `faiss-cpu`、`rank_bm25` 并拥有自己的生命周期与统一检索层。因此只在 Index/Retrieval 层小范围替换，保持 `retrieve()` 外部接口，完整测试和真实 BGE 冒烟通过。

### 7.9 Python Core → 最小 FastAPI

Core 稳定后才增加 `RAGService` 和三个接口，没有让 API 路由接触 FAISS、BM25、Chunk 或 Prompt。LLM 用 Mock 覆盖 API 测试；真实状态由 health 动态读取，不硬编码 322/1804。

### 7.10 经验判断 → Retrieval Evaluation

项目没有先声称 Hybrid 更好，而是建立人工 document-level 数据集比较三条固定 Baseline。结果显示 Hybrid 总体 MRR 更高，也暴露 q022 单题退化、q024 的稀疏检索价值和 q027 Filter Bug。

### 7.11 Top-K 后过滤 → Candidate Filtering

q027 证明旧 filter 语义错误后，修改为检索前限定候选集，并对 Vector/BM25/Hybrid 同时补回归测试。参数没有改，指标变化只来自 Bug 修复。

### 7.12 “有结果即可回答” → Relevance Gate + Holdout

无答案 Query 总能返回食谱，促使项目记录 BGE 相似度分布。开发集支持保守的 0.45 Gate，但分布尾部重叠，所以又建立独立 Holdout。Holdout 原样报告 3 个 false accept 和 h007，不再调阈值。

### 7.13 为什么停止继续增加技术

当前目标是应届生可解释、可测试的完整项目。Reranker、GraphRAG、Agent、Milvus 等没有被现有数据证明为必要；继续加入会扩大依赖和面试承诺，却不能替代真实 Evaluation。项目在 Core、API、开发集和 Holdout 都形成闭环后冻结。

## 八、当前真实限制与面试表达

### 8.1 限制清单

1. **Evaluation 数据量小**：开发集只有 27 个计分问题，Holdout 只有 22 条，不能支持统计显著结论。
2. **食谱语料领域限制**：322 篇公开食谱方便复现，但不能代表企业制度、运维、客服或研发文档。
3. **Document-level relevance**：指标判断目标文档是否出现，没有评价具体 Chunk 是否含完整答案。
4. **没有 Chunk-level answer annotation**：无法计算 Chunk 精确率、答案跨度覆盖率或 context recall。
5. **Gate 存在 false accept**：Holdout 12 个无答案中仍放行 3 个。
6. **高相似 Evidence 不等于能回答**：h007 中目标文档未进入 Hybrid Final-5，但无关候选相似度很高。
7. **Sources 不等于逐句 grounded**：来源确实进入 Prompt，但没有事实蕴含和引用一致性检查。
8. **同步 reindex**：请求会等待 322 文档完整处理，不适合大语料或严格延迟要求。
9. **没有并发索引切换**：重建不是后台构建后原子切换，没有蓝绿索引或版本流量切换。
10. **没有用户权限**：任何能访问服务的人都使用同一知识库。
11. **没有多租户**：进程只有一个 corpus 和一套索引。
12. **没有真正企业数据验证**：没有脱敏企业文档、真实用户或业务指标。
13. **真实 LLM HTTP E2E 的范围**：DeepSeek `deepseek-chat` 已通过六条 FastAPI 请求验证；尚未进行浏览器视觉验收，也不能把这些请求当成生成质量评估。
14. **没有高并发与高可用验证**：未做压测、多副本、限流、熔断或故障切换。

### 8.2 建议主动说明

在介绍结尾主动说四点即可：

- 数据集是小规模公开食谱，结果不能外推到企业知识库。
- Evaluation 是文档级，不是生成答案准确率。
- Relevance Gate 是轻量机制，Holdout 无答案误接受 3/12。
- 真实 DeepSeek OpenAI-compatible HTTP E2E 已完成；浏览器视觉验收仍需人工执行。

这四点直接限定核心结论，主动说明会显得可信。

### 8.3 被问到再说明

以下内容不需要在 3 分钟介绍里逐项展开，但被问到必须如实回答：同步 reindex、没有并发索引切换、无权限/多租户、没有数据库、没有压测、高可用、Chunk-level 标注、引用事实校验、BM25 tokenizer 局限、`rrf_k=60` 未调优。

## 九、简历真实性边界

### 9.1 可以写进简历

- 面向通用 Markdown 文档实现结构感知与 token-aware 分块，基于稳定 `document_id/chunk_id` 建立可追踪的索引和 Evidence 数据模型。
- 使用 BGE、Native FAISS、中文 BM25 和 RRF 实现统一 Hybrid Retrieval，并支持检索前通用 Metadata Candidate Filtering。
- 通过语料、Chunk、模型和索引文件 hash 管理 FAISS/BM25/原文一致性，支持索引保存、校验、复用与完整重建。
- 构建 Answer + Sources 链路，确保 API Sources 来自实际进入 Prompt 的 Evidence；加入基于 BGE cosine 的轻量相关性拒答。
- 使用 FastAPI 和 Pydantic 提供 `/chat`、`/health`、`/reindex`，并通过 79 项 pytest 覆盖核心检索、Evidence、生命周期和 API。
- 建立 39 条开发 Evaluation 与 22 条独立 Holdout；可写清 Vector/BM25/Hybrid 的文档级 Recall@3、Recall@5、MRR，以及 Holdout Gate 的 0/10 误拒、9/12 拒答结果，并同时注明数据范围。

### 9.2 可以面试讲，不建议占简历空间

- LangChain 0.3 到拆包 1.x 的迁移细节。
- 去除 `langchain-community` 和 Windows 中文路径持久化。
- schema version、文件 hash、失败保存时移除 success marker。
- q022、q024、q027 和 h007 的逐条分析。
- Query Rewrite 可配置且默认关闭、失败回退原 Query。
- BM25 tokenizer 的正则和 tie ordering。

这些内容适合证明你做过细节，但不应挤占简历的核心业务与结果描述。

### 9.3 绝对不能声称

- **企业级系统**：没有真实企业数据、权限、多租户、高可用和运维体系。
- **高并发**：没有 QPS、延迟压测、连接池或限流数据。
- **高可用**：没有多副本、故障转移、异地容灾。
- **准确率达到某数值**：项目报告的是小规模 document-level Recall/MRR，不是答案准确率。
- **性能提升 X%**：没有严格基准、重复实验和统计分析；过滤 Bug 修复后的指标变化不能包装成通用性能提升。
- **消除幻觉**：Prompt、Evidence 和 Gate 只能降低风险，不能证明消除幻觉。
- **可靠 OOD 检测**：Holdout false accept 为 3/12。
- **自研 FAISS**：项目调用 `faiss-cpu`，只实现了适合本项目的 adapter 和映射。
- **自研 BM25 算法**：数学实现来自 `rank_bm25.BM25Okapi`，项目实现的是 tokenizer、候选限制和统一接口。
- **自研 Embedding 模型**：使用公开的 `BAAI/bge-small-zh-v1.5`。
- **微服务架构**：当前是单进程 FastAPI + in-process RAG Core。
- **分布式系统或分布式向量库**：项目没有使用或实现。
- **任何 OpenAI-compatible 提供商或浏览器链路都已全面验证**：目前只对 DeepSeek `deepseek-chat` 完成真实 FastAPI HTTP E2E，浏览器视觉验收尚待人工执行。

## 十、代码与结果索引

| 内容 | 文件 |
|---|---|
| 配置 | `config.py` |
| 编排与 Gate | `main.py` |
| Markdown、Chunk、Stable ID | `rag_modules/data_preparation.py` |
| BGE 与索引构建 | `rag_modules/index_construction.py` |
| Native FAISS | `rag_modules/native_faiss.py` |
| BM25、RRF、Metadata Filter | `rag_modules/retrieval_optimization.py` |
| Evidence、Context、Prompt、OpenAI-compatible LLM | `rag_modules/models.py`、`rag_modules/generation_integration.py` |
| 索引生命周期 | `rag_modules/index_metadata.py` |
| FastAPI、Service、Schema | `api.py`、`services/rag_service.py`、`schemas.py` |
| 开发 Evaluation | `evaluation/questions.jsonl`、`evaluation/results/report.md` |
| 独立 Holdout | `evaluation/holdout_questions.jsonl`、`evaluation/holdout_results/report.md` |
| 测试 | `tests/`，最终 `79 passed` |
