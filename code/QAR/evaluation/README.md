# Retrieval Evaluation

本目录对冻结的 QAR Retrieval baseline 做小规模、文档级、离线评估。它只比较现有 `retrieve()` 的 `vector`、`bm25` 和 `hybrid` 三个 mode，不更换模型、不调参、不调用回答生成 LLM。

## Dataset

`questions.jsonl` 是人工阅读当前公开 Markdown 语料后建立的 39 条 Query（27 条有答案、12 条无答案）。`relevant_documents` 使用语料根目录下的相对路径，因为它比 SHA-256 `document_id` 更适合人工复核；脚本运行时会将路径解析为当前数据模型的稳定 `document_id`。

标注是 document-level relevance：只要检索结果中的任一 Chunk 属于目标 Markdown，该文档就被命中。Top-K 文档列表按首次出现去重。由于冻结 baseline 最终只返回 5 个 Chunk，去重后可能少于 5 个文档；评估脚本不会为此增大候选数。

`no_answer` 问题的 relevant documents 必须为空，并且不计入 Recall@K 和 MRR。它们用于记录冻结检索 baseline 会返回什么，并为轻量 Relevance Gate 提供相似度分布与候选阈值分析。

## Metrics

- `Recall@3`：前 3 个去重文档覆盖的 relevant documents 比例。
- `Recall@5`：前 5 个去重文档覆盖的 relevant documents 比例。
- `MRR`：第一个 relevant document 的倒数排名；未命中为 0。
- `latency_ms`：索引和模型加载完成后，单次 `retrieve()` 的本机墙钟时间。它只用于记录，不是严格性能基准。
- Relevance Gate 分析使用最终 Hybrid Chunk 与 Query 的最大 BGE 余弦相似度，同时输出 Answer/No-answer 分布和候选阈值的 false reject/false accept。

## Run

在项目根目录执行（PowerShell）：

```powershell
$env:HF_HOME = "$PWD/.hf-cache"
$env:HF_HUB_OFFLINE = "1"  # 已缓存 BGE 时可用
.venv-lc1/Scripts/python.exe code/QAR/evaluation/evaluate_retrieval.py
```

脚本输出：

- `results/retrieval_results.jsonl`：每条 Query / Mode 的 Top-K document IDs、source、relevant rank、指标和耗时。
- `results/summary.json`：整体、Query Type、failure category 与无答案行为。
- `results/report.md`：可阅读表格。

## Independent holdout

`holdout_questions.jsonl` 是阈值确定后单独建立的留出集，包含 10 条有答案问题和 12 条无答案问题。它不用于选择或调整阈值；验证时固定使用 `retrieval_relevance_threshold=0.45`。

```powershell
.venv-lc1/Scripts/python.exe code/QAR/evaluation/evaluate_retrieval.py `
  --questions code/QAR/evaluation/holdout_questions.jsonl `
  --output-dir code/QAR/evaluation/holdout_results
```

固定阈值下，10 条有答案问题均未被拒绝；12 条无答案问题中拒绝 9 条、误接受 3 条。结果保存在 `holdout_results/`。

## Limits

主评估集和独立留出集仍是小规模人工诊断数据，不是统计显著的通用排行榜。它们只标注文档级 relevance，不评估命中 Chunk 是否包含完整答案，也不评估生成答案的事实性。公开食谱语料存在表达重复和标题强提示，结果不应外推到企业知识库。
