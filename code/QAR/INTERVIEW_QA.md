# QAR 项目面试问答

> 共 52 题。回答时先给结论，再说当前实现和证据，最后主动限定边界。不要逐字背诵；应能根据追问画出数据结构和调用链。

## 一、基础问题

### 1. 这个项目到底解决什么问题？

【面试官问题】你的项目是食谱问答，还是通用知识库？

【推荐回答】最终实现是通用 Markdown 知识库 RAG，公开食谱只是可复现测试语料。核心模型只依赖 `source/title/section/document_id/chunk_id/content`，不依赖 category、difficulty 或菜名；食谱 metadata 默认关闭。系统完成 Markdown 建库、混合检索、证据引用、轻量拒答和 HTTP 服务。

【当前代码证据】`DataPreparationModule` 的通用 metadata；`RAGConfig.recipe_metadata_enabled=False`；`main.py` 没有 list/detail/general 路由；`GenerationIntegrationModule` 使用通用知识库 Prompt。

【可能继续追问】如果把语料换成运维手册，哪些配置和测试需要替换？

### 2. 项目的核心链路是什么？

【面试官问题】请不用框架名，用数据流描述系统。

【推荐回答】离线侧是 Markdown 加载与清洗，按标题切分，超长 Section 按 tokenizer 限制二次切分，生成稳定 ID，用 BGE 建 Native FAISS，同时用当前 Chunk 建 BM25，并保存索引 metadata。在线侧是 Metadata 候选限制，FAISS 和 BM25 各召回 20，RRF 融合取 5，BGE cosine Gate 判断是否拒答，然后构造 S1…Sn Evidence、有限 Context、Prompt，最后调用由环境变量指定的 OpenAI-compatible LLM 并返回 Answer + Sources。

【当前代码证据】`MarkdownRAGSystem.build_knowledge_base()`、`ask_question()`；`RetrievalOptimizationModule.retrieve()`；`GenerationIntegrationModule.generate_basic_answer()`。

【可能继续追问】Gate 为什么放在检索后、Evidence 构造前？

### 3. 你自己实现了什么，哪些是第三方能力？

【面试官问题】这是不是把几个库拼起来？

【推荐回答】模型、FAISS 数学实现和 BM25 数学实现不是我自研的：Embedding 用 BGE，向量索引用 `faiss-cpu`，BM25 用 `rank_bm25`。我实现的是 Markdown/token-aware 数据管线、稳定 ID、索引一致性生命周期、Native FAISS adapter、中文 tokenizer、统一 Vector/BM25/Hybrid 接口、RRF 融合、候选级 Metadata Filter、Evidence/Context 映射、Relevance Gate、服务层、API Schema、Evaluation 和测试。技术价值在于把现成组件做成一致、可追踪、可验证的系统。

【当前代码证据】`rag_modules/` 下各模块；`evaluation/`；`tests/`；`requirements.txt`。

【可能继续追问】Native FAISS adapter 和“自研 FAISS”有什么区别？

### 4. 当前项目规模和验证状态是什么？

【面试官问题】你用多少数据、多少测试，是否跑通过真实模型？

【推荐回答】当前公开语料是 322 篇 Markdown、1804 个 Chunk，FAISS 有 1804 个 512 维向量。最终完整 pytest 是 79 passed；真实 BGE 的 query/document embedding、FAISS 建库、保存加载、Vector/BM25/Hybrid 和 Evidence 冒烟都跑过。另以 DeepSeek `deepseek-chat` OpenAI-compatible API 完成六条真实 FastAPI HTTP 问答，包含有答案、Metadata Filter 和安全拒答；浏览器视觉验收仍需人工完成。这些请求不是答案质量评估。

【当前代码证据】`vector_index/index_metadata.json`；`tools/smoke_real_bge_faiss.py`；测试结果；`evaluation/results/summary.json`。

【可能继续追问】79 个测试中哪些使用真实 BGE，哪些使用 Mock？

## 二、RAG 问题

### 5. 为什么需要 RAG，而不是直接问大模型？

【面试官问题】食谱知识大模型可能本来就知道，RAG 有什么意义？

【推荐回答】项目验证的不是模型会不会做菜，而是能否基于指定私有语料回答、给出可追踪来源并在证据不足时拒答。RAG 把回答依据限制为当前知识库 Evidence；同样的工程可以换成制度、手册等 Markdown。公开食谱只是无隐私且方便人工标注的测试载体。

【当前代码证据】通用 Prompt 明确“只根据证据”；`Evidence` 包含 source 与稳定 ID；无证据返回 `NO_EVIDENCE`。

【可能继续追问】Prompt 约束能否真正保证不幻觉？

### 6. 为什么是 Hybrid Retrieval？

【面试官问题】Vector 和 BM25 留一个不行吗？

【推荐回答】两者处理不同信号。BGE 能处理自然语言语义和改写；BM25 对菜名、原料、数字等精确词项更强。开发集 q024 中 Vector Top-5 漏掉老式锅包肉，BM25 Rank 1 命中，Hybrid 也恢复 Rank 1；但 q022 也说明 Hybrid 不保证每题优于最佳单路，因此我用数据说明互补而不是绝对优势。

【当前代码证据】`RetrievalOptimizationModule.retrieve()`；`evaluation/results/retrieval_results.jsonl` 的 q022、q024。

【可能继续追问】如果 Hybrid 反而变差，你会用什么方法定位？

### 7. Query Rewrite 在系统中是什么地位？

【面试官问题】项目有没有 Query Rewrite，为什么不作为亮点？

【推荐回答】有可配置能力，但默认关闭，Evaluation 主实验和 Holdout 都没有使用。开启时先用 LLM 把问题改成简洁检索 Query；异常、空值或非字符串会回退原 Query，最终生成仍使用原始问题。没有消融证据证明它稳定提升，所以不作为核心技术亮点。

【当前代码证据】`RAGConfig.query_rewrite_enabled=False`；`MarkdownRAGSystem.ask_question()`；`tests/test_system.py::test_optional_rewrite_fallback_and_original_question`。

【可能继续追问】为什么改写用于检索，但生成保留原问题？

### 8. 无证据时系统如何工作？

【面试官问题】空检索和 Gate 拒绝以后会不会还调用 LLM？

【推荐回答】不会。空检索或 Gate 清空 Chunk 后，`_build_context()` 返回空字符串和空 Sources；`generate_basic_answer()` 直接返回固定拒答。流式模式会依次产生 sources、拒答 token、done，也不构造 LLM 链。FastAPI 返回 HTTP 200、固定回答、`sources=[]`、`count=0`。

【当前代码证据】`GenerationIntegrationModule.generate_basic_answer()`、`generate_basic_answer_stream()`；`tests/test_context_builder.py::test_empty_evidence_skips_llm_for_both_output_modes`；`tests/test_api.py::test_no_evidence_is_a_safe_success_without_sources`。

【可能继续追问】为什么无答案用 200 而不是 404？

## 三、Embedding

### 9. 为什么选 BGE-small-zh-v1.5？

【面试官问题】你是怎么选 Embedding 模型的？

【推荐回答】语料和 Query 以中文为主，需要可本地运行、成本适合个人项目并有成熟 Hugging Face 集成的中文向量模型。BGE-small 在 CPU 环境可以完成真实建库，输出 512 维，模型规模与 1804 Chunk 匹配。我没有做多模型对照，所以只说它满足当前约束，不声称它是最优模型。

【当前代码证据】`RAGConfig.embedding_model`；`IndexConstructionModule.setup_embeddings()`；真实 smoke 输出 512 维。

【可能继续追问】如果企业语料中英文混合，你会如何重新验证模型选择？

### 10. Embedding 为什么要归一化？

【面试官问题】`normalize_embeddings=True` 带来什么？

【推荐回答】归一化后每个向量范数为 1，`IndexFlatL2` 的平方距离满足 `2-2cosine`，所以 L2 排序与 cosine 排序一致。这样 Dense Retrieval 的距离顺序和 Gate 的 cosine 相关性使用同一表示空间。Gate 仍显式除以范数计算 cosine，不把归一化假设写死到公式里。

【当前代码证据】`IndexConstructionModule.setup_embeddings()`；`NativeFAISSIndex.semantic_similarities()`。

【可能继续追问】如果向量未归一化，L2 和 cosine 会出现什么差异？

### 11. 模型最大 512 tokens，为什么项目用 450？

【面试官问题】450 是实验调出来的吗？

【推荐回答】不是效果调优结果，而是低于 512 的安全上限，为特殊 token 和不同文本形态保留余量。真正检查使用模型 tokenizer，不再用“1200 字符以内就安全”的错误假设。450 只保证尽量避免截断，不代表检索效果最优。

【当前代码证据】`RAGConfig.embedding_chunk_max_tokens=450`；`DataPreparationModule._token_count()`、`_split_section()`；token budget 测试。

【可能继续追问】为什么不直接设成 510？

### 12. 为什么没有更换 Qwen3 Embedding？

【面试官问题】新模型可能更强，为什么冻结旧模型？

【推荐回答】项目目标是建立可复现基线，不是持续追最新模型。BGE 已完成真实环境验证、开发 Evaluation 和独立 Holdout；换模型需要重新建库、重新评估和重新校准 Gate。当前没有对照实验支持“换了会更好”，所以继续换只会破坏因果解释和冻结结果。

【当前代码证据】Evaluation 参数固定 `BAAI/bge-small-zh-v1.5`；用户约束与 `config.py`；没有 Qwen 依赖或代码。

【可能继续追问】如果允许做模型对照，如何保证公平？

## 四、FAISS

### 13. 为什么使用 `IndexFlatL2`？

【面试官问题】为什么没有使用 IVF 或 HNSW？

【推荐回答】当前只有 1804 个向量，精确搜索成本很低。`IndexFlatL2` 不需要训练、没有近似召回损失，结果确定且易测试。IVF/HNSW 的价值主要在更大规模和延迟压力下，当前没有压测证据要求引入近似索引和更多参数。

【当前代码证据】`NativeFAISSIndex.from_documents()` 创建 `faiss.IndexFlatL2(vectors.shape[1])`；metadata 中 `chunk_count=1804`。

【可能继续追问】数据到百万级后如何选择索引？

### 14. Native FAISS 封装具体做了什么？

【面试官问题】你去掉 LangChain FAISS wrapper 后补了哪些能力？

【推荐回答】封装负责批量 Embedding、创建 IndexFlatL2、校验向量形状、维护 position→chunk_id→Document 映射、复制返回 Document 防止 rank metadata 污染、查询、过滤候选的精确 L2、Gate cosine、持久化 payload 和加载一致性校验。它是业务 adapter，不是重写 FAISS 算法。

【当前代码证据】`rag_modules/native_faiss.py` 的 `NativeFAISSIndex` 全部方法。

【可能继续追问】为什么检索结果要复制 Document？

### 15. FAISS position 和 Chunk 如何保证不串？

【面试官问题】索引 position 本身没有文档信息，如何映射？

【推荐回答】建库时按 Chunk 顺序 `enumerate(chunk_ids)` 建 `index_to_docstore_id`，再用 `documents_by_id` 保存复制后的 Document。加载时 `_validate()` 检查 position 集合必须等于 `0..ntotal-1`、chunk_id 不重复、两个映射的 ID 集合一致。索引 metadata 还校验 index 文件 hash 与 Chunk hash。

【当前代码证据】`NativeFAISSIndex.from_documents()`、`_validate()`；`index_metadata.py`；`tests/test_index_lifecycle.py::assert_consistent`。

【可能继续追问】如果 `index.pkl` 被旧版本覆盖会发生什么？

### 16. 为什么保存索引时先删除 metadata marker？

【面试官问题】索引保存一半进程崩溃怎么办？

【推荐回答】`index_metadata.json` 被当成成功提交标记。保存前先删除它，再写 `index.faiss` 和 `index.pkl`，最后计算文件 hash 并原子替换 metadata。若中途失败，下次 metadata 缺失或 hash 不匹配，会整体重建，不会把半成品当成有效索引。当前不是完整事务系统，但避免了最直接的损坏复用。

【当前代码证据】`IndexConstructionModule.save_index()`；`save_index_metadata()` 使用 `.tmp` 后 replace；`test_failed_save_invalidates_marker_and_next_build_recovers`。

【可能继续追问】并发请求正在读旧索引时同步重建会有什么风险？

## 五、BM25

### 17. BM25 的核心思想是什么？

【面试官问题】请结合项目解释 BM25，而不是背定义。

【推荐回答】BM25 用 Query token 在每个 Chunk 中的出现频率和稀有程度打分，并对词频做饱和、对文档长度做归一化。项目把 1804 个 Chunk 先经过同一个 tokenizer 建 `BM25Okapi`，查询也经过同一 tokenizer，然后按 score 排名。数学实现来自 `rank_bm25`，项目没有自研公式。

【当前代码证据】`NativeBM25Retriever.__init__()` 和 `search()`；`rank_bm25.BM25Okapi` direct dependency。

【可能继续追问】TF 饱和和长度归一化分别解决什么问题？

### 18. 当前中文 tokenizer 如何处理文本？

【面试官问题】“宫保鸡丁 restore-123”会切成什么？

【推荐回答】正则先提取连续中文片段或 ASCII/数字/下划线片段。中文“宫保鸡丁”生成单字“宫、保、鸡、丁”和相邻双字“宫保、保鸡、鸡丁”；英文转小写保留 `restore`，连字符不是允许字符，所以数字形成 `123`。文档与 Query 使用完全相同处理。

【当前代码证据】`RetrievalOptimizationModule.tokenize()`；`test_chinese_tokenizer_has_unigrams_bigrams_and_ascii`。

【可能继续追问】为什么包含“保鸡”这种非词 token？

### 19. 这个 tokenizer 的缺陷是什么？

【面试官问题】为什么不用 jieba 或语言模型 tokenizer？

【推荐回答】当前方法确定、无新增依赖、领域外也能运行，但它不是词级分词：没有词典、停用词、同义词、实体识别，会产生“保鸡”这类噪声，长文本 token 数也会增加。项目第一目标是建立稳定基线；是否换 tokenizer 应通过同一 Evaluation 对照，而不是凭感觉。

【当前代码证据】tokenize 实现只有正则、单字和双字；requirements 没有 jieba。

【可能继续追问】如何设计 tokenizer 的消融实验？

### 20. Metadata Filter 下 BM25 是怎样检索的？

【面试官问题】过滤后是否重新训练 BM25？

【推荐回答】不会重新构建 `BM25Okapi`，IDF 仍来自全语料。查询先对全语料得到 score，但排序候选位置只包含 metadata 匹配的 Chunk，因此 Top-K 一定从过滤集合产生。这满足“在指定集合中检索”的语义，同时保持实现简单；它不等同于为每个过滤子集重新计算 IDF。

【当前代码证据】`NativeBM25Retriever.search()` 中 `scores` 与 `candidate_positions`；`RetrievalOptimizationModule._matching_chunk_ids()`。

【可能继续追问】子集 IDF 与全局 IDF 可能造成什么排名差异？

## 六、RRF

### 21. RRF 为什么适合当前两路检索？

【面试官问题】RRF 相比 score normalization 有什么优势？

【推荐回答】FAISS 返回 L2 距离，BM25 返回无上界相关性分数，两者量纲、方向和分布不同。RRF 只依赖各自排名，不需要在小数据集上学习归一化权重，行为稳定、容易解释。代价是丢失同一路内部的绝对分差，Rank 1 比 Rank 2 强多少无法体现。

【当前代码证据】`RetrievalOptimizationModule._rrf_rerank()` 只使用 `rank`，没有读取 FAISS distance 或 BM25 score。

【可能继续追问】什么情况下会考虑加权 RRF？

### 22. 当前 RRF 公式如何落到代码？

【面试官问题】同一个 Chunk 被两路命中时怎么算？

【推荐回答】以稳定 `chunk_id` 为键，Vector Rank `r_v` 贡献 `1/(60+r_v)`，BM25 Rank `r_b` 贡献 `1/(60+r_b)`，两项相加。只被一路命中就只有一项。最终按分数降序；分数相同用 `chunk_id` 做确定性排序，同时保存两路 rank 和 `rrf_score`。

【当前代码证据】`_rrf_rerank()` 中 `scores[chunk_id] += 1.0/(rrf_k+rank)` 和 `ordered=sorted(...)`。

【可能继续追问】为什么去重键不能使用正文文本？

### 23. `rrf_k=60` 是否调优过？

【面试官问题】为什么是 60，不是 10 或 100？

【推荐回答】没有通过参数搜索调优。60 是当前冻结的平滑常数，使前几名贡献差距不过分尖锐；项目第一轮 Evaluation 刻意不调 K 或 RRF，避免为了小数据集追漂亮数字。因此只能说它是配置默认值，不能说是最优参数。

【当前代码证据】`RAGConfig.rrf_k=60`；Evaluation summary 记录该参数；README 明确未专门调优。

【可能继续追问】如果允许调参，如何避免在 27 条问题上过拟合？

### 24. Hybrid 为什么可能比 Vector 单路更差？

【面试官问题】q022 发生了什么？

【推荐回答】q022 中 Vector 把戚风蛋糕排第一，BM25 把词面相近的魔芋蛋糕排第一、目标排第二。RRF 同时吸收两路排名后，魔芋蛋糕得到更强综合位置，目标降到第二。Hybrid 仍命中，所以 Recall 不变，但 MRR 从 Vector 的 1 变为 0.5。这说明融合优化的是整体稳健性，不保证逐题超过 oracle 最佳单路。

【当前代码证据】`evaluation/results/retrieval_results.jsonl` 的 q022；`failure_categories.hybrid_worse_than_best_single_route`。

【可能继续追问】如果不加 Reranker，还能怎样分析这个问题？

## 七、Chunking

### 25. 为什么不能只用 Markdown Header Splitter？

【面试官问题】既然 Markdown 已经有章节，为什么还二次切？

【推荐回答】标题只提供结构边界，不保证一个 Section 的长度。有的操作章节可能几千字，直接送给 512-token Embedding 会被截断，后半段永远无法参与向量检索。因此 Header Splitter 负责语义结构，token-aware 递归切分负责模型输入安全。

【当前代码证据】`_markdown_header_split()` 后调用 `_split_section()`；`embedding_chunk_max_tokens=450`。

【可能继续追问】为什么不是先固定 token 切分再补标题？

### 26. 二次切分的停止条件是什么？

【面试官问题】RecursiveCharacterTextSplitter 本身还是按字符，怎么保证 token 上限？

【推荐回答】第一次递归字符切分只是生成自然边界候选；代码随后对每个 piece 使用真实 Embedding tokenizer 计数。只要仍有 piece 超过 450，就缩小字符窗口继续切，直到全部满足预算；如果无法产生更小文本就抛错，不会静默接受超限 Chunk。

【当前代码证据】`DataPreparationModule._split_section()` 的 while 循环和 progress guard；中英文混合 token 测试。

【可能继续追问】tokenizer 加载失败时当前 fallback 有什么风险？

### 27. Stable Chunk ID 是怎样生成的？

【面试官问题】增加一篇无关文档会不会改变原 Chunk ID？

【推荐回答】不会。文档 ID 是相对路径 SHA-256，不依赖全局遍历序号；Chunk ID 是文档 ID、`section_index:subchunk_index` 和 Chunk 内容的 SHA-256。增加排序更靠前的无关文档只会改变 `batch_index`，不会改变已有文档身份。原文内容变化时 document_id 保持，但受影响 chunk_id 会变化。

【当前代码证据】`_stable_chunk_id()`；`test_ids_stable_after_reload_and_unrelated_document_added`、`test_content_change_changes_chunk_not_document_id`。

【可能继续追问】重命名文档为什么会改变 document_id？这是优点还是限制？

### 28. overlap 在这里有什么具体作用和限制？

【面试官问题】`section_chunk_overlap=100` 是 100 tokens 吗？

【推荐回答】是 100 字符，不是 tokens。它让二次切分后的相邻 Chunk 共享边界文本，降低答案跨边界被拆散的概率。代码最后仍用 tokenizer 检查每块不超过 450 tokens；在进一步缩小窗口时 overlap 会限制为窗口五分之一以内。测试证明至少有相邻重叠，但没有证明 100 是最佳值。

【当前代码证据】`RAGConfig.section_chunk_overlap=100`；`RecursiveCharacterTextSplitter` 配置；`test_token_aware_recursive_split_keeps_configured_overlap`。

【可能继续追问】overlap 太大会带来什么检索和 Context 问题？

## 八、Evaluation

### 29. Evaluation 数据集是怎样标注的？

【面试官问题】27 条答案问题的 ground truth 是什么？

【推荐回答】人工阅读公开 Markdown 后，为每条 Query 标注一个或多个 relevant source 相对路径，运行时映射成稳定 document_id。覆盖实体、原料、自然语义、模糊表达、长章节和 Metadata。标注粒度是文档级：目标文档任一 Chunk 出现在 Top-K 就算命中；没有标注答案所在精确 Chunk。

【当前代码证据】`evaluation/questions.jsonl`；`evaluate_retrieval.py` 的 `source_to_document_id`、`unique_documents()`、`query_metrics()`。

【可能继续追问】文档级标注会高估哪些能力？

### 30. Recall@3、Recall@5、MRR 分别说明什么？

【面试官问题】为什么三个指标都要保留？

【推荐回答】Recall@3 看前三个去重文档覆盖多少相关文档；Recall@5 看扩大窗口后的召回；MRR 关注第一个相关文档位置，Rank 1 为 1、Rank 2 为 0.5。两个 Recall 能区分“没找到”和“排得稍后”，MRR 能区分同样命中但排序好坏。q022 三种模式 Recall 都命中，但 Hybrid MRR 比 Vector 低。

【当前代码证据】`query_metrics()`；q022 结果。

【可能继续追问】如果每题有多个 relevant documents，当前 Recall 怎么算？

### 31. 为什么 No-answer 不进入 Recall/MRR？

【面试官问题】无答案问题也有 Top-K，为什么不算 Recall=0？

【推荐回答】无答案问题没有 relevant document，普通 Recall 的分母为零，强行记 0 会把“检索不到目标”和“本来就没有目标”混为一谈。项目把它们单独用于观察错误候选和相似度分布，再计算 Gate 的 reject/false accept。

【当前代码证据】`query_metrics()` 只对有答案行调用；`metrics=None if query_type == "no_answer"`；README。

【可能继续追问】无答案检索还可以用哪些指标，但本项目为什么没用？

### 32. 如何解释最终三组结果？

【面试官问题】BM25 Recall 已经 1，Hybrid 还有什么价值？

【推荐回答】在 27 条小数据上，Vector 是 0.8889/0.9630/0.8420，BM25 是 1/1/0.9568，Hybrid 是 1/1/0.9815。Hybrid 没提高 BM25 的 Recall，但提高了平均首个相关文档排名；同时 q024 展示了 BM25 对 Dense 的补偿，q022展示融合也会单题退化。只能说在当前数据上 Hybrid 的整体 MRR 最好，不能推广到企业知识库。

【当前代码证据】`evaluation/results/summary.json` 和 `report.md`。

【可能继续追问】为什么这个数据集可能天然偏向 BM25？

## 九、Relevance Gate

### 33. 为什么用最终 Top Evidence 的最大相似度？

【面试官问题】为什么不是平均值或最小值？

【推荐回答】回答只需要至少一个强相关证据，所以最大值是简单保守的放行规则；平均值会被 Final-5 中辅助或噪声 Chunk 拉低，增加答案误拒。项目也记录 Top-3 分数供分析，但没有构建复杂分类器。最大值的缺陷是一个偶然高相似的错误 Chunk 就能让 OOD Query 通过。

【当前代码证据】`MarkdownRAGSystem.ask_question()` 使用 `max(similarities)`；Evaluation 保存 `top3_semantic_similarities`。

【可能继续追问】h014 为什么能利用这个缺陷通过？

### 34. 0.45 是如何确定和验证的？

【面试官问题】阈值是不是为了让结果好看？

【推荐回答】开发集先记录 27 个答案和 12 个无答案的 Hybrid 最大 cosine。两类尾部重叠，0.45 在开发集上答案误拒 0、无答案误接受 3；它优先保护答案。阈值冻结后，新建 10 Answer + 12 No-answer Holdout，不再调参，结果仍是答案误拒 0、无答案拒绝 9、误接受 3。因此它有独立支持，但远非完美。

【当前代码证据】`evaluation/results/summary.json`；`holdout_results/summary.json`；`RAGConfig.retrieval_relevance_threshold=0.45`。

【可能继续追问】为什么 Holdout 结果出来后不能改成 0.55？

### 35. Gate 为什么不能叫 OOD Detector？

【面试官问题】你已经能拒绝 75% 无答案，为什么不叫 OOD 检测？

【推荐回答】它没有训练 OOD 分类器，也没有概率校准或多领域大样本验证；开发集尾部已经重叠，Holdout 仍有 25% false accept。它测的是 Query 与返回 Chunk 的语义接近度，而不是判断 Query 是否属于知识库支持域，更不能验证证据包含答案。因此准确名称是轻量相关性拒答机制。

【当前代码证据】h014/h018/h020 false accepts；h007 高相似但 Hybrid 漏目标；Holdout report。

【可能继续追问】如果未来要做可靠拒答，需要怎样的标注和评估？

### 36. Gate 如何保证低相关问题不调用 LLM？

【面试官问题】只是把 Sources 隐藏了吗？

【推荐回答】不是隐藏。`ask_question()` 在生成模块之前把低相关 `chunks` 设为空；`_build_context()` 因此产生空 Context；`generate_basic_answer()` 在调用 `_chain()` 前直接返回固定拒答，所以 `setup_llm()` 都不会执行。同步和流式路径都有测试记录调用列表为空。

【当前代码证据】`main.py` Gate 分支；`generation_integration.py` 空 Context 分支；`test_relevance_gate_rejects_low_similarity_without_calling_llm`。

【可能继续追问】Query Rewrite 开启时，低相关问题是否可能先调用一次 LLM？

## 十、FastAPI

### 37. 为什么要有 `RAGService`？

【面试官问题】路由直接调用 `MarkdownRAGSystem` 不行吗？

【推荐回答】Service 隔离了 Core 生命周期和 HTTP 协议：负责初始化、ready 检查、调用 Core、识别外部 LLM 异常、把内部 Evidence 转换成稳定 API Schema。路由只做输入输出和状态码映射，所以测试能注入 FakeSystem，不需要加载真实 BGE 或外部 LLM。

【当前代码证据】`services/rag_service.py`；`api.create_app(service=...)`；`tests/test_api.py::FakeSystem`。

【可能继续追问】Service 是否线程安全？当前代码做了哪些保证？

### 38. `/health` 的数字为什么不是硬编码？

【面试官问题】如何判断服务是真就绪还是进程活着？

【推荐回答】`health()` 从当前 `index_module.embeddings`、`vectorstore`、`retrieval_module` 和 `data_module.documents/chunks` 读取状态。Embedding 和索引都存在才是 `ok`，否则 `degraded` 并返回初始化错误摘要。322/1804 只是当前语料运行值，不写死在 Schema 或路由。

【当前代码证据】`RAGService.health()`；正常和 degraded API 测试。

【可能继续追问】生产环境还应增加哪些 readiness/liveness 检查？

### 39. API 异常如何映射？

【面试官问题】为什么客户端看不到 traceback？

【推荐回答】Pydantic 参数错误由 FastAPI 返回 422；Core 未初始化映射 503；OpenAI-compatible LLM 调用失败或缺少 `LLM_*` 配置映射 502；普通检索/生成异常和 reindex 失败映射 500。Service 对外部 LLM 错误只记录异常类型，对内部错误保留服务端堆栈；API handler 均返回固定公开消息，不向客户端暴露私有 URL、路径或 traceback。

【当前代码证据】`RAGServiceError` 子类、`_is_external_llm_error()`；`api.py` exception handlers；API 异常测试。

【可能继续追问】为什么 Retrieval 异常是 500 而不是 503？

### 40. `/reindex` 为什么是同步接口？

【面试官问题】重建 1804 个向量会阻塞，为什么没做 Celery？

【推荐回答】当前语料只有 322 文档，任务目标是完整个人项目，不是生产平台，所以允许同步复用统一重建逻辑并返回 elapsed time。没有业务量、并发或任务可靠性证据支持 Redis/Celery。限制是重建期间请求等待，且没有后台建新索引后原子切换，面试中应主动承认。

【当前代码证据】`RAGService.reindex()` 调 `build_knowledge_base(force=True)`；`api.py` 普通同步 POST；没有队列依赖。

【可能继续追问】如果未来需要不停服重建，最小架构会怎样变化？

## 十一、系统设计

### 41. 如何保证 FAISS、BM25 和原文版本一致？

【面试官问题】修改一个 Markdown 后系统如何知道要重建？

【推荐回答】每次建库只读取一次 Markdown，当前内存 Chunk 同时作为预期 metadata、FAISS 和 BM25 的输入。metadata 包含原文 source+content 的 corpus hash、Chunk 文本+metadata 的 chunks hash、Embedding 模型、分块版本、数量、recipe 开关和两个索引文件 hash。任一不匹配就不复用 FAISS；BM25始终从本次 Chunk 重建。单独 `add_documents` 被明确禁止。

【当前代码证据】`MarkdownRAGSystem.build_knowledge_base()`；`index_metadata.py`；`IndexConstructionModule.add_documents()`；生命周期测试。

【可能继续追问】为什么 BM25 没有单独持久化？

### 42. 文档修改、增加、重命名和删除分别会怎样？

【面试官问题】生命周期是否覆盖所有语料变化？

【推荐回答】四种变化都会改变 corpus hash，从而触发完整重建。修改同一路径保持 document_id，但内容相关 chunk_id 改变；增加和删除改变文档/Chunk 数量及 hash；重命名改变 relative source，所以 document_id 也改变。测试逐项执行这些动作并检查 Dense/Sparse/当前 Chunk 一致。

【当前代码证据】`test_real_faiss_bm25_reuse_and_corpus_changes`；Stable ID 测试。

【可能继续追问】如果想让文档重命名后 ID 不变，需要引入什么外部身份？

### 43. Windows 中文路径为什么需要特殊处理？

【面试官问题】FAISS 保存路径包含中文时做了什么？

【推荐回答】直接调用 FAISS 原生文件 API 在部分 Windows 非 ASCII 路径上不可靠。项目用 `faiss.serialize_index()` 先在内存得到字节，再用 Python `Path.write_bytes()` 保存；加载时 `read_bytes()` 加 `faiss.deserialize_index()`。Document 映射单独 pickle，限定为可信本地项目目录。

【当前代码证据】`IndexConstructionModule._save_index_files()`、`_load_index_files()`；`test_faiss_save_and_load_supports_unicode_windows_path`。

【可能继续追问】为什么加载 pickle 需要信任边界？

### 44. 当前系统在并发方面有什么真实能力？

【面试官问题】FastAPI 是否就意味着高并发？

【推荐回答】不能这样说。API 路由是同步函数，Embedding、FAISS、BM25 和 LLM 调用也按当前实现同步；没有压测、限流、连接池策略、索引读写锁、多进程共享或异步队列。FastAPI 只说明有 HTTP 服务边界，不代表已经验证高并发。同步 reindex 与查询同时发生的行为也没有生产级保证。

【当前代码证据】`api.py` 的普通 `def` 路由；`RAGService.reindex()`；requirements 无 Redis/Celery；没有并发测试。

【可能继续追问】如果面试官要求 100 QPS，你首先测什么？

## 十二、Failure Case

### 45. q027 为什么是实现 Bug，而不是模型能力问题？

【面试官问题】目标文档没召回，为什么不换 Embedding？

【推荐回答】用户已经通过 source filter 明确限定目标文档，正确语义应是在该集合内排序。旧实现先全局 Top-K 再过滤，目标连候选都没机会进入；这与 Embedding 好坏无关，是查询执行顺序错误。改成候选生成后，Vector、BM25、Hybrid 都 Rank 1，且另外 26 个答案问题无 regression。

【当前代码证据】`report_before_fix.md` 与最终 `report.md`；`_matching_chunk_ids()`；三模式参数化测试。

【可能继续追问】当前 filtered FAISS 为什么没有直接调用 `index.search`？

### 46. q024 说明了什么？

【面试官问题】Vector 为什么会漏掉带菜名的问题？

【推荐回答】q024 问“老式锅包肉需要炸几次，每次油温”，Vector Top-5 被小酥肉、红烧肉等烹饪语义相近文档占据，目标未命中；BM25 利用“老式锅包肉、炸、油温”等词面信号 Rank 1 命中，Hybrid 也 Rank 1。它说明 Dense 不是所有实体/步骤查询都可靠，稀疏路线有实际补偿价值。当前实验没有进一步判断是 Chunk、Embedding 还是候选数量造成 Dense 漏检。

【当前代码证据】q024 三模式 `retrieval_results.jsonl`；failure category `bm25_success_vector_failure`。

【可能继续追问】如何区分是 Embedding 问题还是 Chunking 问题？

### 47. h007 为什么是 Gate 的关键反例？

【面试官问题】答案 Query 相似度很高，不是说明 Gate 成功了吗？

【推荐回答】h007 的正确文档是韩式拌饭。BM25 找到它，但 Hybrid Final-5 没有；Final-5 里其他酱汁文档让最大 cosine 达到 0.7347，所以 Gate 放行。按 false reject 指标它算通过，但实际上 Evidence 不含标注目标。这证明 Gate 只检测“返回内容与问题语义近不近”，不能保证“返回内容含答案”，也说明文档级 Retrieval 与拒答评估要分别解释。

【当前代码证据】`holdout_results/retrieval_results.jsonl` 的 h007；Holdout report 的 failure category。

【可能继续追问】为什么不把 h007 记成 Answer false reject？

### 48. 三个 Holdout false accept 有什么共同启示？

【面试官问题】钢筋混凝土、MOSFET、珊瑚白化为什么会过 Gate？

【推荐回答】Embedding 会捕捉一般过程词和表面语义：混凝土“养护方式”接近食谱操作说明，MOSFET 的功率/损耗与加热描述存在弱关联，珊瑚白化的海水/温度与水产语料接近。最大值规则只要一个 Chunk 偶然高相似就放行。它们说明单一 cosine 阈值不是知识边界分类器。

【当前代码证据】h014=0.5116、h018=0.4734、h020=0.4956；固定阈值 0.45。

【可能继续追问】为什么不根据这三个失败把阈值升到 0.55？

## 十三、项目取舍

### 49. 为什么没有加入 Reranker？

【面试官问题】h007 和 q022 看起来都可以用 Reranker 修复，为什么不加？

【推荐回答】有可能，但当前任务强调最小、可解释闭环。加入 Cross-Encoder 会增加模型、延迟、依赖和新的评估变量；现有小数据不足以证明收益和成本。项目选择如实保留失败案例，而不是每看到一个错误就堆新模型。不能把“可能修复”说成“已经验证”。

【当前代码证据】requirements 无 reranker；Retrieval 只含 Vector/BM25/RRF；Evaluation 报告保留 q022/h007。

【可能继续追问】如果允许增加 Reranker，应插在哪一层、如何做消融？

### 50. 为什么不用 Milvus、Elasticsearch 或 Redis？

【面试官问题】企业项目不是应该有这些组件吗？

【推荐回答】技术选择应由规模和需求驱动。当前单机 1804 个向量，IndexFlatL2 精确检索足够；BM25 用本地 rank_bm25；没有分布式数据、在线增量、高并发缓存或多节点检索要求。引入这些服务会增加部署面，却不能证明检索更准。项目更重视索引一致性、可测试和真实 Evaluation。

【当前代码证据】metadata 的 1804 Chunk；requirements 和架构中没有这些服务；同步完整重建。

【可能继续追问】达到什么规模或需求时你会重新评估？

### 51. 为什么没有数据库、用户系统和多租户？

【面试官问题】这是否说明工程不完整？

【推荐回答】项目边界是单知识库 RAG Core + 最小 HTTP 服务，目标用户状态、权限和租户隔离并不存在。为了简历加入空壳数据库或用户表不会提高 RAG 正确性，反而扩大无法验证的功能。它作为个人项目是完整的，但不能称企业级平台；如果业务真的要求多租户，需要重新设计索引隔离、授权、审计和资源配额。

【当前代码证据】三个 API Schema 没有 user/session/tenant；没有数据库依赖；`MarkdownRAGSystem` 明确 one in-process corpus。

【可能继续追问】多租户下 metadata filter 能否替代权限隔离？

### 52. 为什么现在选择冻结项目？

【面试官问题】还有失败案例，为什么不继续优化？

【推荐回答】工程项目不以“零失败”为停止条件，而以目标和证据闭环为准。当前已经有通用数据模型、稳定索引生命周期、Hybrid、Evidence、拒答、API、79 项测试、开发 Evaluation 和独立 Holdout；已知限制也被量化。继续换模型、加 Reranker 或调参会开启新的实验周期，并可能对小数据过拟合。对秋招个人项目，更重要的是能解释设计、失败和取舍，而不是堆无限技术栈。

【当前代码证据】`PROJECT_GUIDE.md` 的结果与限制；最终 Evaluation/Holdout；冻结配置和测试结果。

【可能继续追问】如果进入企业后只能优先改一项，你会如何基于业务数据决定？

## 使用建议

1. 先熟练讲第 1、2、3、6、29、32、34、45、47、52 题，它们覆盖项目主线。
2. 对每个数字都说明样本范围，不把 Recall/MRR 说成答案准确率。
3. 被追问未实现能力时，先回答“当前没有实现”，再说明为什么当时没有必要。
4. 能在白板上写出三个式子：BM25 词项思想、`RRF=Σ1/(k+rank)`、单位向量 `L2²=2-2cosine`。
5. 能手写三种核心数据结构：Chunk `Document.metadata`、`Evidence`、`ChatResponse`。
