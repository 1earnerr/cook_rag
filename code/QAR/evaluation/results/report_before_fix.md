# Frozen Retrieval Baseline Results

No-answer queries are excluded from Recall@K and MRR.

Run corpus and frozen parameters: 322 documents, 1804 chunks,
`BAAI/bge-small-zh-v1.5`, `vector_fetch_k=20`, `bm25_fetch_k=20`,
`final_k=5`, `rrf_k=60`, Query Rewrite disabled.

## Overall

| Method | Recall@3 | Recall@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|
| vector | 0.8519 | 0.9259 | 0.8049 | 11.078 |
| bm25 | 0.9630 | 0.9630 | 0.9198 | 13.347 |
| hybrid | 0.9630 | 0.9630 | 0.9444 | 20.613 |

## By query type

### entity

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 5 | 1.0000 | 1.0000 | 1.0000 |
| bm25 | 5 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 5 | 1.0000 | 1.0000 | 1.0000 |

### fuzzy

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 4 | 0.7500 | 1.0000 | 0.6750 |
| bm25 | 4 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 4 | 1.0000 | 1.0000 | 1.0000 |

### ingredient

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 6 | 0.8333 | 1.0000 | 0.7556 |
| bm25 | 6 | 1.0000 | 1.0000 | 0.8889 |
| hybrid | 6 | 1.0000 | 1.0000 | 1.0000 |

### long_section

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 4 | 0.7500 | 0.7500 | 0.7500 |
| bm25 | 4 | 1.0000 | 1.0000 | 0.8750 |
| hybrid | 4 | 1.0000 | 1.0000 | 0.8750 |

### metadata

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 3 | 0.6667 | 0.6667 | 0.6667 |
| bm25 | 3 | 0.6667 | 0.6667 | 0.6667 |
| hybrid | 3 | 0.6667 | 0.6667 | 0.6667 |

### semantic

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 5 | 1.0000 | 1.0000 | 0.9000 |
| bm25 | 5 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 5 | 1.0000 | 1.0000 | 1.0000 |

## Failure categories

- **hybrid_worse_than_best_single_route**: q022
- **bm25_success_vector_failure**: q024
- **hybrid_repairs_single_route_failure**: q024
- **all_modes_failure**: q027

## No-answer behavior

- q028 / vector: dishes/breakfast/蒸花卷.md, dishes/dessert/魔芋蛋糕/魔芋蛋糕.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/dessert/龟苓膏/龟苓膏.md, dishes/semi-finished/懒人蛋挞/懒人蛋挞.md
- q028 / bm25: dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md, dishes/breakfast/微波炉蛋糕.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/staple/日式肥牛丼饭/日式肥牛丼饭.md, dishes/breakfast/燕麦鸡蛋饼.md
- q028 / hybrid: dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/soup/罗宋汤.md, dishes/dessert/提拉米苏/提拉米苏.md, dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md, dishes/breakfast/蒸花卷.md
- q029 / vector: dishes/breakfast/牛奶燕麦.md, dishes/staple/汤面.md, dishes/drink/耙耙柑茶/耙耙柑茶.md, dishes/staple/扬州炒饭/扬州炒饭.md, dishes/staple/河南蒸面条/河南蒸面条.md
- q029 / bm25: dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/staple/蛋炒饭.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/meat_dish/贵州辣子鸡/贵州辣子鸡.md
- q029 / hybrid: dishes/breakfast/牛奶燕麦.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/staple/汤面.md, dishes/staple/蛋炒饭.md
- q030 / vector: dishes/vegetable_dish/红烧茄子.md, dishes/breakfast/牛奶燕麦.md, dishes/breakfast/温泉蛋/温泉蛋.md, dishes/staple/汤面.md, dishes/meat_dish/小炒黄牛肉/小炒黄牛肉.md
- q030 / bm25: dishes/breakfast/水煮玉米.md, dishes/breakfast/蒸水蛋.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/staple/肉蛋盖饭.md, dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md
- q030 / hybrid: dishes/breakfast/水煮玉米.md, dishes/vegetable_dish/红烧茄子.md, dishes/breakfast/蒸水蛋.md, dishes/breakfast/牛奶燕麦.md, dishes/staple/基础牛奶面包/基础牛奶面包.md

## Manual failure analysis

### Vector succeeds while BM25 fails

None at Recall@5 in this dataset. Even the semantic and fuzzy questions retain
substantial ingredient, cooking-action or dish-description token overlap with the
public corpus. The current Chinese unigram/bigram tokenizer therefore has a strong
advantage on this particular collection. This dataset does not establish that BM25
will dominate on a less lexically aligned enterprise corpus.

### BM25 succeeds while Vector fails

- **q024** asks how many times old-style guobaorou is fried and at what oil
  temperatures. BM25 ranks the annotated document first, while Vector does not put
  it in the final five chunks. Exact terms such as “炸几次” and “油温”, plus the
  numeric procedural wording in the document, favor lexical matching. Vector
  instead returns other fried or braised meat documents.

### Hybrid repairs a single-route failure

- **q024** is repaired by BM25 candidates: Hybrid ranks the correct document first
  although Vector misses it. This is the clearest direct evidence in this small set
  that combining the two routes has value.
- Relative to Vector MRR, Hybrid also improves q009, q011, q014, q018 and q019.
  Relative to BM25, Hybrid improves q006 from document rank 3 to rank 1.

### Hybrid becomes worse

- **q022** is Vector rank 1 but Hybrid rank 2. A chunk from `魔芋蛋糕` is Vector
  rank 2 and BM25 rank 1, giving it RRF score 0.032522; a `戚风蛋糕` appendix
  chunk is Vector rank 1 and BM25 rank 3, giving 0.032266. The relevant
  `戚风蛋糕 / 操作 / 打发蛋白` chunk appears at Hybrid chunk rank 3. RRF therefore
  promotes a lexically similar competing cake document. The relevant document is
  still inside Top-3, so Recall is unchanged while MRR declines.

### All three modes fail

- **q027** uses the generic query “翻面以后还需要加热多久？” with a source filter
  for `空气炸锅羊排`. The target source is absent from all frozen fetch sets even
  before filtering (18 unique Vector documents, 17 BM25 documents and 30 Hybrid
  union documents were inspected). The current implementation applies metadata
  filtering after candidate retrieval, so the filter can remove candidates but
  cannot inject the requested source. This is a candidate-generation/filter-order
  limitation, rather than missing corpus data or a chunking failure.

## Interpretation

- Vector is usable on the current corpus (Recall@5 0.9259), but it ranks several
  ingredient/fuzzy questions lower and misses q024.
- BM25 is unusually strong here (Recall@5 0.9630, MRR 0.9198) because recipe names,
  ingredients and procedural verbs create substantial lexical overlap.
- Hybrid matches BM25 Recall@5 and raises MRR to 0.9444. It improves six questions
  relative to Vector and one relative to BM25, while lowering one question relative
  to the best single route. Its measured benefit is primarily ranking robustness,
  not an increase over BM25 Recall@5.
- All three no-answer questions still return plausible-looking recipe documents.
  The baseline has no relevance threshold, so retrieval alone cannot identify an
  out-of-domain question.
- These 30 manually authored questions do not provide evidence that the embedding
  model must be replaced or that a reranker must be added. The most concrete exposed
  limitation is post-retrieval metadata filtering; no retrieval change is made in
  this phase.

Query Rewrite comparison and real Moonshot HTTP E2E were not executed because no
valid `MOONSHOT_API_KEY` was configured. This does not affect the offline retrieval
metrics above.
