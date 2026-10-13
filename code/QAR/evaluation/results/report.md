# Frozen Retrieval Baseline Results

No-answer queries are excluded from Recall@K and MRR.

## Overall

| Method | Recall@3 | Recall@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|
| vector | 0.8889 | 0.9630 | 0.8420 | 8.132 |
| bm25 | 1.0000 | 1.0000 | 0.9568 | 15.039 |
| hybrid | 1.0000 | 1.0000 | 0.9815 | 23.313 |

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
| vector | 3 | 1.0000 | 1.0000 | 1.0000 |
| bm25 | 3 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | 3 | 1.0000 | 1.0000 | 1.0000 |

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

## Relevance similarity

The score is the maximum cosine similarity between the query and final Hybrid chunks.

| Group | Count | Min | P25 | Median | Mean | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| answer | 27 | 0.4999 | 0.6806 | 0.7316 | 0.7195 | 0.7680 | 0.8187 |
| no_answer | 12 | 0.3434 | 0.3957 | 0.4203 | 0.4280 | 0.4459 | 0.5165 |

### Threshold candidates

| Threshold | Answer false reject | No-answer false accept |
|---:|---:|---:|
| 0.3000 | 0 | 12 |
| 0.3500 | 0 | 11 |
| 0.4000 | 0 | 8 |
| 0.4500 | 0 | 3 |
| 0.5000 | 1 | 2 |
| 0.5082 | 1 | 1 |
| 0.5500 | 1 | 0 |
| 0.6000 | 2 | 0 |
| 0.6500 | 3 | 0 |
| 0.7000 | 9 | 0 |
| 0.7500 | 16 | 0 |
| 0.8000 | 24 | 0 |

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
- q031 / vector: dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/soup/罗宋汤.md, dishes/meat_dish/香煎五花肉/香煎五花肉.md, dishes/meat_dish/萝卜炖羊排.md
- q031 / bm25: dishes/soup/腊八粥.md, dishes/dessert/反沙芋头/反沙芋头.md, dishes/staple/基础牛奶面包/基础牛奶面包.md
- q031 / hybrid: dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/soup/罗宋汤.md, dishes/meat_dish/酱牛肉/酱牛肉.md, dishes/soup/腊八粥.md, dishes/dessert/反沙芋头/反沙芋头.md
- q032 / vector: dishes/staple/微波炉腊肠煲仔饭/微波炉腊肠煲仔饭.md, dishes/vegetable_dish/素炒豆角.md, dishes/drink/砂糖椰子冰沙/砂糖椰子冰沙.md, dishes/staple/扬州炒饭/扬州炒饭.md, dishes/semi-finished/半成品意面.md
- q032 / bm25: dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/staple/葱油拌面.md, dishes/semi-finished/炸薯条/炸薯条.md, dishes/breakfast/桂圆红枣粥.md, dishes/breakfast/吐司果酱.md
- q032 / hybrid: dishes/vegetable_dish/素炒豆角.md, dishes/staple/微波炉腊肠煲仔饭/微波炉腊肠煲仔饭.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/staple/葱油拌面.md, dishes/drink/砂糖椰子冰沙/砂糖椰子冰沙.md
- q033 / vector: dishes/staple/微波炉腊肠煲仔饭/微波炉腊肠煲仔饭.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/breakfast/空气炸锅面包片.md, dishes/drink/柠檬水/柠檬水.md, dishes/breakfast/燕麦鸡蛋饼.md
- q033 / bm25: dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/咖喱炒蟹.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/soup/金针菇汤.md, dishes/meat_dish/杀猪菜.md
- q033 / hybrid: dishes/vegetable_dish/脆皮豆腐.md, dishes/staple/微波炉腊肠煲仔饭/微波炉腊肠煲仔饭.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/咖喱炒蟹.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md
- q034 / vector: dishes/meat_dish/小米辣炒肉.md, dishes/drink/酸梅汤（半成品加工）.md, dishes/drink/B52轰炸机.md, dishes/staple/汤面.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md
- q034 / bm25: dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/咖喱炒蟹.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/staple/螺蛳粉.md, dishes/staple/中式馅饼/中式馅饼.md
- q034 / hybrid: dishes/meat_dish/小米辣炒肉.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/咖喱炒蟹.md, dishes/drink/酸梅汤（半成品加工）.md, dishes/staple/基础牛奶面包/基础牛奶面包.md
- q035 / vector: dishes/breakfast/煎饺.md, dishes/vegetable_dish/红烧茄子.md, dishes/meat_dish/麻婆豆腐/麻婆豆腐.md, dishes/drink/B52轰炸机.md, dishes/semi-finished/空气炸锅羊排/空气炸锅羊排.md
- q035 / bm25: dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/breakfast/金枪鱼酱三明治.md, dishes/dessert/戚风蛋糕/戚风蛋糕.md, dishes/staple/炸酱面.md
- q035 / hybrid: dishes/meat_dish/麻辣香锅.md, dishes/breakfast/煎饺.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/breakfast/金枪鱼酱三明治.md, dishes/vegetable_dish/红烧茄子.md
- q036 / vector: dishes/drink/菠萝咖啡特调/菠萝咖啡特调.md, dishes/soup/银耳莲子粥/银耳莲子粥.md, dishes/condiment/油泼辣子/油泼辣子.md, dishes/semi-finished/凉皮.md, dishes/drink/酒酿醪糟/酒酿醪糟.md
- q036 / bm25: dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/soup/腊八粥.md, dishes/staple/披萨饼皮/披萨饼皮.md
- q036 / hybrid: dishes/drink/菠萝咖啡特调/菠萝咖啡特调.md, dishes/meat_dish/梅菜扣肉/梅菜扣肉.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/soup/银耳莲子粥/银耳莲子粥.md
- q037 / vector: dishes/staple/汤面.md, dishes/meat_dish/小炒黄牛肉/小炒黄牛肉.md, dishes/staple/热干面.md, dishes/semi-finished/半成品意面.md, dishes/vegetable_dish/西红柿炒鸡蛋.md
- q037 / bm25: dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/咖喱炒蟹.md, dishes/breakfast/桂圆红枣粥.md, dishes/meat_dish/小酥肉.md, dishes/soup/玉米排骨汤/玉米排骨汤.md
- q037 / hybrid: dishes/staple/汤面.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/咖喱炒蟹.md, dishes/meat_dish/小炒黄牛肉/小炒黄牛肉.md, dishes/staple/热干面.md
- q038 / vector: dishes/drink/酒酿醪糟/酒酿醪糟.md, dishes/meat_dish/西红柿土豆炖牛肉/西红柿土豆炖牛肉.md, dishes/aquatic/蒜香黄油虾/蒜香黄油虾.md, dishes/soup/生汆丸子汤.md, dishes/condiment/油泼辣子/油泼辣子.md
- q038 / bm25: dishes/staple/米饭/电饭煲蒸米饭.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/staple/蛋炒饭.md, dishes/dessert/戚风蛋糕/戚风蛋糕.md
- q038 / hybrid: dishes/drink/酒酿醪糟/酒酿醪糟.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/meat_dish/西红柿土豆炖牛肉/西红柿土豆炖牛肉.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/aquatic/蒜香黄油虾/蒜香黄油虾.md
- q039 / vector: dishes/meat_dish/柱候牛腩/柱候牛腩.md, dishes/drink/酸梅汤/酸梅汤.md, dishes/soup/银耳莲子粥/银耳莲子粥.md, dishes/breakfast/金枪鱼酱三明治.md
- q039 / bm25: dishes/meat_dish/香干芹菜炒肉/香干芹菜炒肉.md, dishes/staple/炒河粉.md, dishes/meat_dish/萝卜炖羊排.md, dishes/meat_dish/红烧猪蹄/红烧猪蹄.md, dishes/soup/菌菇炖乳鸽/菌菇炖乳鸽.md
- q039 / hybrid: dishes/meat_dish/香干芹菜炒肉/香干芹菜炒肉.md, dishes/meat_dish/柱候牛腩/柱候牛腩.md, dishes/staple/炒河粉.md, dishes/drink/酸梅汤/酸梅汤.md, dishes/soup/银耳莲子粥/银耳莲子粥.md

## Before/after comparison

The retrieval parameters, tokenizer, chunking, embedding model, and RRF algorithm were unchanged. The only retrieval change was moving metadata filtering into candidate generation.

| Method | Metric | Before | After | Change |
|---|---|---:|---:|---:|
| vector | Recall@3 | 0.8519 | 0.8889 | +0.0370 |
| vector | Recall@5 | 0.9259 | 0.9630 | +0.0370 |
| vector | MRR | 0.8049 | 0.8420 | +0.0370 |
| bm25 | Recall@3 | 0.9630 | 1.0000 | +0.0370 |
| bm25 | Recall@5 | 0.9630 | 1.0000 | +0.0370 |
| bm25 | MRR | 0.9198 | 0.9568 | +0.0370 |
| hybrid | Recall@3 | 0.9630 | 1.0000 | +0.0370 |
| hybrid | Recall@5 | 0.9630 | 1.0000 | +0.0370 |
| hybrid | MRR | 0.9444 | 0.9815 | +0.0370 |

The change is entirely explained by q027. The other 26 answerable queries did not regress.

## q027 metadata filter regression test

- Filter: `source=dishes/semi-finished/空气炸锅羊排/空气炸锅羊排.md`
- Before: vector, BM25, and hybrid all returned no result because each route truncated the global ranking before applying the filter.
- After: all three modes search only chunks satisfying the source filter and rank the relevant document first.
- Hybrid maximum semantic similarity for q027: 0.4999.

## Relevance gate decision

The answer and no-answer score distributions overlap in their tails: the lowest answer score is 0.4999 while the highest no-answer score is 0.5165. A single similarity threshold therefore cannot provide perfect OOD classification.

A conservative threshold of `0.45` is enabled as a lightweight safety gate because it produced zero false rejections on the 27 answerable queries while rejecting 9 of 12 no-answer queries. Three no-answer queries remain false accepts. The threshold is configurable and is not treated as a universal knowledge-boundary classifier.

When the score is below the threshold, the application returns the existing no-evidence response, exposes no sources, and does not call the external LLM.

## Remaining retrieval failures

- q022: Hybrid ranks the relevant document lower than the best single route, so fusion can still reduce rank quality for an individual query.
- q024: Vector misses at the reported cutoff; BM25 succeeds and Hybrid repairs the single-route failure.
- No answerable query fails all three modes after the q027 fix.

Query Rewrite was not included. Moonshot generation was not part of this retrieval experiment.
