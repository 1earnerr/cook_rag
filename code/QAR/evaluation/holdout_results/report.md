# Frozen Retrieval Baseline Results

No-answer queries are excluded from Recall@K and MRR.

## Overall

| Method | Recall@3 | Recall@5 | MRR | Mean latency (ms) |
|---|---:|---:|---:|---:|
| vector | 0.9000 | 0.9000 | 0.8500 | 9.015 |
| bm25 | 1.0000 | 1.0000 | 0.8833 | 21.560 |
| hybrid | 0.9000 | 0.9000 | 0.9000 | 29.963 |

## By query type

### answer

| Method | Queries | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| vector | 10 | 0.9000 | 0.9000 | 0.8500 |
| bm25 | 10 | 1.0000 | 1.0000 | 0.8833 |
| hybrid | 10 | 0.9000 | 0.9000 | 0.9000 |

## Failure categories

- **bm25_success_vector_failure**: h007
- **hybrid_worse_than_best_single_route**: h007

## Relevance similarity

The score is the maximum cosine similarity between the query and final Hybrid chunks.

| Group | Count | Min | P25 | Median | Mean | P75 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| answer | 10 | 0.6693 | 0.7348 | 0.7578 | 0.7614 | 0.7985 | 0.8322 |
| no_answer | 12 | 0.3744 | 0.4158 | 0.4209 | 0.4339 | 0.4500 | 0.5116 |

### Threshold candidates

| Threshold | Answer false reject | No-answer false accept |
|---:|---:|---:|
| 0.3000 | 0 | 12 |
| 0.3500 | 0 | 12 |
| 0.4000 | 0 | 10 |
| 0.4500 | 0 | 3 |
| 0.5000 | 0 | 1 |
| 0.5500 | 0 | 0 |
| 0.5905 | 0 | 0 |
| 0.6000 | 0 | 0 |
| 0.6500 | 0 | 0 |
| 0.7000 | 1 | 0 |
| 0.7500 | 4 | 0 |
| 0.8000 | 7 | 0 |

## No-answer behavior

- h011 / vector: dishes/semi-finished/空气炸锅羊排/空气炸锅羊排.md, dishes/semi-finished/空气炸锅鸡翅中/空气炸锅鸡翅中.md, dishes/meat_dish/糖醋里脊.md, dishes/vegetable_dish/蒲烧茄子.md
- h011 / bm25: dishes/semi-finished/空气炸锅羊排/空气炸锅羊排.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md, dishes/meat_dish/牛排/牛排.md, dishes/semi-finished/空气炸锅鸡翅中/空气炸锅鸡翅中.md
- h011 / hybrid: dishes/semi-finished/空气炸锅羊排/空气炸锅羊排.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/semi-finished/空气炸锅鸡翅中/空气炸锅鸡翅中.md, dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md
- h012 / vector: dishes/meat_dish/无骨鸡爪/无骨鸡爪.md, dishes/soup/羊肉汤/羊肉汤.md, dishes/vegetable_dish/红烧茄子.md, dishes/meat_dish/柱候牛腩/柱候牛腩.md, dishes/meat_dish/香煎五花肉/香煎五花肉.md
- h012 / bm25: dishes/aquatic/水煮鱼.md, dishes/breakfast/茶叶蛋.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/dessert/戚风蛋糕/戚风蛋糕.md
- h012 / hybrid: dishes/meat_dish/无骨鸡爪/无骨鸡爪.md, dishes/aquatic/水煮鱼.md, dishes/soup/羊肉汤/羊肉汤.md, dishes/breakfast/茶叶蛋.md, dishes/staple/基础牛奶面包/基础牛奶面包.md
- h013 / vector: dishes/vegetable_dish/糖拌西红柿/糖拌西红柿.md, dishes/breakfast/煎饺.md, dishes/staple/汤面.md, dishes/breakfast/溏心蛋.md, dishes/dessert/芋泥雪媚娘/芋泥雪媚娘.md
- h013 / bm25: dishes/semi-finished/炸薯条/炸薯条.md, dishes/staple/肉蛋盖饭.md, dishes/breakfast/蒸水蛋.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md
- h013 / hybrid: dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md, dishes/semi-finished/炸薯条/炸薯条.md, dishes/vegetable_dish/糖拌西红柿/糖拌西红柿.md, dishes/breakfast/煎饺.md, dishes/staple/肉蛋盖饭.md
- h014 / vector: dishes/dessert/草莓冰淇淋/草莓冰淇淋.md, dishes/soup/羊肉汤/羊肉汤.md, dishes/meat_dish/瘦肉土豆片/瘦肉土豆片.md, dishes/meat_dish/老式锅包肉/老式锅包肉.md, dishes/meat_dish/血浆鸭/血浆鸭.md
- h014 / bm25: dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/staple/蛋炒饭.md, dishes/breakfast/燕麦鸡蛋饼.md, dishes/dessert/戚风蛋糕/戚风蛋糕.md, dishes/meat_dish/萝卜炖羊排.md
- h014 / hybrid: dishes/dessert/戚风蛋糕/戚风蛋糕.md, dishes/dessert/草莓冰淇淋/草莓冰淇淋.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/staple/蛋炒饭.md, dishes/soup/羊肉汤/羊肉汤.md
- h015 / vector: dishes/meat_dish/巴基斯坦牛肉咖喱/巴基斯坦牛肉咖喱.md, dishes/aquatic/咖喱炒蟹.md, dishes/staple/日式咖喱饭/日式咖喱饭.md, dishes/drink/酸梅汤/酸梅汤.md, dishes/staple/微波炉腊肠煲仔饭/微波炉腊肠煲仔饭.md
- h015 / bm25: dishes/dessert/玛格丽特饼干/玛格丽特饼干.md, dishes/staple/蛋炒饭.md, dishes/drink/冰粉/冰粉.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md
- h015 / hybrid: dishes/staple/螺蛳粉.md, dishes/meat_dish/巴基斯坦牛肉咖喱/巴基斯坦牛肉咖喱.md, dishes/dessert/玛格丽特饼干/玛格丽特饼干.md, dishes/aquatic/咖喱炒蟹.md, dishes/staple/蛋炒饭.md
- h016 / vector: dishes/meat_dish/麻婆豆腐/麻婆豆腐.md, dishes/vegetable_dish/红烧茄子.md, dishes/staple/汤面.md, dishes/soup/皮蛋瘦肉粥.md, dishes/meat_dish/干煸仔鸡/干煸仔鸡.md
- h016 / bm25: dishes/meat_dish/麻辣香锅.md, dishes/semi-finished/速冻汤圆/速冻汤圆.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/meat_dish/干煸仔鸡/干煸仔鸡.md, dishes/drink/砂糖椰子冰沙/砂糖椰子冰沙.md
- h016 / hybrid: dishes/meat_dish/麻辣香锅.md, dishes/meat_dish/干煸仔鸡/干煸仔鸡.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/meat_dish/麻婆豆腐/麻婆豆腐.md, dishes/semi-finished/速冻汤圆/速冻汤圆.md
- h017 / vector: dishes/dessert/炸鲜奶/炸鲜奶.md, dishes/drink/菠萝咖啡特调/菠萝咖啡特调.md, dishes/vegetable_dish/红烧茄子.md, dishes/breakfast/空气炸锅面包片.md
- h017 / bm25: dishes/dessert/戚风蛋糕/戚风蛋糕.md, dishes/dessert/魔芋蛋糕/魔芋蛋糕.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md
- h017 / hybrid: dishes/dessert/戚风蛋糕/戚风蛋糕.md, dishes/dessert/魔芋蛋糕/魔芋蛋糕.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/dessert/炸鲜奶/炸鲜奶.md
- h018 / vector: dishes/condiment/简易版炒糖色.md, dishes/breakfast/牛奶燕麦.md, dishes/meat_dish/麻辣香锅.md, dishes/breakfast/温泉蛋/温泉蛋.md, dishes/meat_dish/香煎五花肉/香煎五花肉.md
- h018 / bm25: dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md, dishes/staple/米饭/电饭煲蒸米饭.md, dishes/soup/米粥.md, dishes/staple/豆角焖面/豆角焖面.md, dishes/breakfast/蒸水蛋.md
- h018 / hybrid: dishes/soup/米粥.md, dishes/meat_dish/麻辣香锅.md, dishes/condiment/简易版炒糖色.md, dishes/vegetable_dish/鸡蛋羹/鸡蛋羹.md, dishes/staple/米饭/电饭煲蒸米饭.md
- h019 / vector: dishes/meat_dish/洋葱炒猪肉.md, dishes/staple/火腿饭团/火腿饭团.md, dishes/drink/杨枝甘露.md, dishes/meat_dish/土豆炖排骨/土豆炖排骨.md, dishes/staple/酸辣蕨根粉.md
- h019 / bm25: dishes/meat_dish/番茄红酱.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/aquatic/小龙虾/小龙虾.md, dishes/soup/腊八粥.md, dishes/vegetable_dish/手撕包菜/手撕包菜.md
- h019 / hybrid: dishes/aquatic/咖喱炒蟹.md, dishes/drink/杨枝甘露.md, dishes/aquatic/糖醋鲤鱼/糖醋鲤鱼.md, dishes/meat_dish/番茄红酱.md, dishes/meat_dish/洋葱炒猪肉.md
- h020 / vector: dishes/aquatic/微波葱姜黑鳕鱼.md, dishes/drink/冰粉/冰粉.md, dishes/condiment/蔗糖糖浆/蔗糖糖浆.md, dishes/staple/醪糟小汤圆.md, dishes/meat_dish/萝卜炖羊排.md
- h020 / bm25: dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md, dishes/vegetable_dish/小炒藕丁/小炒藕丁.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/staple/西红柿鸡蛋挂面/西红柿鸡蛋挂面.md, dishes/meat_dish/意式烤鸡.md
- h020 / hybrid: dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/condiment/简易版炒糖色.md, dishes/drink/酒酿醪糟/酒酿醪糟.md, dishes/aquatic/清蒸鲈鱼/清蒸鲈鱼.md, dishes/aquatic/微波葱姜黑鳕鱼.md
- h021 / vector: dishes/meat_dish/小炒黄牛肉/小炒黄牛肉.md, dishes/vegetable_dish/红烧茄子.md, dishes/dessert/魔芋蛋糕/魔芋蛋糕.md, dishes/breakfast/蒸花卷.md, dishes/breakfast/温泉蛋/温泉蛋.md
- h021 / bm25: dishes/aquatic/白灼虾/白灼虾.md, dishes/vegetable_dish/鸡蛋羹/微波炉鸡蛋羹.md, dishes/aquatic/小龙虾/小龙虾.md, dishes/aquatic/葱油桂鱼/葱油桂鱼.md, dishes/vegetable_dish/凉拌黄瓜.md
- h021 / hybrid: dishes/breakfast/煎饺.md, dishes/aquatic/白灼虾/白灼虾.md, dishes/meat_dish/小炒黄牛肉/小炒黄牛肉.md, dishes/vegetable_dish/红烧茄子.md, dishes/vegetable_dish/鸡蛋羹/微波炉鸡蛋羹.md
- h022 / vector: dishes/dessert/魔芋蛋糕/魔芋蛋糕.md, dishes/staple/河南蒸面条/河南蒸面条.md, dishes/staple/基础牛奶面包/基础牛奶面包.md, dishes/vegetable_dish/虎皮青椒/虎皮青椒.md, dishes/vegetable_dish/糖拌西红柿/糖拌西红柿.md
- h022 / bm25: dishes/meat_dish/老式锅包肉/老式锅包肉.md, dishes/breakfast/吐司果酱.md, dishes/drink/B52轰炸机.md, dishes/meat_dish/荷兰豆炒腊肠/荷兰豆炒腊肠.md, dishes/breakfast/牛奶燕麦.md
- h022 / hybrid: dishes/dessert/魔芋蛋糕/魔芋蛋糕.md, dishes/meat_dish/老式锅包肉/老式锅包肉.md, dishes/staple/河南蒸面条/河南蒸面条.md, dishes/breakfast/吐司果酱.md, dishes/staple/基础牛奶面包/基础牛奶面包.md

## Fixed 0.45 holdout result

The threshold was fixed before this holdout was executed and was not adjusted from these results.

- Answer false reject: 0/10 (0%)
- No-answer reject: 9/12 (75%)
- No-answer false accept: 3/12 (25%)
- False accepts: h014 (0.5116), h018 (0.4734), h020 (0.4956)

All answer queries exceeded the threshold, but h007 exposes a separate retrieval limitation: BM25 retrieves the annotated Korean mixed-rice document while Hybrid does not include it in the final five chunks. Its high similarity score therefore prevents rejection without proving that the returned evidence contains the answer.
