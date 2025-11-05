# 信息检索评估指标详解

本文档解释 `evaluate.py` 中使用的各种评估指标的含义和计算方法。

## 基本概念

在开始之前，需要理解几个概念：

- **检索结果 (Retrieved)**: 检索系统返回的文档列表（按相关性排序）
- **相关文档 (Relevant)**: 根据 ground truth（LLM 打分）判定为相关的文档
- **Top-K**: 只考虑前 K 个检索结果进行评估

---

## 1. Precision@K（精确率@K）

### 含义
在前 K 个检索结果中，有多少比例是真正相关的。

### 计算公式
```
Precision@K = (前 K 个结果中的相关文档数) / K
```

### 代码实现
```python
hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
return hits / k
```

### 例子
- Query: "not vegan"
- Top-10 检索结果: [A, B, C, D, E, F, G, H, I, J]
- 相关文档（根据 LLM 打分 >= 阈值）: [A, C, E, G]
- Precision@10 = 4/10 = 0.4（前 10 个中有 4 个相关）
- Precision@5 = 2/5 = 0.4（前 5 个中有 2 个相关，假设是 A, C）

### 解释
- **范围**: 0.0 到 1.0（越高越好）
- **意义**: 衡量检索系统的准确性，避免返回过多不相关的结果

---

## 2. Recall@K（召回率@K）

### 含义
在所有相关文档中，有多少被检索到了前 K 个结果中。

### 计算公式
```
Recall@K = (前 K 个结果中的相关文档数) / (所有相关文档数)
```

### 代码实现
```python
hits = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_set)
return hits / len(relevant_set)
```

### 例子
- Query: "not vegan"
- 所有相关文档: [A, B, C, D, E]（共 5 个）
- Top-10 检索结果: [A, C, E, F, G, H, I, J, K, L]
- Recall@10 = 3/5 = 0.6（5 个相关文档中检索到了 3 个）

### 解释
- **范围**: 0.0 到 1.0（越高越好）
- **意义**: 衡量检索系统的完整性，避免遗漏相关文档

---

## 3. NDCG@K（归一化折扣累积增益@K）

### 含义
考虑排序质量的指标，既考虑相关文档是否被检索到，也考虑它们的位置（排名越靠前越好）。

### 计算公式

#### DCG@K (Discounted Cumulative Gain)
```
DCG@K = Σ(i=1 to k) (2^relevance_i - 1) / log2(i + 1)
```
- `relevance_i`: 第 i 个位置的文档的相关度分数（0-10）
- 分母是位置折扣因子（排名越靠后，权重越小）

#### NDCG@K (Normalized DCG)
```
NDCG@K = DCG@K / IDCG@K
```
- IDCG@K: Ideal DCG，即完美排序下的 DCG（所有相关文档按分数从高到低排列）

### 代码实现
```python
# 计算实际 DCG
gains = [relevances.get(doc_id, 0.0) for doc_id in retrieved_ids]
actual_dcg = dcg_at_k(gains, k)

# 计算理想 DCG
ideal = sorted(relevances.values(), reverse=True)
ideal_dcg = dcg_at_k(ideal, k)

# NDCG = 实际 DCG / 理想 DCG
return actual_dcg / ideal_dcg
```

### 例子
假设：
- Ground truth 相关文档及分数: {A: 8.0, B: 7.0, C: 6.0, D: 5.0}
- Top-10 检索结果: [C, E, A, F, B, G, H, I, J, D]

计算：
```
实际排序的 relevance: [6.0, 0.0, 8.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 5.0]
DCG@10 = (2^6-1)/log2(2) + (2^8-1)/log2(4) + (2^7-1)/log2(6) + (2^5-1)/log2(11)
        = 63/1 + 255/2 + 127/2.58 + 31/3.46
        ≈ 63 + 127.5 + 49.2 + 9.0 = 248.7

理想排序的 relevance: [8.0, 7.0, 6.0, 5.0]
IDCG@10 = (2^8-1)/log2(2) + (2^7-1)/log2(3) + (2^6-1)/log2(4) + (2^5-1)/log2(5)
         = 255/1 + 127/1.58 + 63/2 + 31/2.32
         ≈ 255 + 80.4 + 31.5 + 13.4 = 380.3

NDCG@10 = 248.7 / 380.3 ≈ 0.654
```

### 解释
- **范围**: 0.0 到 1.0（越高越好）
- **意义**: 同时考虑相关性和排序位置，是信息检索中最常用的指标之一

---

## 4. MAP（平均精确率均值）

### 含义
所有查询的平均精确率（Average Precision）的平均值。每个查询的平均精确率考虑了所有相关文档的位置。

### 计算公式

#### Average Precision (单个查询)
```
AP = Σ(k=1 to n) (P@k × rel_k) / (相关文档总数)
```
- `P@k`: 前 k 个结果的精确率
- `rel_k`: 第 k 个位置是否为相关文档（1 或 0）

#### Mean Average Precision
```
MAP = Σ(AP_i) / (查询总数)
```

### 代码实现
```python
hits = 0
precision_sum = 0.0
for idx, doc_id in enumerate(retrieved_ids, start=1):
    if doc_id in relevant_set:
        hits += 1
        precision_sum += hits / idx  # 累积精确率
return precision_sum / len(relevant_set)
```

### 例子
- Query: "not vegan"
- 所有相关文档: [A, B, C]（3 个）
- 检索结果: [A, X, B, Y, C, Z]

计算：
```
位置 1: A 相关 → P@1 = 1/1 = 1.0
位置 2: X 不相关 → 跳过
位置 3: B 相关 → P@3 = 2/3 = 0.667
位置 4: Y 不相关 → 跳过
位置 5: C 相关 → P@5 = 3/5 = 0.6

AP = (1.0 + 0.667 + 0.6) / 3 = 0.756
```

### 解释
- **范围**: 0.0 到 1.0（越高越好）
- **意义**: 综合衡量精确率和排序质量，常用作整体性能指标

---

## 5. MRR（平均倒数排名）

### 含义
所有查询的第一个相关文档出现位置的倒数，再求平均。

### 计算公式
```
MRR = Σ(1 / rank_i) / (查询总数)
```
- `rank_i`: 第 i 个查询的第一个相关文档的排名位置

### 代码实现
```python
for idx, doc_id in enumerate(retrieved_ids, start=1):
    if doc_id in relevant_set:
        return 1.0 / idx  # 返回第一个相关文档的倒数排名
return 0.0  # 没有找到相关文档
```

### 例子
- Query 1: 第一个相关文档在位置 3 → RR = 1/3 = 0.333
- Query 2: 第一个相关文档在位置 1 → RR = 1/1 = 1.0
- Query 3: 没有相关文档 → RR = 0.0

MRR = (0.333 + 1.0 + 0.0) / 3 = 0.444

### 解释
- **范围**: 0.0 到 1.0（越高越好）
- **意义**: 特别关注第一个相关结果的位置，适合重视"第一个结果"质量的场景

---

## 6. Coverage@K（覆盖率@K）

### 含义
在 Top-K 结果中包含至少一个相关文档的查询占总查询的比例。

### 计算公式
```
Coverage@K = (至少在前 K 个结果中有一个相关文档的查询数) / (总查询数)
```

### 代码实现
```python
if any(doc_id in qr.relevance for doc_id in doc_ids[:k]):
    coverage_counts[k] += 1
# 最后计算
coverage = coverage_counts[k] / len(retrievals)
```

### 例子
- 10 个查询
- 其中 8 个查询的 Top-10 结果中至少包含一个相关文档
- Coverage@10 = 8/10 = 0.8

### 解释
- **范围**: 0.0 到 1.0（越高越好）
- **意义**: 衡量检索系统的覆盖能力，确保每个查询都有结果

---

## 指标选择建议

根据不同场景选择合适的指标：

| 场景 | 推荐指标 |
|------|---------|
| **整体性能评估** | MAP, NDCG@K |
| **关注排序质量** | NDCG@K |
| **关注精确率** | Precision@K |
| **关注召回率** | Recall@K |
| **关注第一个结果** | MRR |
| **系统覆盖度** | Coverage@K |

---

## 实际使用示例

在你的评估结果中，你会看到类似这样的输出：

```json
{
  "metrics": {
    "precision@5": 0.45,
    "precision@10": 0.38,
    "recall@5": 0.32,
    "recall@10": 0.48,
    "ndcg@5": 0.56,
    "ndcg@10": 0.62,
    "average_precision": 0.54,
    "reciprocal_rank": 0.71
  }
}
```

**解读**：
- **Precision@10 = 0.38**: 前 10 个结果中 38% 是相关的
- **Recall@10 = 0.48**: 检索到了 48% 的相关文档
- **NDCG@10 = 0.62**: 排序质量是理想排序的 62%
- **MAP = 0.54**: 平均精确率均值
- **MRR = 0.71**: 平均第一个相关文档在位置 1.4 左右出现

---

## 注意事项

1. **min_relevance 阈值**: 只有 score >= min_relevance 的文档才被认为是"相关"的
2. **空结果**: 如果某个查询没有相关文档，部分指标会为 0
3. **K 值选择**: 通常使用 K=5, 10, 20 等常见值，根据实际应用场景选择

