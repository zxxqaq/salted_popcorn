# 使用 BAAI/bge-m3 作为向量召回模型

## 安装依赖

首先，确保安装了 `FlagEmbedding` 库：

```bash
pip install FlagEmbedding
```

## 配置

在 `.env` 文件中设置：

```bash
# 使用 bge-m3 作为本地向量召回模型
VECTOR_LOCAL_MODEL_NAME=BAAI/bge-m3

# 可选：如果查询和文档使用不同的模型，可以单独设置查询模型
# VECTOR_QUERY_EMBEDDING_MODEL=BAAI/bge-m3
```

## 使用方式

### 1. 生成向量索引

使用 `scripts/generate_vector_index.py` 生成 HNSW 索引：

```bash
# 在 .env 中设置：
# VECTOR_LOCAL_MODEL_NAME=BAAI/bge-m3
# VECTOR_INDEX_PATH=data/vector_indices  # 可选，指定索引保存路径

python scripts/generate_vector_index.py
```

### 2. 在 Hybrid Retrieval 中使用

在 `hybrid_retrieval_no_llm.py` 或相关脚本中，`bge-m3` 会自动被识别并使用 `BGEM3FlagModel` 进行编码。

### 3. 模型特性

- **维度**: `bge-m3` 的密集向量（dense vector）维度为 **1024**
- **多向量支持**: `bge-m3` 支持密集向量、稀疏向量和多重向量，当前实现仅使用密集向量
- **归一化**: 默认启用归一化（`normalize_embeddings=True`），使用余弦相似度

## 注意事项

1. **维度匹配**: 确保查询和文档使用相同的模型，否则会出现维度不匹配错误
2. **首次加载**: `bge-m3` 模型较大，首次加载可能需要一些时间
3. **内存使用**: `bge-m3` 模型需要较多内存，建议至少 8GB RAM

## 示例

```python
from src.vector_retrieval import VectorRetriever
from src.bm25_retrieval import load_food_candidates

# 加载候选文档
candidates, _ = load_food_candidates("data/test/500_items.csv")

# 创建 VectorRetriever，使用 bge-m3
retriever = VectorRetriever(
    candidates=candidates,
    local_model_name="BAAI/bge-m3",  # 使用 bge-m3
    normalize_embeddings=True,
    use_hnsw=True,
    index_path="data/vector_indices/hnsw_bge_m3.index",
)

# 搜索
results = retriever.search("vegano", top_k=10)
for candidate, score in results:
    print(f"{candidate.name}: {score:.4f}")
```

