# Food Semantic Search
Design and implement a semantic search system that matches natural language (Portuguese)
food queries to the most relevant food items in dataset.

## Architecture Design

![](docs/imgs/arch.png)

## 🧪 Test Set Construction

1. **Query Categorization**  
   The 100 queries are categorized into four groups:  
   - `queries_location_based.csv`  
   - `non_food.csv`  
   - `specific_name.csv`  
   - `vague_concept.csv`  
   as shown in the directory `data/test/query_category`.

2. **Query Selection**  
   From each category, representative queries are selected, a total of **30 queries** are used as the **test set** (`30_queries.csv`).

3. **Candidate Pool Generation**  
   For each query, perform both **vector retrieval** and **text (BM25) retrieval**, each returning **top-50** results.  
   After deduplication, all retrieved items form the **candidate pool**,  
   with a maximum size of `30 × 100 = 3000` pairs.

4. **Ground Truth Scoring**  
   Each `(query, item)` pair is rated for semantic relevance on a **0–10 scale** using **GPT-4.1-mini**.



```
================================================================================
  ✓ Updated 2576 scores

Statistics:
  • Total pairs: 2576
  • Pairs with scores: 2576
  • Pairs without scores: 0
  • Average score: 3.52/10
  • High relevance (≥5): 751 (29.2%)
  • Low relevance (<5): 1825 (70.8%)

================================================================================
Scoring completed!
================================================================================
```

### ❌ Failed Attempts

1. **Random Sampling**  
   Randomly selected 10 queries and 500 items for scoring →  
   The 500 items were too random, resulting in extremely sparse relevance scores, most pairs received 0 and were unusable.

2. **GPT-Generated Relevant Items**  
   Based on (1), used GPT to generate strongly related items for the 10 queries and mixed them into the 500-item pool →  
   The relevance was *too strong* and overly specific, causing the text retriever (BM25) to easily capture them, thus failing to reflect true *semantic* retrieval performance.

3. **Manual Labeling**  
   Tried human labeling →  
   Too much data to handle manually and time-consuming.

## Experiments

Detailed evaluation reports are available in `artifacts/eval_runs`.  
If you want to run the experiments yourself, execute the scripts under `scripts/evaluation`, but make sure to complete the prerequisites first and verify the parameter settings in each script’s `main` function.

### Text Retriever (BM25)

| Average (ms) | Max (ms) | Min (ms) |
|:-------------:|:--------:|:--------:|
| 2.69 | 4.21 | 1.73 |

| Metric | Precision | Recall | NDCG | Coverage |
|:--------|:----------:|:-------:|:------:|:----------:|
| **@K=5**  | 0.9667 | 0.0589 | 0.5920 | 0.0572 |
| **@K=10** | 0.9633 | 0.1177 | 0.6225 | 0.1160 |

### Vector Retriever


| Model name                                                      | Average (ms) | Max (ms) | Min (ms) |
|-----------------------------------------------------------------|:-------------:|:--------:|:--------:|
| **BAAI/bge-m3**                                                 | 46.95 | 89.88 | 33.89 |
| **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** | 109.62 | 510.61 | 7.45 |

| Model name | Metric | Precision | Recall | NDCG | Coverage |
|------------|:--------|:----------:|:-------:|:------:|:----------:|
|      **BAAI/bge-m3**          | **@K=5**  | 0.9800 | 0.0598 | 0.6697 | 0.0580 |
|      **BAAI/bge-m3**          | **@K=10** | 0.9833 | 0.1202 | 0.6963 | 0.1184 |
|       **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**                        | **@K=5**  | 0.4667 | 0.0286 | 0.3351 | 0.0268 |
|       **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**                        | **@K=10** | 0.3867 | 0.0478 | 0.3125 | 0.0461 |

I also tried both text-embedding-3-small and text-embedding-3-large, but embedding through the API takes around 1 second per query, which is too slow. It also depends on the network speed and OpenAI’s service performance, and the token usage is billed, making it quite expensive for large-scale datasets and high concurrency queries.

