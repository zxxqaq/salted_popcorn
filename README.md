
## Intro
A semantic search system that matches natural language food queries to the most relevant food items in our dataset.

## Quick-start 🚀

### Prerequisites

- **Docker** and **Docker Compose** installed

### Docker Deployment 

   ```bash
   git clone <repository-url>
   # and then cd into the dir
   
   cp .env.example .env
   # Edit .env file with your configuration
   
   docker-compose build
   # Build Docker image (installs dependencies and downloads models)
   ```

**Generate cache files (no need, already in data folder)**

Put here to specify in case you want to regenerate cache files. Make sure check .env parameters and parameters in main before generation.
   
   ```bash
   # Option 1: Run each script individually
   docker-compose run --rm semantic-search python scripts/generate/generate_bm25_cache.py
   docker-compose run --rm semantic-search python scripts/generate/generate_vector_index.py
   docker-compose run --rm semantic-search python scripts/generate/generate_reranker_tokenization_cache.py
   
   # Option 2: Enter container and run scripts interactively
   docker-compose run --rm semantic-search bash
   # Inside container:
   python scripts/generate/generate_bm25_cache.py
   python scripts/generate/generate_vector_index.py
   python scripts/generate/generate_reranker_tokenization_cache.py
   exit
   
   # you can also check scripts/evaluation to run evaluation test
   ```
**Run demo**
   
   ```bash
   # Option 1: One-liner (recommended)
   docker-compose run --rm semantic-search python scripts/demo_query_selector.py 5
   
   # Option 2: Interactive mode
   docker-compose run --rm semantic-search bash
   # Inside container:
   python scripts/demo_query_selector.py 5
   # Or run without arguments for interactive query selection:
   python scripts/demo_query_selector.py
   exit
   ```


## Demo

```bash
$ python scripts/demo_query_selector.py [query_index]
# query_index: Row number (1-100) in given query dataset data/raw/queries.csv
```

```
================================================================================
Performing Hybrid Search...
================================================================================
Query: Sanduíche de café da manhã com abacate
Query ID: 5
  Query embedding (local model) completed in 0.350s
  HNSW search completed in 0.007s

================================================================================
RESULTS
================================================================================

Top-10 Results:
--------------------------------------------------------------------------------
   1. [148c2ee9-ddc7-42db-b342-639cacdaafc4] Queijo Quente Bacon (score: -3.4102)
   2. [56d13cdc-f76a-4412-a0b4-af09839b2689] Pão francês com presunto e queijo (score: -3.7090)
   3. [3725415a-7fd2-4981-a28c-e07c91def2c6] Sanduíche natural (score: -3.7812)
   4. [252373d9-89b4-485a-8dda-d790216ded09] Promoção da casa (score: -4.2969)
   5. [86a220ca-a5b4-4df6-bd02-6aed29bd7981] Cesta de Mccafé da Manhã (score: -4.3945)
   6. [5a039127-9b48-4166-935a-ab3e555234aa] Sanduíche de Churrasco de Carne Bovina (score: -4.6211)
   7. [9033455a-4ac7-4a23-a399-de0e049164fe] Egg Cheese Bacon - Promo Mcd (score: -4.6484)
   8. [d44737ed-9026-426e-9a07-7a6ee1948358] Peru & Queijo (score: -4.8242)
   9. [e90a5039-94b9-4ce4-8d89-bb362b67432c] Misto quente Com Ovo (score: -4.8555)
  10. [853172c5-8035-4195-a851-a2b3671a2a1d] Misto Quente (score: -5.1016)

Score Range: -5.1016 - -3.4102

================================================================================
TIMING BREAKDOWN
================================================================================
  BM25 Retrieval:        16.30 ms
  Vector Retrieval:      357.38 ms
  RRF Fusion:            0.30 ms
  Cross-Encoder Re-rank: 1130.29 ms
--------------------------------------------------------------------------------
  TOTAL LATENCY:         1494.82 ms (1.495 s)
================================================================================

```

## Architecture Design

![](docs/imgs/arch.png)

## Test Set Construction

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

## Components and Experiments

Detailed evaluation reports are available in `artifacts/eval_runs`.  
If you want to run the experiments yourself, execute the scripts under `scripts/evaluation`, but make sure to complete the prerequisites first and verify the parameter settings in each script’s `main` function.

### Experiment Environment

```
Hardware Configuration:
  Platform: macOS-15.6.1-arm64-arm-64bit-Mach-O
  Processor: arm
  Machine: arm64
  Architecture: 64bit
  CPU Count: 8
  CPU Frequency: 3504.00 MHz
  Total Memory: 16.00 GB
  GPU: Apple Metal (MPS)
  GPU Memory: N/A
  Python Version: 3.13.2
```


### Text Retriever (BM25)

| Average (ms) | Max (ms) | Min (ms) |
|:------------:|:--------:|:--------:|
|     2.75     |   4.44   |   1.79   |

| Metric | Precision | Recall | NDCG |
|:--|--:|--:|--:|
| **@K = 5** | 0.3933 | 0.2111 | 0.5920 |
| **@K = 10** | 0.3900 | 0.2850 | 0.6225 |

### Vector Retriever


| Model name                                                      | Average (ms) | Max (ms) | Min (ms) |
|-----------------------------------------------------------------|:------------:|:--------:|:--------:|
| **BAAI/bge-m3**                                                 |    59.25     |  284.47   |  33.55  |
| **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** |    22.67     |  149.13  |   8.42   |

| Model name | Metric | Precision | Recall | NDCG | 
|------------|:--------|:---------:|:------:|:------:|
|      **BAAI/bge-m3**          | **@K=5**  |  0.5067   | 0.2244 | 0.6697 | 
|      **BAAI/bge-m3**          | **@K=10** |  0.4700   | 0.3112 | 0.6963 | 
|       **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**                        | **@K=5**  |  0.2267   | 0.1517 | 0.3351 |
|       **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**                        | **@K=10** |  0.1933   | 0.1774 | 0.3125 | 

I also tried both text-embedding-3-small and text-embedding-3-large, but embedding through the API takes around 1 second per query, which is too slow. It also depends on the network speed and OpenAI’s service performance, and the token usage is billed, making it quite expensive for large-scale datasets and high concurrency queries.

### Hybrid Retriever

Concurrent text retriever + vector retriever (top-50 + top-50) → RRF fusion (top-20) → filtering → reranker → top-10



| Metric | Average (ms) | Max (ms) | Min (ms) |
|:--|--:|--:|--:|
| **Overall Latency** | 1851.93 | 4071.48 | 1124.13 |


| Stage | Avg (ms) | Max (ms) | Min (ms) | Notes |
|:--|--:|--:|--:|:--|
| **BM25 Retrieval** | 16.80 | 162.80 | 7.32 | Text-based retrieval |
| **Vector Retrieval** | 82.39 | 357.26 | 42.31 | HNSW + Embedding Search |
| **RRF Fusion / Merge** | 0.16 | 0.27 | 0.09 | Reciprocal Rank Fusion |
| **Cross-Encoder Re-ranking** | 1767.78 | 4013.13 | 761.19 | Re-ranking with BGE Reranker |


| Metric | Precision | Recall | NDCG |
|:--|--:|--:|--:|
| **@K = 5** | 0.5733 | 0.2494 | 0.7141 |
| **@K = 10** | 0.5033 | 0.3726 | 0.7321 |

### Evaluation and Reflection


| Metric | My Result | Typical Range (Ref: MS MARCO / BEIR / MTEB) | Evaluation |
|:--|:---------:|:--:|:----------:|
| **NDCG@10** |   0.73    | 0.60–0.80 |    Good    |
| **Precision@10** |   0.50    | 0.40–0.60 |    Good    |
| **Recall@10** |   0.37    | 0.30–0.50 |  Moderate  |
| **Latency** |   1.85s   | <1s Ideal |   Slow   |

| Reference                         | Link                                                                                                 |
|-----------------------------------|------------------------------------------------------------------------------------------------------|
| MS MARCO Leaderboard              | [https://microsoft.github.io/msmarco/leaderboard/](https://microsoft.github.io/msmarco/leaderboard/) |
| BEIR Leaderboard                  | [https://github.com/beir-cellar/beir#leaderboard](https://github.com/beir-cellar/beir#leaderboard)   |
| HuggingFace Retrieval Leaderboard | [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)     |


The overall latency (≈1.85s) is relatively high compared to the ideal target (<1s).
Several reasons were discovered during implementation:
1. **Limited GPU parallelism on macOS (MPS)**  
   Apple’s Metal Performance Shader (MPS) backend currently does not support true parallel inference or concurrent batching. 
   As a result, multiple batches of Cross-Encoder re-ranking are executed sequentially rather than concurrently, leading to longer total inference time.
2. **Lack of GPU acceleration (no CUDA available)**  
   The experiments were run on a **MacBook Air (M2, 16GB)**.  
   Without CUDA or discrete GPU, computation relies mainly on CPU and limited MPS acceleration, resulting in higher latency.



