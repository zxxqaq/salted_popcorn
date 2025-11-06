
```text
# Items in ground truth with scores >= this threshold are considered relevant
RELEVANCE_THRESHOLD=5.0 # Used to determine relevance when calculating Precision@K and Recall@K
NDCG_THRESHOLD=1.0
```


Precision@K = (Number of relevant items retrieved) / K

Recall@K = (Number of relevant items retrieved) / (Number of ground truth relevant items in top-K)

NDCG@K = DCG@K / IDCG@K

DCG@K = Σ(score_i / log2(rank_i + 1))

- score_i: The ground truth relevance score at position i  
- rank_i: The rank position (starting from 1)  
- log2(rank_i + 1): Position discount factor  

IDCG@K = DCG@K (for the ideal ranking)

```
Example:  
Assume K = 3, and the ground truth scores are [5.0, 4.0, 3.0]:

- Ideal ranking: DCG = 5.0/log2(2) + 4.0/log2(3) + 3.0/log2(4) = 5.0 + 2.52 + 1.5 = 9.02  
- Retrieved ranking: [3.0, 5.0, 4.0] → DCG = 3.0/log2(2) + 5.0/log2(3) + 4.0/log2(4) = 3.0 + 3.15 + 2.0 = 8.15  
- NDCG = 8.15 / 9.02 ≈ 0.90  
```

MRR = 1 / rank_first_relevant

where rank_first_relevant is the rank position of the first relevant item in the retrieval results.

---

RRF_score(d) = sum(1 / (k + rank_i(d)))

Example (k = 60)
For a given document:

- BM25 ranking: 1st → contribution = 1 / (60 + 1) = 0.0164
- Vector ranking: 3rd → contribution = 1 / (60 + 3) = 0.0159
- RRF score: 0.0164 + 0.0159 = 0.0323