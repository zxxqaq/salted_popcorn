
30_queries.csv

```csv
id,search_term_en
==== non_food ====
88,General Use Disinfectant Lavender Lysoform

==== specific_name ====
6,Rustic country bread  
13,Crispy rice bowl with protein  
20,Pub-style fish and chips  
26,Açaí bowl with superfoods  
7,Spicy Korean chicken wings  
51,Coastal seafood dish  
52,Fresh farmer’s market salad  
69,Stuffed baked sweet potato  
94,Warm winter soup bowl  
99,Homemade roast meat with sides  
60,Rustic Italian pasta  
54,Comforting slow-cooked stew  
42,Elements of Neapolitan pizza  
30,Vegetarian protein bowl  

==== queries_location_based ====
4,Hawaiian-style lunch  
34,Lebanese street food  
35,Japanese curry bowl  
96,North African spiced rice  
86,French bistro classic  

==== vague_concept ====
100,Comforting late-night snack  
97,Office lunch dish  
92,Night market street food  
82,Movie night snack combo  
78,Post-workout protein meal  
73,Favorite stadium snacks  
71,Homemade family dinner  
66,Quick breakfast with cakes  
39,Classic breakfast  
9,Bento-style lunch box
```

### LLM scoring prompt

```text
You are a relevance scoring expert. Please evaluate the relevance of the following item to the query.

Query: {query}

Item Name: {item_name}
Item Details: {item_text}

Scoring Guidelines:
- 10: Perfect match, item perfectly meets the query requirements
- 8-9: Highly relevant, item very well meets the query requirements
- 6-7: Relevant, item basically meets the query requirements
- 5: Moderately relevant, item partially meets the query requirements
- 3-4: Low relevance, item has weak connection to the query
- 1-2: Almost irrelevant, item has minimal connection to the query
- 0: Completely irrelevant, item has no connection to the query

Please return only a number between 0-10, without any other text or explanation.

```
### LLM score summry 
```
LLM score summry
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

### sample query 
35, Japanese curry bowl - grading results

```text
5.0 - Chickpea Curry

4.0 - 42 pieces grilled salmon hot roll

3.0 - Tropical Poke
3.0 - Kaizen Sushi Premium Festival - For 2 people
3.0 - Toru Combo 50 pieces
3.0 - 1 Temaki + 10 Hot Rolls
3.0 - Shimeji Temaki
3.0 - Hot Kit 1 Person 18 Pieces

2.0 - Combo 2 (50 pieces)
2.0 - Grilled Doritos Temaki
2.0 - Mini Rodizio 1 person (No Changes Made)

1.0 - 1000ml Bowl / açaí, banana, Ninho milk and granola + 3 syrups
1.0 - Big House - 56 pieces
1.0 - Dulce de Leite Mousse with Coconut

0.0 - Super Ice Cream Cake Slice
0.0 - Nachos with Dulce de Leite from 13.90 for only 6.90
0.0 - Coconut Water - 400ml

summary
Score 5.0: 1 item (Chickpea Curry)
Score 4.0: 1 item
Score 3.0: 17 items (mostly Japanese dishes)
Score 2.0: 3 items
Score 1.0: 69 items
Score 0.0: 3 items
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