 Week 4 Theory Note: Attention Fundamentals 

The Core Equation
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) V

- Q (Query): What I am looking for.
- K (Key): What I offer.
- V (Value): What I actually contain.
- sqrt(d_k): Scaling factor. Since Q and K have dimension d_k, their dot product has variance d_k. Dividing by sqrt(d_k) keeps the variance at 1, preventing the softmax from becoming too "peaky" (which would kill gradients).
- M (Mask): Used to hide future tokens (causal mask) or padding.

 Worked Numeric Example : 

Input:
X = [[2, 1],
     [1, 3]]
W_Q = W_K = W_V = identity, d_k = 2

Step 1 — Projections:
Q = K = V = X (since weights are identity)

Step 2 — Score matrix QK^T:
[[2, 1], [1, 3]] · [[2, 1], [1, 3]]^T
= [[2·2 + 1·1, 2·1 + 1·3],
   [1·2 + 3·1, 1·1 + 3·3]]
= [[5, 5],
   [5, 10]]

Step 3 — Scale by √d_k = √2 ≈ 1.414:
[[3.536, 3.536],
 [3.536, 7.071]]

Step 4 — Row-wise softmax:

Row 1: exp(3.536) = 34.33, exp(3.536) = 34.33
       sum = 68.66 → [0.500, 0.500]

Row 2: exp(3.536) = 34.33, exp(7.071) = 1176.9
       sum = 1211.2 → [0.028, 0.972]

Attention weights A:
[[0.500, 0.500],
 [0.028, 0.972]]

Step 5 — Multiply by V:
A · V = [[0.5·2 + 0.5·1, 0.5·1 + 0.5·3],
         [0.028·2 + 0.972·1, 0.028·1 + 0.972·3]]
= [[1.500, 2.000],
   [1.028, 2.944]]

Interpretation:
The second token attends 97% to itself because
its key (1, 3) is much more similar to its own query
(1, 3) than the first token's key (2, 1) is.


The Three Types of Attention :
1. Encoder Self-Attention:Q, K, V all come from the same input. No mask. Every token sees every other token.

2. Decoder Self-Attention (Causal): Q, K, V come from the decoder. Masked so position i can only see positions <= i.

3. Cross-Attention: Q comes from the decoder, K and V come from the encoder. Allows the decoder to look at the source text.

Why Attention Beats BiLSTM
- BiLSTM: Sequential (slow), fixed-size context (loses long-range info), O(T) complexity.
- Attention: Parallel (fast), full context (no distance decay), O(T^2) complexity.