 Week 4 Theory Note: Attention Fundamentals 

The Core Equation
Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + M ) V

- Q (Query): What I am looking for.
- K (Key): What I offer.
- V (Value): What I actually contain.
- sqrt(d_k): Scaling factor. Since Q and K have dimension d_k, their dot product has variance d_k. Dividing by sqrt(d_k) keeps the variance at 1, preventing the softmax from becoming too "peaky" (which would kill gradients).
- M (Mask): Used to hide future tokens (causal mask) or padding.

The Three Types of Attention
1. Encoder Self-Attention:Q, K, V all come from the same input. No mask. Every token sees every other token.

2. Decoder Self-Attention (Causal): Q, K, V come from the decoder. Masked so position i can only see positions <= i.

3. Cross-Attention: Q comes from the decoder, K and V come from the encoder. Allows the decoder to look at the source text.

Why Attention Beats BiLSTM
- BiLSTM: Sequential (slow), fixed-size context (loses long-range info), O(T) complexity.
- Attention: Parallel (fast), full context (no distance decay), O(T^2) complexity.