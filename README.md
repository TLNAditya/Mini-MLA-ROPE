# Mini-MLA-ROPE

[Paper](https://arxiv.org/pdf/2405.04434)

![alt text](image.png)

# This is Deep dive into Multi Head Latent Attention(MLA) mechanism + RoPE.

[Detail Explaination of it - Medium](https://medium.com/@myselftlnaditya/detailed-explanation-of-mla-rope-along-with-code-8137dd7141e1)

## Motivation

I was fascinated by how DeepSeek-V2 redesigned attention.

Instead of just using standard Multi-Head Attention (MHA), they introduced MLA, which:

- Compresses Key/Value into a latent space
- Reduces KV cache by ~93%
- Makes long-context inference much more efficient

I wanted to:

- Understand the intuition
- Connect all concepts (MHA → MQA → GQA → MLA)
- Implement it from scratch with RoPE integration

## Implementation Details

Key Concepts Used
Latent KV Compression
Low-Rank Factorization
Decoupled RoPE
Matrix Absorption (Inference)
KV Cache Optimization

## What This Project Covers

- MLA from scratch (PyTorch)

- RoPE integration (correct way)
- Training vs Inference distinction
- Absorption trick (DeepSeek-style optimization)
- Tensor shape intuition (very important)

## This project helped me:

Understand attention beyond surface level
See how real-world LLM optimizations work
Appreciate the importance of:

- Linear algebra
- Tensor shapes
- Memory vs compute trade-offs

## Closing Thought:

DeepSeek didn’t just optimize attention — they changed what attention stores and computes.
