# Lesson 1.1: Introduction to Large Language Models (LLMs)

## Theory

Large Language Models (LLMs) are neural networks trained on vast text corpora to understand and generate human-like language. At their core, modern LLMs like Kimi-K2 are based on the **Transformer architecture**, specifically the decoder-only variant used in models like GPT and DeepSeek.

### Key Concepts
- **Tokens and Vocabulary**: Text is tokenized into subword units (e.g., using Byte-Pair Encoding). The model learns embeddings for a fixed vocabulary size (e.g., 100k+ tokens).
- **Autoregressive Generation**: LLMs predict the next token given previous ones: \( p(x_t | x_1, \dots, x_{t-1}) \). This enables text completion.
- **Transformer Decoder**: Stacks layers of self-attention and feed-forward networks. No encoder; focuses on causal (masked) attention to prevent future peeking.
- **Scaling Laws**: Performance improves with model size (parameters), data, and compute. Kimi-K2 uses Mixture-of-Experts (MoE) for efficient scaling.

### Mathematical Foundation
The probability of a sequence \( X = (x_1, \dots, x_n) \) is:
\[
P(X) = \prod_{t=1}^n P(x_t | x_{<t})
\]
Training minimizes cross-entropy loss over next-token prediction.

Embeddings map tokens to vectors: \( e_i = E[x_i] \), where \( E \) is the embedding matrix.

## Code Walkthrough

In PyTorch, basic embedding setup:

```python
import torch
import torch.nn as nn

vocab_size = 50257  # Example GPT-2 vocab
hidden_size = 768   # Embedding dim
embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

input_ids = torch.tensor([[1, 2, 3]])  # Batch of sequences
embeds = embed_tokens(input_ids)  # Shape: (batch, seq_len, hidden_size)
print(embeds.shape)  # torch.Size([1, 3, 768])
```

- `nn.Embedding`: Lookup table. Indices → dense vectors.
- `padding_idx`: Ignores padding tokens (e.g., 0).

This is the starting point for input processing in Kimi-K2's `DeepseekV3Model`.

## Why This Matters
Understanding embeddings is crucial as they initialize the hidden states fed into transformer layers. Poor embeddings limit model capacity.

Next lesson: Kimi-K2 architecture overview.
