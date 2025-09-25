# Lesson 2.1.1: Embeddings - Initialization of Embed Tokens

## Theory

Token embeddings transform discrete token IDs into continuous vector representations, enabling the model to process language numerically. In Kimi-K2 (DeepSeek V3), embeddings are a core part of the input layer, mapping from a large vocabulary to high-dimensional space.

### Key Theory
- **Lookup Table**: Embeddings are a matrix \( E \in \mathbb{R}^{V \times D} \), where \( V \) is vocab_size, \( D \) is hidden_size (model dim). Each row is a learnable vector for a token.
- **Why Learn Embeddings?**: Captures semantic/syntactic similarities (e.g., "king" - "man" + "woman" ≈ "queen"). Initialized randomly, optimized via backprop.
- **Padding**: padding_idx (e.g., 0 for <pad>) allows masking; embeddings for pad are often zeroed or ignored in loss.
- **Tying Weights**: In some models, embed_tokens weights are tied to LM head for parameter efficiency (done in Kimi-K2 via _tied_weights_keys).

From config: vocab_size (e.g., 100352 for DeepSeek), hidden_size (e.g., 4096), pad_token_id.

### Mathematical View
For input \( x \in \{0, \dots, V-1\}^T \) (seq_len T), embeds = \( E[x] \), shape (batch, T, D).

## Code Walkthrough

Line from DeepseekV3Model.__init__:

```python
self.padding_idx = config.pad_token_id
self.vocab_size = config.vocab_size

self.embed_tokens = nn.Embedding(
    config.vocab_size, config.hidden_size, self.padding_idx
)
```

- `nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=p)`: Creates learnable matrix, initializes with normal dist (mean=0, std=init_range, e.g., 0.02).
- `padding_idx`: Sets weight[p] = 0, and gradients to 0 during training (no learning for pad).
- Post-init: Weights normalized in _init_weights (normal dist).

This initializes ~ V * D params (e.g., 100k * 4k = 400M params, ~1.6GB fp32).

## PyTorch Functions
- `nn.Embedding`: Core module, uses indexing for forward.
- Initialization: Handled in PreTrainedModel._init_weights (normal_ for Linear/Embedding).

## Why This Matters
Embeddings set the representational basis; size affects capacity. In MoE models like Kimi-K2, they remain dense.

Next: Forward pass of embeddings.
