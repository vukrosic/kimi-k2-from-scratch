# Lesson 2.1.1: Tasks - Embeddings Initialization

## Theory Exercises

1. **Lookup Table Explanation**: Describe the embedding matrix \( E \). If V=1000, D=128, what is the shape? How many parameters? Why is this memory-intensive for large V?

2. **Padding and Masking**: Why set padding_idx? How does it affect training (gradients)? Give an example where padding is crucial (e.g., batched sequences of different lengths).

3. **Weight Tying**: Explain weight tying between embed_tokens and lm_head. Pros (param efficiency), cons (fixed output dim). Is it used in Kimi-K2? (Yes, via _tied_weights_keys).

4. **Initialization Strategy**: Why normal distribution (mean=0, std=0.02)? How does it relate to Xavier/Glorot init? Discuss variance preservation in deep nets.

## Code Tasks

1. **Basic Initialization**:
   - Create nn.Embedding(vocab_size=100352, hidden_size=4096, padding_idx=0).
   - Print model.weight.shape.
   - Access embedding for token 1: model.weight[1]. Verify it's a tensor of size (4096,).

   ```python
   import torch
   import torch.nn as nn

   embed = nn.Embedding(100352, 4096, padding_idx=0)
   print(embed.weight.shape)  # Expected: torch.Size([100352, 4096])
   print(embed.weight[1].shape)  # torch.Size([4096])
   ```

2. **Parameter Count and Memory**:
   - Compute total params: `sum(p.numel() for p in embed.parameters())`.
   - Estimate FP32 memory: params * 4 bytes. For V=100k, D=4k: ~1.6GB.
   - Bonus: Use torch.manual_seed(42); print norm of first 5 embeddings (should be ~1 after init).

3. **Padding Effect**:
   - Init embed with padding_idx=0.
   - Set embed.weight[0] manually to torch.zeros(4096).
   - Forward: input_ids = torch.tensor([[0, 1, 0]]); embeds = embed(input_ids).
   - Verify embeds[0,0,:] is zeros, and gradients: embeds[0,0,:].requires_grad (False implicitly via padding).

4. **Custom Init**:
   - Subclass nn.Embedding, override init in __init__ to use uniform(-0.1, 0.1).
   - Compare weight norms before/after.

## Quiz

1. Shape of embed_tokens.weight?  
   a) (hidden_size, vocab_size) b) (vocab_size, hidden_size) c) (batch, seq, hidden)

2. True/False: padding_idx sets weight[p] = 0 and blocks gradients for that row.

3. What is the role of config.initializer_range? (Short: std for normal init)

## Advanced Task

Implement a tied embedding: Create embed_tokens and lm_head with shared weights (lm_head.weight = embed_tokens.weight). Forward input_ids through embed, then lm_head, show shape. Discuss tying in causal LM.

Submit code outputs and answers.
