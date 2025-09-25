# Lesson 2.1.2: Tasks - Embeddings Forward Pass

## Theory Exercises

1. **Forward Lookup**: Explain the embedding forward mathematically. For input_ids = [[1,2]], with E rows e1, e2, what is hidden_states? Why no addition of positions here (contrast with absolute PE models)?

2. **Inputs vs Embeds Flexibility**: When would you use inputs_embeds instead of input_ids? Examples: Custom embeddings for images/text, fine-tuning with frozen embeds, or multimodal models.

3. **Efficiency Considerations**: Why is embedding lookup fast? Discuss GPU implications for large batch/seq. What if V is 1M+? (Hint: Sparse matrices or hashing.)

4. **Error Handling**: Why raise ValueError if both input_ids and inputs_embeds provided? How does the code infer batch_size, seq_length?

## Code Tasks

1. **Basic Forward**:
   - Init embed_tokens as in previous lesson (V=100352, D=4096, pad=0).
   - input_ids = torch.tensor([[1, 2, 3, 0]])  # With pad
   - hidden = embed_tokens(input_ids)
   - Print hidden.shape, hidden[0,0,:5] (first 5 dims of first token), verify hidden[0,3,:] is zeros (pad).

   ```python
   import torch
   import torch.nn as nn

   embed_tokens = nn.Embedding(100352, 4096, padding_idx=0)
   input_ids = torch.tensor([[1, 2, 3, 0]])
   hidden = embed_tokens(input_ids)
   print(hidden.shape)  # (1,4,4096)
   print(hidden[0,0,:5])  # Non-zero
   print(torch.allclose(hidden[0,3,:], torch.zeros(4096)))  # True for pad
   ```

2. **Inputs_Embeds Usage**:
   - Create fake inputs_embeds = torch.randn(1, 5, 4096)
   - In a mock forward function, if inputs_embeds is not None, use it; else embed_tokens(input_ids).
   - Test both paths: Print shapes, ensure seq_length inferred correctly.

3. **Manual Gather Simulation**:
   - Take embed_tokens.weight[:5] (first 5 rows).
   - indices = torch.tensor([1,3])
   - Manual embeds = torch.index_select(embed_tokens.weight, 0, indices.unsqueeze(0).expand(1, -1))
   - Compare to embed_tokens(torch.tensor([[1,3]])). Verify equality.

4. **Batch with Padding**:
   - input_ids = torch.tensor([[1,2,0], [3,0,4]])  # Different lengths implicitly padded
   - Compute embeds, mask pads: Create attention_mask = (input_ids != 0).float()
   - Bonus: Sum embeds * mask.unsqueeze(-1) to ignore pads in mean pooling.

## Quiz

1. Shape of hidden_states from forward? For B=2, T=10:  
   a) (2,4096,10) b) (2,10,4096) c) (10,2,4096)

2. True/False: Positions are added to embeds in this forward (RoPE handles later).

3. What happens if inputs_embeds=None and input_ids=None? (Short: ValueError)

## Advanced Task

Implement a forward wrapper that supports both, adds optional absolute PE (sin/cos). Test with small embed (V=10, D=8), input_ids=[[1,2]], show before/after PE addition. Discuss why Kimi-K2 skips absolute PE.

Submit code, outputs, and answers.
