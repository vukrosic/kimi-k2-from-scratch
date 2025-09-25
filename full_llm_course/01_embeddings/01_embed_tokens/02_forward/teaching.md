# Lesson 2.1.2: Embeddings - Forward Pass of Embed Tokens

## Theory

The forward pass of embeddings converts input token IDs into hidden states, the starting point for transformer processing. In Kimi-K2, this happens early in the model's forward method, allowing flexibility for pre-computed embeds.

### Key Theory
- **Forward Operation**: Given input_ids (batch x seq_len), lookup rows from embedding matrix: hidden_states = E[input_ids], shape (batch, seq_len, hidden_size).
- **Inputs vs Embeds**: Model supports either raw input_ids (lookup here) or pre-embedded inputs_embeds (e.g., for custom tokenizers or fine-tuning). If both provided, error.
- **No Positional Encoding Here**: Unlike BERT/GPT-2, Kimi-K2 uses RoPE in attention, so no absolute positions added to embeds. Pure token semantics.
- **Efficiency**: Embedding lookup is O(1) per token via indexing, but large V requires sparse access in practice.

Mathematical: For input_ids \( I \in \mathbb{N}^{B \times T} \), hidden = \( [E_{i_{b,t}}]_{b,t} \), where \( i_{b,t} = I[b,t] \).

## Code Walkthrough

Lines from DeepseekV3Model.forward:

```python
# retrieve input_ids and inputs_embeds
if input_ids is not None and inputs_embeds is not None:
    raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
elif input_ids is not None:
    batch_size, seq_length = input_ids.shape[:2]
elif inputs_embeds is not None:
    batch_size, seq_length = inputs_embeds.shape[:2]
else:
    raise ValueError("You have to specify either input_ids or inputs_embeds")

...

if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
```

- Checks mutual exclusivity, infers batch/seq from input.
- If no inputs_embeds, call self.embed_tokens(input_ids): PyTorch gathers rows from weight matrix.
- Result: inputs_embeds (B, T, D), used as hidden_states for layers.
- Later: position_ids computed if None, but not added here—used in RoPE.

PyTorch Internals: nn.Embedding.forward uses torch.index_select or gather for batch indexing.

Example:
```python
input_ids = torch.tensor([[1, 2, 3]])  # B=1, T=3
embeds = embed_tokens(input_ids)  # (1,3,D), each row from weight[1], weight[2], weight[3]
```

## Why This Matters
This step bridges discrete text to continuous space. Flexibility with inputs_embeds allows advanced usage (e.g., multimodal embeds).

Next: Utility methods like get_input_embeddings.
