# Lesson 1.1: Tasks - Introduction to LLMs

## Theory Exercises

1. **Explain Autoregressive Generation**: Describe in your own words how LLMs predict text autoregressively. Why is causal masking necessary in the transformer decoder? Provide a simple example with a 3-token sequence.

2. **Scaling Laws**: Research and summarize Kaplan et al.'s scaling laws for LLMs. How does MoE (like in Kimi-K2) help with efficient scaling compared to dense models?

3. **Mathematical Derivation**: Derive the cross-entropy loss for next-token prediction. Given logits \( z_t \) for token \( x_t \), write the loss formula and explain why we use softmax.

## Code Tasks

1. **Basic Embedding Implementation**:
   - Create a PyTorch `nn.Embedding` layer with vocab_size=10000, hidden_size=512, padding_idx=0.
   - Input: `input_ids = torch.tensor([[0, 5, 10, 0]])` (note padding).
   - Output the shape of embeds and print the embedding for token 5.
   - Bonus: Manually compute the embedding for token 5 using indexing on the weight matrix.

   ```python
   # Starter code
   import torch
   import torch.nn as nn
   # Your code here
   ```

2. **Token to Vector Mapping**:
   - Initialize random embeddings (seed=42 for reproducibility).
   - Show how embeddings for similar tokens (e.g., if you had a toy vocab) might cluster post-training, but for now, just compute cosine similarity between two random token embeddings.

3. **Padding Handling**:
   - Modify the embedding code to mask padding tokens (set to zero vector).
   - Verify the output for the padded input.

## Quiz

1. What is the shape of `embeds` for batch_size=2, seq_len=10, hidden_size=1024?  
   a) (2, 1024, 10) b) (2, 10, 1024) c) (10, 2, 1024)

2. True/False: Embeddings are learned during training via backpropagation on the embedding weights.

3. Why use subword tokenization? (Short answer)

## Advanced Task

Implement a simple tokenizer stub (e.g., split on spaces) and embed a sentence: "Hello world". Discuss limitations.

Submit your code and answers in a notebook or file for review.
