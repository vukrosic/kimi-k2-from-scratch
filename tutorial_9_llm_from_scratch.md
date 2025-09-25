# Tutorial 9: Building an LLM from Scratch - Basics to Attention (Based on Llama Concepts)

This tutorial is inspired by a comprehensive video explanation of coding an LLM (like Llama) from scratch. We'll cover the fundamentals step-by-step, starting from beginner concepts and building up to the core self-attention mechanism. The goal is to demystify LLMs, showing how they work under the hood. We'll use simple Python examples, math explanations, and code snippets. By the end, you'll understand the shared architecture behind models like Llama, GPT, DeepSeek, and more. (This covers the first ~1 hour of content, focusing on basics, tokens, embeddings, tokenization, and attention intro.)

## Prerequisites
- Basic Python knowledge (lists, dictionaries, loops).
- High school math (vectors, basic probability).
- No prior AI experience needed—we'll explain everything.

Realistic note: LLMs involve PhD-level ideas, but we'll break them down intuitively. It may take days/weeks to fully grasp, but once you do, adapting to any LLM is straightforward.

All code is runnable in Python with NumPy/PyTorch (install via `pip install torch numpy`). Repository for materials: [GitHub link placeholder—adapt from transcript].

## 1. Introduction to Large Language Models (LLMs)
LLMs are AI systems that generate human-like text. Think ChatGPT: You input a message, and it responds coherently. But at their core, LLMs are statistical models that predict the next word (or "token") in a sequence based on previous context.

### Why Build from Scratch?
- Understand the math/Python behind models like Llama, GPT, DeepSeek.
- Customize for your needs (e.g., fine-tune on specific data).
- No high-end hardware needed for learning—we focus on concepts, not full training.

Famous LLMs: GPT (powers ChatGPT), Claude, Gemini, DeepSeek, Qwen, Grok, Llama (Meta). They share ~90% similar architecture (Transformers). Master one, master them all.

## 2. How LLMs Work: Predicting the Next Token
An LLM takes text input, analyzes it, and outputs the most likely next token. Tokens are the building blocks (more on this later)—could be words, subwords, characters, or symbols.

### Basic Example
Input: "The sun is shining and the sky is"

Possible outputs: "blue" (makes sense), "cloudy" (possible), "apple" (unlikely).

LLM doesn't pick one word—it computes a **probability distribution** over its entire vocabulary (e.g., 50,000+ tokens).

- P("blue") = 0.35
- P("clear") = 0.25
- P("cloudy") = 0.17
- P("apple") = 0.0002
- ... (sums to 1.0)

### Why Probabilities?
LLMs learn patterns from massive data (e.g., internet text). "Blue" follows "sky is" often, so high probability. Rare words like "apple" get low probs.

### Sampling: Choosing the Next Token
Once probabilities are computed, how to pick?

1. **Greedy Sampling**: Pick the highest probability (e.g., "blue"). Simple, but repetitive (e.g., always "the" after "a").
   - Good for math/coding (one correct answer).
   - Bad for creative writing.

2. **Random Sampling**: Pick based on probabilities (roulette wheel: "blue" 35% chance).
   - More creative, but can be nonsensical.

3. **Top-K Sampling**: Narrow to top K (e.g., K=3: "blue", "clear", "cloudy"), then sample from them.
   - Balances creativity and coherence.

4. **Temperature Sampling**: Modify probabilities before sampling.
   - Low temperature (e.g., 0.1): Sharpen highs/lows (deterministic, like greedy).
   - High temperature (e.g., 1.5): Flatten probs (more random/creative).
   - Formula: New probs = original_probs^(1/temperature), then normalize.

   Example: Temperature=0.5 on above → P("blue") even higher (~0.5), others lower.

Often, combine methods (e.g., top-K + temperature) for better outputs.

### Exercise 1: Simulate Sampling
```python
import numpy as np
import random

# Dummy probs for tokens: ['blue', 'clear', 'cloudy', 'apple']
probs = [0.35, 0.25, 0.17, 0.0002]  # Sums ~0.77; normalize in real code
tokens = ['blue', 'clear', 'cloudy', 'apple']

# Normalize
probs = np.array(probs)
probs /= probs.sum()

# Greedy
greedy = tokens[np.argmax(probs)]
print(f"Greedy: {greedy}")

# Random (with temperature=1.0)
def sample_with_temp(probs, temp=1.0):
    adjusted = np.log(probs + 1e-10) / temp  # Avoid log(0)
    adjusted = np.exp(adjusted)
    adjusted /= adjusted.sum()
    return random.choices(tokens, weights=adjusted)[0]

random_token = sample_with_temp(probs)
print(f"Random: {random_token}")

# Top-K (K=2)
top_k_indices = np.argsort(probs)[-2:][::-1]
top_k_probs = probs[top_k_indices]
top_k_token = random.choices([tokens[i] for i in top_k_indices], weights=top_k_probs)[0]
print(f"Top-2: {top_k_token}")
```
Run this: Observe how sampling affects output variety.

## 3. Tokens: The Building Blocks of Text
Tokens are the units LLMs process—not always whole words. Vocabulary: All possible tokens (e.g., 50K-150K in Llama/Qwen).

### What Can Be a Token?
- Words: "car", "run".
- Subwords: "run" + "ning" → "running".
- Characters: "c", "a", "r".
- Symbols/Emojis: "!", "🚀".
- Phrases/Sentences: "The sun rises in the east." (rare, due to vocab explosion).
- Multi-language: Chinese "车" (chē, meaning "car"), Japanese kanji.

Example: Token "车" learns: Vehicle, transportation, compound words (e.g., "汽车" = car).

Emoji "🌳": Tree/forest, used for nature/emotion.

Arbitrary: "abc123!": Random string, low semantic meaning (e.g., password/code).

### Semantic Meaning
Each token's vector embedding (array of numbers) encodes its "meaning":
- "car": High in "vehicle", "wheels", "transport"; low in "alive", "fluffy".
- "grass": High in "green", "alive", "grows"; low in "vehicle".

LLM learns this from data: Adjusts numbers so correct next tokens have high probability.

### Why Not Letters Only? Words Only? Sentences?
- **Letters (44 tokens for "the quick brown...")**: 44x compute (each token same cost). Hard to learn combinations.
- **Words**: Vocab explodes (run, runs, ran, running → millions). Can't handle misspellings ("runnig" → unknown).
- **Sentences**: Even worse—trillions possible. Rigid, no new phrases.

**Solution: Subwords (BPE)**: Balance—group common pairs (e.g., "run" + "ning"). Vocab ~150K. Handles new words ("un" + "believ" + "able" → "unbelievable").

Example: "running" → ["run", "ning"] (2 tokens vs. 7 letters).

Misspellings: "runnig" → "run" + "nig" (infer from parts).

## 4. Vector Embeddings: Encoding Meaning
Tokens → Indices (0 to vocab_size-1) → Vector Embeddings (arrays of ~1K-7K numbers).

### What is a Vector Embedding?
Array representing token's semantics. E.g., dim=4 (real: 1K+):
- "car": [0.8 (speed), 0.2 (alive), 0.3 (green), 0.1 (fluffy)]
- "grass": [0.1 (speed), 0.9 (alive), 0.95 (green), 0.7 (fluffy)]

Each dimension = feature (unknown to us—AI learns them):
- Dim1: Movement speed.
- Dim2: Aliveness.
- Features relative: Differences between tokens (e.g., car vs. grass in "vehicle-ness").

AI doesn't "think" like humans—numbers capture patterns statistically.

### Example: Dog vs. Cat
- "dog": [0.42 (fluffiness), 0.8 (save you?), 0.9 (grateful)]
- "cat": [0.73 (fluffiness), 0.1 (save you?), 0.3 (grateful)]

Dimensions abstract: Could be "playfulness", "independence", etc. Order/mixing unknown—research ongoing.

### Training Embeddings
- Input text → Tokens → Indices → Lookup embeddings.
- Predict next token: Adjust embeddings so correct token's prob ↑.
- Data: Trillions of tokens (internet). Iteratively: Predict → Error → Update (backprop).

No manual features—AI discovers via gradient descent.

### Exercise 2: Embeddings Visualization
```python
import numpy as np
import matplotlib.pyplot as plt

# Dummy embeddings (dim=2 for plot)
embeddings = {
    'car': np.array([0.8, 0.2]),
    'grass': np.array([0.1, 0.9]),
    'dog': np.array([0.4, 0.8]),
    'cat': np.array([0.7, 0.3])
}

fig, ax = plt.subplots()
for token, emb in embeddings.items():
    ax.scatter(emb[0], emb[1])
    ax.annotate(token, emb)

ax.set_xlabel('Dim 1 (e.g., Speed/Vehicle)')
ax.set_ylabel('Dim 2 (e.g., Alive/Green)')
plt.title('2D Embedding Visualization')
plt.show()
```
Plot: See how similar tokens cluster (e.g., animals vs. objects).

## 5. From Text to Embeddings: The Pipeline
1. Text: "The dog ran quickly."
2. Tokenize: ["The", " dog", " ran", " quick", "ly", "."] (subwords + spaces).
3. Indices: [5, 1234, 567, 890, 12, 3] (lookup in vocab).
4. Embeddings: Matrix (seq_len, embed_dim), e.g., (6, 1024).
   - Row 0: Embedding for "The" (lookup index 5).
   - No fixed mapping—vocab fixed, embeddings swapable (e.g., different model versions).

Why indices? Fast lookup (O(1)) vs. string search. Integers use less memory.

## 6. Tokenization: Building a BPE Tokenizer from Scratch
BPE (Byte-Pair Encoding): Algorithm to create subword tokens from text corpus. Trades granularity vs. efficiency.

### Step 1: Prepare Corpus
Text data (e.g., documents). Example corpus:
- Doc1: "this is the first document"
- Doc2: "this is the second document"
- etc.

### Step 2: Initial Vocabulary & Pre-Tokenize
- Unique chars: [' ', '!', '?', 'a', 'b', ..., '<end>'] (add special tokens).
- Pre-tokenize: Split into words, then chars + '<end>':
  - "this" → ('t', 'h', 'i', 's', '<end>') : count=2 (appears twice).
  - Dict: {('t','h','i','s','<end>'): 2, ...}

Code:
```python
from collections import defaultdict

corpus = ["this is the first document", "this is the second document", ...]  # Full corpus

# Initial vocab: unique chars + <end>
unique_chars = set(''.join(corpus))
vocab = sorted(list(unique_chars)) + ['<end>']
print(f"Initial vocab: {vocab}, size: {len(vocab)}")  # e.g., 20

# Pre-tokenize: words → char tuples + count
word_splits = defaultdict(int)
for doc in corpus:
    words = doc.split(' ')
    for word in words:
        if word:  # Skip empty
            chars = list(word) + ['<end>']
            word_tuple = tuple(chars)
            word_splits[word_tuple] += 1

print("Sample word_splits:", dict(list(word_splits.items())[:3]))
```

### Step 3: Get Pair Stats (Frequency of Adjacent Pairs)
Count how often char pairs appear across words.

Code:
```python
def get_pair_stats(splits):
    pair_counts = defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)  # To mutable list
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            pair_counts[pair] += freq
    return pair_counts

pair_stats = get_pair_stats(word_splits)
print("Most frequent pairs:", sorted(pair_stats.items(), key=lambda item: item[1], reverse=True)[:5])
# e.g., (('s', '<end>'), 8), (('t', 'h'), 7), ...
```

### Step 4: Merge Most Frequent Pair
Replace frequent pairs with new token (e.g., 's<end>' → new token).

Code:
```python
def merge_pair(best_pair, splits):
    first, second = best_pair
    merge_token = first + second  # New token, e.g., 's<end>'
    new_splits = defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == first and symbols[i+1] == second:
                new_symbols = symbols[:i] + [merge_token] + symbols[i+2:]
                i += 2  # Skip merged
            else:
                new_symbols = symbols[:i+1] + symbols[i+1:]  # No merge
                i += 1
        new_tuple = tuple(new_symbols)
        new_splits[new_tuple] += freq
    return new_splits

# Example merge (most frequent: ('s', '<end>'))
best_pair = max(pair_stats, key=pair_stats.get)
new_splits = merge_pair(best_pair, word_splits)
print("After merge:", dict(list(new_splits.items())[:3]))
```

### Step 5: BPE Loop (Train Merges)
Repeat: Get pairs → Merge best → Update vocab/merges (e.g., 15 merges).

Code:
```python
num_merges = 15
merges = {}  # Track merges: {('s', '<end>'): 's<end>', ...}
current_splits = word_splits.copy()
vocab_set = set(vocab)  # For lookup

for i in range(num_merges):
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        break
    best_pair = max(pair_stats, key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Merge {i+1}: {best_pair} (freq {best_freq})")
    
    merge_token = ''.join(best_pair)  # New token
    merges[best_pair] = merge_token
    vocab_set.add(merge_token)
    
    current_splits = merge_pair(best_pair, current_splits)

print(f"Final vocab size: {len(vocab_set)}")
print("Sample merges:", list(merges.items())[:5])
```

Output example:
- Merge 1: ('s', '<end>') → 's<end>'
- ... Up to 15 merges, vocab grows to ~35.
- Common: 'th', 'the', 'docu' (frequent in corpus).

### Encoding with BPE (Apply Merges)
To tokenize new text: Split to chars → Apply merges in order → Indices.

(Full encoder code omitted for brevity—see BPE impl in Hugging Face.)

### Why BPE?
- Vocab ~150K (efficient).
- Handles OOV (out-of-vocab): Fall back to subwords.
- Rare words: Compose from parts (e.g., "unbelievable" = "un" + "believ" + "able").

### Exercise 3: Build Your BPE
Use the code above on a small corpus (e.g., sentences about animals). Run 10 merges. What tokens form? Encode "the quick brown fox" → Indices.

## 7. Self-Attention: Infusing Context
Now, embeddings have meaning but no position/context. Self-attention modifies each embedding with relevant previous context.

### The Problem
Input: "The dog ran quickly."
- Embeddings: Semantic only (dog = [fluffy, animal, ...]).
- Goal: "quickly" embedding → Infuse "dog ran" context (not future ".").

During training: Predict every token (teacher forcing). Inference: Predict last.

### Solution: Query, Key, Value (QKV)
For each token:
- **Query (Q)**: What context am I looking for? (Derived from embedding.)
- **Key (K)**: What context do I offer? (Derived from each previous embedding.)
- **Value (V)**: The actual context to add (embedding or projection).

Q, K, V: Smaller dims (e.g., 512 vs. embedding 1024). Positional encoding added (later).

### Computing Attention Scores
- For token i (e.g., "quickly"): Compute Q_i · K_j for each j < i (dot product).
- Score = similarity (higher = more relevant).
- Dot product: Sum (Q_i[k] * K_j[k]) for dim k. Measures alignment (cosine-like).

Example (2D for viz):
- Q_quickly = [0.8, 0.4]
- K_dog = [0.7, 0.5] → Dot = 0.8*0.7 + 0.4*0.5 = 0.86 (high, relevant).
- K_cat (earlier) = [0.2, 0.9] → Dot = 0.8*0.2 + 0.4*0.9 = 0.52 (low).

Normalize scores (softmax) → Weights (sum=1).

### Attention Matrix
Matrix of scores (seq_len x seq_len):
- Rows: Queries (tokens seeking context).
- Cols: Keys (tokens offering context).
- Upper triangle: 0 (no future peeking, causal mask).

Example ("Life is short, eat dessert first"):
- Scores for "eat": High self, "short", "is"; low "dessert" (future=0).

### Applying Attention
- Weighted sum: New_embedding_i = sum (weight_j * V_j) for j <= i.
- Includes self (high weight) + relevant prior.

Positional: Modify Q/K with position (RoPE/absolute—later tutorial).

### Exercise 4: Simple Dot Product Attention
```python
import torch
import torch.nn.functional as F

# Dummy embeddings (seq=4, dim=3)
embeds = torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]])  # (1,4,3)

# Project to QKV (simplified: linear = identity)
Q = embeds  # (1,4,3)
K = embeds
V = embeds

# Scores: Q @ K.transpose(-2,-1) / sqrt(dim)  (causal mask later)
scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(3)
print("Raw scores:\n", scores.squeeze())  # 4x4 matrix

# Softmax (per row)
attn_weights = F.softmax(scores, dim=-1)
print("Attention weights:\n", attn_weights.squeeze())

# Output: attn_weights @ V
output = torch.matmul(attn_weights, V).squeeze()
print("Context-infused embeddings:\n", output)
```
Run: See how each position attends to priors (add causal mask: scores[:, :, future]= -inf before softmax).

## Next Steps
This covers basics to attention intro. Next: Positional encoding, full Transformer layers, training loop. Practice exercises—explain concepts in your words!

(Adapted from transcript; concepts universal to LLMs like Llama/DeepSeek.)
