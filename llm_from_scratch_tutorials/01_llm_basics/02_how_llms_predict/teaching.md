# Micro-Lecture 1.2: How LLMs Predict Tokens

## Overview
This micro-lecture explains prediction: From input to probability distribution, sampling methods (greedy, random, top-K, temperature). Builds on intro—focus on why probs and how to choose tokens. (10-15 min read + 10 min exercise.)

### From Input to Prediction
1. Text → Tokens (e.g., "The sky is" → ["The", " sky", " is"]).
2. Embeddings: Numbers encoding meaning/context.
3. Process (attention/MLP): Infuse context.
4. Output Head: Vector → Probs over vocab (softmax).

Example: Input "The sky is" → Probs:
- "blue": 0.35 (frequent pattern).
- "clear": 0.25.
- "cloudy": 0.17.
- "apple": 0.0002 (low, doesn't fit).
- Sum = 1.0 (normalized).

Probs from data: "sky is blue" common → High P("blue").

### Why Probabilities?
- Captures uncertainty: Multiple sensible continuations.
- Learned patterns: Data shows "blue" follows often.
- Enables control: Sampling varies output (creative vs. deterministic).

### Sampling: Choosing from Probs
Pick one token to append.

1. **Greedy**: Max prob (e.g., "blue"). Fast, consistent, but repetitive ("the the the...").
   - Use: Math/coding (exact answers).

2. **Random (Nucleus)**: Sample by probs ("blue" 35% chance).
   - Use: Creative text, but risky (nonsensical).

3. **Top-K**: Sample from top K highest probs (e.g., K=3: "blue", "clear", "cloudy").
   - Balances: Limits bad choices, adds variety.

4. **Temperature**: Adjust probs before sampling.
   - Formula: adjusted_p = exp(log(p) / temp) / sum.
   - Temp <1 (e.g., 0.7): Sharpen (highs higher, deterministic).
   - Temp >1 (e.g., 1.5): Flatten (more random).
   - Example: Temp=0.5 → P("blue") ~0.5, others lower.

Combine: Top-K + Temp for production (e.g., ChatGPT).

### Step-by-Step: Prediction Example
Input: "The sun is shining and the sky is"

1. Tokens: ["The", " sun", " is", " shining", " and", " the", " sky", " is"].
2. Embeddings: Matrix (8, embed_dim).
3. Attention/MLP: Context-infused (e.g., "sky" knows "shining").
4. LM Head: Linear to (8, vocab_size), softmax on last row → Probs.
5. Sample: E.g., top-K=5, temp=0.8 → Pick "blue".

Repeat: Append "blue" → New input, predict next.

## Quick Check
- Q: Why not always greedy? A: Too repetitive.
- Q: Temp=0.1 effect? A: More deterministic.

## Next
Micro-Lecture 1.3: Tokens in Detail.
