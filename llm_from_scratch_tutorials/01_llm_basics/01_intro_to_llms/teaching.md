# Micro-Lecture 1.1: Introduction to Large Language Models (LLMs)

## Overview
This micro-lecture introduces LLMs as text-generating AI. We'll cover what they are, why build from scratch, and famous examples. (5-10 min read + 5 min exercise.)

### What is an LLM?
LLMs are AI that generate human-like text by predicting the next "token" (word/part) in a sequence. E.g., Input: "The sky is"; Output: "blue" (likely).

Core: Statistical models trained on internet-scale data to learn patterns.

### Why Build from Scratch?
- Demystify: Understand math/code behind Llama, GPT, DeepSeek.
- Customize: Fine-tune for tasks (no black box).
- Learn: Concepts apply to all Transformers-based AI.
- Low hardware: Focus on code/math, not full training.

Famous LLMs:
- GPT (ChatGPT)
- Llama (Meta)
- DeepSeek, Qwen, Claude, Gemini, Grok

~90% shared (Transformers). Master basics → Adapt any.

### Step-by-Step: LLM in Action
1. Input text → Tokens (breakdown).
2. Embeddings (numbers encoding meaning).
3. Process (attention/MLP) → Context-rich vectors.
4. Predict: Probability over vocab → Sample token.
5. Repeat for sequence.

Example: "Hello, world!" → Predicts coherent continuation.

## Quick Check
- Q: What's the core task of an LLM? A: Next-token prediction.
- Q: Why probabilities? A: Captures uncertainty/patterns.

## Next
Micro-Lecture 1.2: How LLMs Predict Tokens.
