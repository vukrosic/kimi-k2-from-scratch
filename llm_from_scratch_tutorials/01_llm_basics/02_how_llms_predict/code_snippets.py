# Code Snippets: How LLMs Predict Tokens
# Runnable examples for prediction and sampling. Run with: python code_snippets.py

import numpy as np
import torch
import torch.nn.functional as F
import random

print("=== Snippet 1: From Logits to Probs (Softmax) ===")
logits = torch.tensor([2.0, 1.0, 0.5, -5.0])  # Raw outputs for ["blue", "clear", "cloudy", "apple"]
probs = F.softmax(logits, dim=0)
print(f"Logits: {logits.numpy()}")
print(f"Probs: {probs.numpy()} (sum=1.0)")

print("\n=== Snippet 2: Sampling Methods ===")
vocab = ["blue", "clear", "cloudy", "apple"]
probs_np = probs.numpy()

# Greedy
greedy_idx = np.argmax(probs_np)
greedy_token = vocab[greedy_idx]
print(f"Greedy: '{greedy_token}' (prob {probs_np[greedy_idx]:.2f})")

# Random
random_token = random.choices(vocab, weights=probs_np)[0]
print(f"Random: '{random_token}'")

# Top-K (K=2)
k = 2
top_k_probs = probs_np[np.argsort(probs_np)[-k:][::-1]]
top_k_vocab = [vocab[i] for i in np.argsort(probs_np)[-k:][::-1]]
top_k_token = random.choices(top_k_vocab, weights=top_k_probs)[0]
print(f"Top-K=2: '{top_k_token}'")

# Temperature
def sample_temp(probs, temp=1.0):
    logits = torch.log(torch.tensor(probs) + 1e-10)
    adjusted = logits / temp
    adjusted_probs = F.softmax(adjusted, dim=0).numpy()
    return random.choices(vocab, weights=adjusted_probs)[0]

low_temp = sample_temp(probs_np, temp=0.5)
high_temp = sample_temp(probs_np, temp=1.5)
print(f"Temp=0.5: '{low_temp}' (sharper)")
print(f"Temp=1.5: '{high_temp}' (more random)")

print("\nRun tasks.md for exercises!")
