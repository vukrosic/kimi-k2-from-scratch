# Code Snippets: Introduction to LLMs
# Runnable examples for key concepts. Run with: python code_snippets.py

import numpy as np
import random

print("=== Snippet 1: Simple LLM Pipeline Simulation ===")
# Simulate text → tokens → prediction
text = "The sky is"
tokens = ["The", "sky", "is"]  # Tokenized
vocab = ["blue", "clear", "cloudy", "apple"]
probs = np.array([0.35, 0.25, 0.17, 0.0002])
probs /= probs.sum()  # Normalize

print(f"Input tokens: {tokens}")
print(f"Vocab probs: {dict(zip(vocab, probs))}")

# Predict next
next_token = random.choices(vocab, weights=probs)[0]
print(f"Predicted next: '{next_token}'")

print("\n=== Snippet 2: Why Build from Scratch? ===")
# Placeholder: Print reasons (user expands)
reasons = [
    "Understand internals (e.g., attention).",
    "Customize for tasks.",
    "Apply to any Transformer model."
]
print("Reasons to build LLM from scratch:")
for reason in reasons:
    print(f"- {reason}")

print("\nRun tasks.md for exercises!")
