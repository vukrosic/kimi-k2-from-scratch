# Tests: How LLMs Predict Tokens
# Run with: python tests.py
# Verifies probs and sampling understanding

import torch
import torch.nn.functional as F
import numpy as np
import random

def test_softmax():
    logits = torch.tensor([2.0, 1.0, 0.5, -5.0])
    probs = F.softmax(logits, dim=0)
    assert np.isclose(probs.sum().item(), 1.0, atol=1e-6), "Probs must sum to 1"
    assert probs[0].item() > probs[3].item(), "Higher logit → higher prob"
    print("✓ Softmax test passed")

def test_sampling():
    vocab = ["blue", "clear", "cloudy", "apple"]
    probs = F.softmax(torch.tensor([2.0, 1.0, 0.5, -5.0]), dim=0).numpy()
    
    # Greedy
    greedy = vocab[np.argmax(probs)]
    assert greedy == "blue", f"Greedy should be 'blue', got {greedy}"
    print("✓ Greedy test passed")
    
    # Random (run multiple for variance)
    samples = [random.choices(vocab, weights=probs)[0] for _ in range(10)]
    unique = len(set(samples))
    assert unique > 1, "Random should vary"
    print(f"✓ Random test: {unique} unique samples (expect >1)")

def test_temperature():
    probs = F.softmax(torch.tensor([2.0, 1.0]), dim=0).numpy()
    low_temp = F.softmax(torch.tensor([2.0, 1.0]) / 0.5, dim=0).numpy()
    assert low_temp[0] > probs[0], "Low temp sharpens"
    print("✓ Temperature test passed")

if __name__ == "__main__":
    test_softmax()
    test_sampling()
    test_temperature()
    print("All tests complete. Review tasks for deeper learning.")
