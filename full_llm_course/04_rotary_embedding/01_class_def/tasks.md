# Tasks: DeepseekV3RotaryEmbedding Class Definition

## Task 1: Theory - RoPE Basics (5 min)
- Read line: `class DeepseekV3RotaryEmbedding(nn.Module):`
- Explain in 1 para: What is RoPE? Why class? Role in attention?
- Hint: Rotation for relative pos.

## Task 2: Code - Inheritance Check (5 min)
```python
import torch.nn as nn

class MyRoPE(nn.Module):
    pass

model = MyRoPE()
print(isinstance(model, nn.Module))  # Should be True
```
- Run, explain output. Add `def __init__(self): super().__init__()`—why needed?

## Task 3: PyTorch Function - nn.Module (5 min)
- List 3 nn.Module benefits (e.g., .parameters() for optim).
- Modify class: Add dummy param `self.d = nn.Parameter(torch.tensor(64))`. Print `list(model.parameters())`.

## Submission
- Write explanations.
- Share code output.

Solutions: RoPE rotates Q/K for pos; class for PyTorch. Init calls parent for setup.
