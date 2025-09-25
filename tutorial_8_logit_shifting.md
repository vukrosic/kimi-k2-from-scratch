# Tutorial 8: Logit Shifting for Causal LM Loss in DeepSeek-V3

## What is Logit Shifting?

In causal language models like DeepSeek-V3, the model predicts the next token given previous ones. During training, logits (raw predictions over vocab) are shifted rightward to align with labels: the prediction for position i is compared to label at i+1. The line `shift_logits = logits[..., :-1, :].contiguous()` slices the last position from logits, making it match the shifted labels (`shift_labels = labels[..., 1:].contiguous()`). This is a standard but crucial step for autoregressive loss computation using CrossEntropyLoss.

Less known: The `contiguous()` ensures the tensor is C-contiguous in memory for efficient ops, though often implicit.

### Key Benefits:
- Enables next-token prediction loss.
- Handles variable-length sequences.
- Compatible with padding via ignore_index=-100 in loss.

## Code Implementation

This appears in the `forward` method of `DeepseekV3ForCausalLM`:

```python
# Inside forward, after computing logits
logits = self.lm_head(hidden_states)
logits = logits.float()

loss = None
if labels is not None:
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
```

### Line-by-Line Breakdown

#### Context: Loss Computation Block
- `logits = self.lm_head(hidden_states)`: Projects hidden states to vocab_size logits, shape (batch_size, seq_len, vocab_size).
- `logits = logits.float()`: Ensures fp32 for loss stability.
- `if labels is not None:`: Only compute if training labels provided.

#### The Shifting Lines:
- `shift_logits = logits[..., :-1, :].contiguous()`:
  - `logits[..., :-1, :]`:
    - `...`: Ellipsis for batch and seq dimensions (equivalent to [:, :, :-1, :] but general).
    - `:-1`: Slices all but the last position along seq_len (dim=1).
    - `:`: All vocab dims.
    - Result: Shape (batch_size, seq_len-1, vocab_size). Removes the last prediction (no next token to predict).
  - `.contiguous()`: Ensures the tensor is contiguous in memory (copies if strided from slicing). Important for some ops like view() or CUDA kernels; prevents errors in flattened computations.
- `shift_labels = labels[..., 1:].contiguous()`:
  - Similar slicing: `...` for batch, `1:` starts from second position to end (shifts left, removes first label as no prior prediction).
  - Shape: (batch_size, seq_len-1).
  - `contiguous()`: Same memory fix.
- `loss_fct = CrossEntropyLoss()`: Standard loss (includes log_softmax + NLL).
- `shift_logits.view(-1, self.config.vocab_size)`: Flattens to (batch_size*(seq_len-1), vocab_size) for per-token loss.
- `shift_labels.view(-1)`: Flattens to (batch_size*(seq_len-1),).
- `shift_labels.to(shift_logits.device)`: Ensures device match (for multi-GPU).
- `loss = loss_fct(shift_logits, shift_labels)`: Computes average cross-entropy, ignoring -100 labels (padding).

This aligns: logit at pos i (predicting i+1) vs label at i+1.

## Step-by-Step Example Walkthrough

Consider a mini-example: batch_size=1, seq_len=4, vocab_size=5. Input tokens: [1,2,3,4], labels same (predict next).

1. Model outputs logits: shape (1,4,5), e.g., rows predict tokens 2,3,4,5 (but last has no target).
2. `shift_logits = logits[..., :-1, :]` -> (1,3,5): Rows 0-2 (predict 2,3,4).
3. `shift_labels = labels[..., 1:]` -> (1,3): [2,3,4].
4. Flatten: shift_logits (3,5), shift_labels [2,3,4].
5. CrossEntropy: For row0 vs 2, row1 vs 3, row2 vs 4. Loss = avg(-log(p_correct)).
6. If label[0]=-100 (ignore BOS), it's masked.

With contiguous: If slicing created strided tensor, view() might fail; contiguous copies to safe layout.

## Exercises

### Exercise 1: Manual Shift and Loss
Implement without slicing. Given logits (1,4,5) and labels (1,4):

```python
import torch
import torch.nn.functional as F

logits = torch.randn(1, 4, 5)  # Dummy
labels = torch.tensor([[1, 2, 3, 4]])  # Tokens

# Manual shift: Extract logits[0,0:3,:] vs labels[0,1:4]
manual_shift_logits = logits[:, :-1, :]  # Or manual indexing
manual_shift_labels = labels[:, 1:]
loss = F.cross_entropy(manual_shift_logits.view(-1, 5), manual_shift_labels.view(-1), ignore_index=-100)

print(f"Loss: {loss.item()}")
```

Verify matches the code's loss.

### Exercise 2: Visualize Alignment
Plot or print: For seq_len=5, show which logit row predicts which label.

```python
seq_len = 5
print("Logits positions: 0->label1, 1->label2, ..., 3->label4 (4 ignored)")
# Add padding: labels[0]=-100, compute loss only on 1-4.
```

### Exercise 3: Without Contiguous
Create strided tensor (e.g., transpose then slice), try view() – see RuntimeError. Add .contiguous() to fix.

### Exercise 4: Generation vs Training
In inference (no labels), no shift. Explain why shift only for training: Generation uses argmax on full logits sequentially.

Solutions: Run code; shift ensures teacher-forcing alignment. Contiguous fixes memory layout for efficiency.

## Why Use Logit Shifting in DeepSeek-V3?

This standard technique enables efficient next-token training in causal LMs. In DeepSeek-V3, it integrates with its large vocab (128K+?) and long contexts, using contiguous for GPU perf. Essential for fine-tuning/generation pipelines.

Next: Perhaps full training loop.
