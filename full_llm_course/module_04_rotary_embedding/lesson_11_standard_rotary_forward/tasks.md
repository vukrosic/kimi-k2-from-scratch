# Lesson 4.1.3: Tasks - RoPE Standard Forward Method

## Theory Exercises

1. **Cache Check Logic**: Why check if seq_len > max_seq_len_cached? How does it support variable batch seq lengths? What if seq_len < cached (just slice, no recompute).

2. **Slicing and Broadcasting**: cos/sin (seq_len, dim) broadcast to Q/K (bs, heads, seq_len, head_dim=dim). Explain how in apply_rotary_pos_emb: cos[position_ids].unsqueeze(unsqueeze_dim).

3. **Dtype Cast in Forward**: Why to(x.dtype)? If cache fp32, Q fp16, what happens without cast (precision loss in matmul)? Discuss for AMP training.

4. **Dummy x Role**: Why pass x (not used for computation)? (For device/dtype inference if seq_len=None; shape[-2] for seq_len).

## Code Tasks

1. **Basic Forward Test**:
   - Define full DeepseekV3RotaryEmbedding class (init, _set_cos_sin_cache, forward).
   - rope = DeepseekV3RotaryEmbedding(dim=32, max_position_embeddings=5)
   - x = torch.randn(1,4,7,32)  # seq=7 >5, should extend cache
   - cos, sin = rope(x)
   - Print cos.shape (7,32), rope.max_seq_len_cached (7)
   - Second call with seq=3: cos.shape (3,32), no extend

   ```python
   import torch
   import torch.nn as nn

   class DeepseekV3RotaryEmbedding(nn.Module):
       def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
           super().__init__()
           self.dim = dim
           self.max_position_embeddings = max_position_embeddings
           self.base = base
           inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
           self.register_buffer("inv_freq", inv_freq, persistent=False)
           self._set_cos_sin_cache(max_position_embeddings, self.inv_freq.device, torch.get_default_dtype())
           self.max_seq_len_cached = max_position_embeddings  # Fix from None

       def _set_cos_sin_cache(self, seq_len, device, dtype):
           self.max_seq_len_cached = seq_len
           t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
           freqs = torch.outer(t, self.inv_freq.to(t.device))
           emb = torch.cat((freqs, freqs), dim=-1)
           self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
           self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

       def forward(self, x, seq_len=None):
           if seq_len is None:
               seq_len = x.shape[-2]
           if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
               self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
           return (
               self.cos_cached[:seq_len].to(dtype=x.dtype),
               self.sin_cached[:seq_len].to(dtype=x.dtype),
           )

   rope = DeepseekV3RotaryEmbedding(32, 5)
   x = torch.randn(1,4,7,32)
   cos, sin = rope(x)
   print(cos.shape, rope.max_seq_len_cached)
   cos2, sin2 = rope(torch.randn(1,4,3,32))
   print(cos2.shape)
   ```

2. **Seq_Len=None Inference**:
   - x_short = torch.randn(1,2,4,32)  # seq=4
   - cos, sin = rope(x_short, seq_len=None)
   - Verify seq_len inferred as 4, slice from cache.

3. **Dtype Cast Test**:
   - rope = DeepseekV3RotaryEmbedding(16)  # fp32 cache
   - x_fp16 = torch.randn(1,1,5,16, dtype=torch.float16)
   - cos, sin = rope(x_fp16)
   - Print cos.dtype (fp16), cos.device (cpu/gpu from x)

4. **Time Comparison**:
   - Init with max=100
   - Time forward seq=50 (slice, fast)
   - Time forward seq=200 (extend cache, slower first time)
   - Second call seq=200 (slice, fast)

## Quiz

1. If seq_len > cached, what happens?  
   a) Error b) Extend cache c) Use old cache

2. True/False: forward computes cos/sin from scratch every time.

3. Shape of returned cos? (Short: (seq_len, dim))

## Advanced Task

Modify forward to infer seq_len from position_ids if passed (optional arg). Test with position_ids = torch.tensor([0,5,10]), seq_len=None, x any—use max(position_ids)+1 for cache. Discuss for KV-cache in generation (position_ids offset).

Submit code, outputs, and answers.
