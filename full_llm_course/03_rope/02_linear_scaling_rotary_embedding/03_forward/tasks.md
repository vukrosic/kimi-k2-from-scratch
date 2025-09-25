# Lesson 4.2.3: Tasks - RoPE Linear Scaling Forward Method

## Theory Exercises

1. **Inheritance and Transparency**: Why no override in forward? How does scaling "hide" in cache? For s=2, forward pos=5 returns cos(5/2 * ω) = cos(2.5 * ω), vs standard cos(5 * ω) (slower).

2. **Scaling in Attention**: In apply_rotary_pos_emb, scaled cos/sin used same way. For long seq, relative θ_rel = (pos2 / s - pos1 / s) * ω = ((pos2 - pos1)/s) * ω, so short rel dist at long pos has smaller θ than trained short-short (distorted, but approximate).

3. **Dynamic Extension with Scaling**: If cached for short seq with s=4, extend to long: Recompute t /4 for all, old positions same as before (consistent).

4. **Cost Analysis**: Forward O(1), but extend calls cache O(seq * dim), scaled t no extra cost. For s=8, long seq extend more frequent? No, same as standard.

## Code Tasks

1. **Inheritance Test**:
   - Define full linear class (init, scaled _set_cos_sin_cache, inherit forward).
   - rope_std = DeepseekV3RotaryEmbedding(dim=16, max_position_embeddings=4)
   - rope_scale = DeepseekV3LinearScalingRotaryEmbedding(dim=16, scaling_factor=2.0, max_position_embeddings=4)
   - x = torch.randn(1,2,6,16)  # seq=6 >4, both extend
   - cos_std, sin_std = rope_std(x)
   - cos_scale, sin_scale = rope_scale(x)
   - For pos=5, print cos_std[5,0] = cos(5 * inv[0]), cos_scale[5,0] = cos(5/2 * inv[0]) = cos(2.5 * inv[0])
   - Verify cos_scale[5] != cos_std[5], but cos_scale[2] ≈ cos_std[1] (2/2=1)

   ```python
   # Use full class definitions from previous lessons
   # Assume DeepseekV3RotaryEmbedding defined
   # Linear as subclass with scaled cache

   rope_std = DeepseekV3RotaryEmbedding(16, 4)
   rope_scale = DeepseekV3LinearScalingRotaryEmbedding(16, scaling_factor=2.0, max_position_embeddings=4)
   x = torch.randn(1,2,6,16)
   cos_std, _ = rope_std(x)
   cos_scale, _ = rope_scale(x)
   inv = rope_std.inv_freq[0].item()
   print('Standard pos5 cos0:', math.cos(5 * inv))
   print('Scaled pos5 cos0:', math.cos(2.5 * inv))
   print('Scaled pos2 cos0 ≈ standard pos1:', math.isclose(math.cos(2/2 * inv), math.cos(1 * inv)))
   ```

2. **Dynamic Extension Test**:
   - rope_scale._set_cos_sin_cache(3)  # Initial small
   - Call forward seq=5 (extends, t/2 for 0-4)
   - Print rope_scale.max_seq_len_cached (5), cos.shape (5,16)
   - Second forward seq=5 (no extend, slice)

3. **Dtype in Scaled**:
   - rope_scale = ... (fp32 default)
   - x_fp16 = torch.randn(1,1,4,16, dtype=torch.float16)
   - cos, sin = rope_scale(x_fp16)
   - Print cos.dtype (fp16), even if cache recomputed in fp16

4. **Compare Outputs**:
   - For seq=4, compute cos_scale[3, :] vs cos_std[1.5 * something], but since non-integer, compute manual cos((3/2) * inv_freq) for each dim//2, cat even/odd.

## Quiz

1. Does linear scaling override forward?  
   a) Yes b) No, inherits c) Partial

2. True/False: Scaling affects returned cos/sin values but not shape/API.

3. For s=2, angle for pos=4? (Short: 2 * ω)

## Advanced Task

Integrate with attention stub: Create simple attention forward that calls rope(x), uses cos/sin in apply_rotary_pos_emb stub. Test standard vs scaled on seq=10, print QK before/after rotation (scaled should have slower pos effect). Discuss for long seq inference.

Submit code, outputs, and answers.
