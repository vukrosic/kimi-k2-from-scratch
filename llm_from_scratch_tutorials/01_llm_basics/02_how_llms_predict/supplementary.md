# Supplementary: How LLMs Predict Tokens

## Additional Resources
- Reading: Hugging Face docs on "Generation Strategies" (top-K, temperature).
- Video: [Link to sampling explanation video].
- Tool: Use `transformers` library's `generate` with different `do_sample`, `top_k`, `temperature`.

## Advanced Notes
- Logits vs. Probs: Model outputs logits (un-normalized scores); softmax → probs.
- Nucleus Sampling: Alternative to top-K—sample until cumulative prob > p (e.g., 0.9).
- In DeepSeek/Llama: Combined sampling in `generate()` for chat.

## Glossary
- **Logits**: Raw model outputs (pre-softmax).
- **Softmax**: Converts logits to probs (exp(x)/sum(exp)).
- **Temperature**: Scales logits for sharpness/randomness.

## Further Exploration
- Experiment: In code_snippets.py, set temp=0 (greedy-like). Run 10x—always same?
- Question: Why combine top-K + temp? (Hint: Prevents rare bad tokens while adding variety.)
