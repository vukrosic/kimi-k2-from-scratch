# Supplementary: Introduction to LLMs

## Additional Resources
- Video Inspiration: [Link to original transcript video] – Watch for visual aids on LLM pipeline.
- Reading: "Attention Is All You Need" paper (Transformer basics).
- Tools: Install PyTorch (`pip install torch`) for later code.

## Advanced Notes
- LLM Scale: Llama-3 has 405B params; we focus on architecture, not size.
- Variants: Some LLMs (e.g., DeepSeek) add MoE for efficiency—covered later.
- Common Misconception: LLMs "understand" like humans; they statistically predict, no true comprehension.

## Glossary
- **Token**: Unit of text (word/subword/char).
- **Vocab**: Set of all possible tokens (~50K-150K).
- **Autoregressive**: Predict sequentially, using past outputs.

## Further Exploration
- Experiment: Use Hugging Face's `pipeline("text-generation")` with Llama model to see predictions.
- Question: How does LLM handle non-English? (Hint: Multi-lingual tokens in vocab.)
