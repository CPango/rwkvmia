# rwkvmia

This project studies membership inference attacks for autoregressive language models.

## Current model strategy

The initial experiment will use public Hugging Face models:

- Transformer baseline: EleutherAI/gpt-neo-1.3B
- RWKV baseline: RWKV/rwkv-4-169m-pile


## Main workflow

1. Build a reproducible project structure.
2. Prepare a conda environment for the remote server.
3. Run smoke tests for GPT-Neo-1.3B and RWKV-4.
4. Prepare TOFU data with author-level split.
5. Run controlled fine-tuning.
6. Extract static and drift features.
7. Train a membership inference classifier.
8. Compare Transformer baseline with RWKV state-drift features.

## Local and remote division

- Local machine: project setup, Git management, documentation, script drafting.
- Remote server: conda environment, model download, GPU smoke test, fine-tuning, feature extraction, evaluation.
