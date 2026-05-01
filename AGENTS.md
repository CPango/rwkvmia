# AGENTS.md

## Project role

This repository is for a controlled fine-tuning membership inference attack project on autoregressive language models.

The initial model pair is:

- EleutherAI/gpt-neo-1.3B
- RWKV/rwkv-4-169m-pile



## Directory rules

- Put runnable scripts in scripts/
- Put reusable code in src/
- Put configuration files in configs/
- Put processed data under data/processed/
- Put logs under results/logs/
- Put tables under results/tables/
- Put figures under results/figures/
- Put predictions under results/predictions/
- Do not modify external/ third-party repositories directly

## Safety and execution rules

- Do not run long training jobs without asking first.
- Do not download large models unless explicitly approved.
- Do not run Hugging Face login commands automatically.
- Do not print or store tokens in files.
- Do not commit checkpoints, model weights, caches, raw large datasets, or generated feature files.
- Always create or update scripts first, then explain the next commands the user should run.

## Current expected tasks

Codex should help build:

1. scripts/01_smoke_test_models.py
2. scripts/02_prepare_tofu.py
3. configs/models.yaml
4. configs/paths.yaml
5. basic src/ modules for data loading, model loading, feature extraction, and attack classification

## Experiment order

1. Environment check
2. Model smoke test
3. TOFU loading and author-level split
4. Static loss / perplexity baseline
5. Controlled fine-tuning
6. Drift feature extraction
7. Logistic regression membership classifier
8. RWKV state-drift ablation
