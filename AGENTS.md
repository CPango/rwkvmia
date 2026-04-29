# AGENTS.md

## Project rules
- Put runnable entry scripts in scripts/
- Put reusable code in src/
- Do not modify external/ third-party repos directly
- Do not commit checkpoints, model weights, caches, or large raw artifacts
- Save logs to results/logs
- Save figures to results/figures
- Save tables to results/tables

## Current workflow
- local machine: repo setup, code drafting, documentation
- remote server: conda env, model download, smoke tests, training, evaluation
