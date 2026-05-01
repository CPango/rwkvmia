# Prompt for Codex

You are working inside the rwkvmia repository.

Please help me complete the next project construction stage.

Important context:

- The initial models are:
  - EleutherAI/gpt-neo-1.3B
  - RWKV/rwkv-4-169m-pile
- The local machine is only for project setup and code drafting.
- The remote server will run conda, download models, and execute GPU experiments.
- Do not run Hugging Face login.
- Do not download large models.
- Do not start long training jobs.
- Do not modify external/ third-party repositories directly.

Please do the following:

1. Inspect the current repository structure.
2. Check README.md, AGENTS.md, .gitignore, and environment.yml.
3. Create or update configs/paths.yaml.
4. Create or update configs/models.yaml.
5. Create or update scripts/01_smoke_test_models.py for:
   - EleutherAI/gpt-neo-1.3B
   - RWKV/rwkv-4-169m-pile
6. Create scripts/02_prepare_tofu.py as a first minimal TOFU preprocessing script.
7. Create src/data/, src/models/, src/features/, src/attack/, and src/utils/ modules only if needed.
8. Add clear comments explaining what each script does.
9. Do not execute long-running commands.
10. After editing, summarize:
    - which files changed
    - what each file does
    - what commands I should run on the server next

The immediate goal is to make the repository ready for remote server smoke testing, not to run the full experiment yet.
