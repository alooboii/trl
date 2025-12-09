# Alignment training scripts (DPO, PPO, GRPO)

This folder provides minimal, config-driven scripts to fine-tune **SmolLM2-135M-SFT-Only** on preference data (e.g., ORCA DPO pairs) using DPO, PPO (sparse and dense rewards), and GRPO. Outputs are written to the configured `output_dir` and include checkpoints plus JSONL logs.

## Quick start (Kaggle-friendly)
1. Clone the repo into your Kaggle working directory and install requirements:
   ```bash
   pip install -e .
   pip install bitsandbytes accelerate datasets transformers peft
   ```
2. Run one of the provided configs. Example (DPO):
   ```bash
   python examples/alignment/run_alignment.py --method dpo --config examples/alignment/configs/dpo.json
   ```
   PPO sparse rewards:
   ```bash
   python examples/alignment/run_alignment.py --method ppo --config examples/alignment/configs/ppo_sparse.json
   ```
   PPO dense rewards:
   ```bash
   python examples/alignment/run_alignment.py --method ppo --config examples/alignment/configs/ppo_dense.json
   ```
   GRPO (reuses reward model saved by PPO if present):
   ```bash
   python examples/alignment/run_alignment.py --method grpo --config examples/alignment/configs/grpo.json
   ```

## Configuration
Each JSON config controls:
- **model_name_or_path**: defaults to `HuggingFaceH4/smolLM2-135M-SFT-Only`.
- **dataset_name / dataset_split**: HF dataset and slice (defaults target the ORCA DPO pairs sample `mlabonne/Orca-Mini-DPO`).
- **prompt_column / chosen_column / rejected_column**: dataset fields.
- **max_prompt_length / max_response_length**: tokenizer truncation bounds.
- **output_dir**: where checkpoints, logs (`log_history.jsonl`, `ppo_metrics.jsonl`), and `evaluation.json` are saved.
- **load_in_8bit**: toggles 8-bit quantization via bitsandbytes.
- **verbosity_word_limit**: optional cap for compliance testing in verbosity analysis.
- Method-specific knobs are included in each config (DPO `beta`, PPO `target_kl`, dense vs sparse rewards, GRPO `group_size`, etc.).

You can modify the JSON configs directly to change dataset size (e.g., `train[:1000]`), learning rates, or LoRA parameters. The scripts automatically create the `outputs/` directory tree.

## Logging and evaluation artifacts
- **Training traces**: `log_history.jsonl` (Hugging Face trainer logs) and, for PPO, `ppo_metrics.jsonl` (step-wise KL/advantage stats).
- **Checkpoints**: saved under `output_dir` (and intermediate `ppo_step_*` folders).
- **Evaluation**: after training, the script produces `evaluation.json` with perplexity on the selected dataset slice and verbosity statistics (mean/median/std token counts and optional compliance rate when `verbosity_word_limit` is set). Prompts come from `examples/alignment/prompts.txt` by default.

## Notes on experiments
- The scripts use 8-bit loading and LoRA-ready configs to fit within common Kaggle GPU limits.
- Reward hacking investigations can be performed by editing `prompts.txt` to include perturbed/hacky prompts; rewards and outputs are logged so you can compare across DPO/PPO/GRPO runs.
- To track catastrophic forgetting, monitor perplexity in `evaluation.json` and KL metrics inside `ppo_metrics.jsonl` or `log_history.jsonl`.

