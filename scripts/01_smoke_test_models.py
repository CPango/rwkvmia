"""Smoke test configured causal language models on the remote server.

This script intentionally performs only a short tokenizer/model load, one
forward pass, and a tiny deterministic generation. It does not train, save
weights, or write feature files.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.loading import load_causal_lm, load_tokenizer, move_batch_to_model_device
from src.utils.config import load_models_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short model smoke tests.")
    parser.add_argument(
        "--model",
        default="all",
        help="Model key from configs/models.yaml, or 'all'.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device placement strategy for model loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generation length from configs/models.yaml.",
    )
    parser.add_argument(
        "--config",
        default="configs/models.yaml",
        help="Path to the model YAML config.",
    )
    return parser.parse_args()


def torch_dtype_from_name(torch_module: Any, dtype_name: str | None):
    if dtype_name is None:
        return None
    if dtype_name == "float16":
        return torch_module.float16
    if dtype_name == "bfloat16":
        return torch_module.bfloat16
    if dtype_name == "float32":
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype in config: {dtype_name}")


def summarize_state(state: Any) -> None:
    if state is None:
        print("state: None")
        return

    print("state type:", type(state))
    if isinstance(state, (list, tuple)):
        print("state length:", len(state))
        for index, item in enumerate(state[:5]):
            if hasattr(item, "shape"):
                print(f"state[{index}].shape:", tuple(item.shape))
            else:
                print(f"state[{index}] type:", type(item))
        return

    if hasattr(state, "shape"):
        print("state.shape:", tuple(state.shape))
    else:
        print("state value type:", type(state))


def select_model_items(config: dict[str, Any], requested_model: str) -> list[tuple[str, dict[str, Any]]]:
    models = config.get("models", {})
    if requested_model == "all":
        return list(models.items())
    if requested_model not in models:
        available = ", ".join(["all", *sorted(models)])
        raise KeyError(f"Unknown model '{requested_model}'. Available: {available}")
    return [(requested_model, models[requested_model])]


def run_forward_and_generation(
    *,
    key: str,
    model_cfg: dict[str, Any],
    defaults: dict[str, Any],
    device: str,
    max_new_tokens: int,
) -> bool:
    import torch

    model_id = model_cfg["model_id"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", defaults.get("trust_remote_code", False)))
    prompt = str(model_cfg.get("smoke_prompt", defaults.get("smoke_prompt", "Hello.")))

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    use_gpu_dtype = (device == "cuda") or (device == "auto" and torch.cuda.is_available())
    dtype_name = defaults.get("torch_dtype_gpu" if use_gpu_dtype else "torch_dtype_cpu")
    torch_dtype = torch_dtype_from_name(torch, dtype_name)
    load_device = "auto" if device == "auto" and torch.cuda.is_available() else device
    if device == "auto" and not torch.cuda.is_available():
        load_device = "cpu"

    print("=" * 100)
    print(f"Testing model key: {key}")
    print(f"Model ID: {model_id}")
    print(f"device: {load_device}")
    print(f"torch_dtype: {dtype_name}")

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code)
    print("Tokenizer loaded.")

    model = load_causal_lm(
        model_id,
        device=load_device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    print("Model loaded.")

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = move_batch_to_model_device(inputs, model)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )

    print("Forward pass succeeded.")
    print("logits.shape:", tuple(outputs.logits.shape))

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        print("hidden_states: None")
    else:
        print("num_hidden_states:", len(hidden_states))
        print("last_hidden_state.shape:", tuple(hidden_states[-1].shape))

    if hasattr(outputs, "state"):
        summarize_state(outputs.state)
    else:
        print("state attribute: not present")

    gen_inputs = tokenizer(prompt, return_tensors="pt")
    gen_inputs = move_batch_to_model_device(gen_inputs, model)
    generated = model.generate(
        **gen_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("generation sample:", decoded)
    print(f"[OK] {key}")
    return True


def main() -> int:
    args = parse_args()

    import torch

    config = load_models_config(args.config)
    defaults = config.get("defaults", {})
    max_new_tokens = args.max_new_tokens or int(defaults.get("max_new_tokens", 20))

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda device name:", torch.cuda.get_device_name(0))

    failures = 0
    for key, model_cfg in select_model_items(config, args.model):
        try:
            run_forward_and_generation(
                key=key,
                model_cfg=model_cfg,
                defaults=defaults,
                device=args.device,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            failures += 1
            print(f"[FAILED] {key}: {exc}")
            traceback.print_exc()

    if failures:
        print(f"Smoke test finished with {failures} failure(s).")
        return 1
    print("Smoke test finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
