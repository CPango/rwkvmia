"""Small Transformers loading wrappers used by smoke tests and experiments."""

from __future__ import annotations

from typing import Any


def load_tokenizer(model_id: str, trust_remote_code: bool = False, **kwargs: Any):
    """Load a tokenizer and set pad_token to eos_token when possible."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(
    model_id: str,
    *,
    device: str = "auto",
    torch_dtype: Any = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an AutoModelForCausalLM with conservative device handling."""
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if device == "auto":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device in {"cpu", "cuda"}:
        model = model.to(device)
    model.eval()
    return model


def move_batch_to_model_device(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    """Move tensor values in a tokenizer batch to the model's current device."""
    try:
        device = model.device
    except Exception:
        return batch
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
