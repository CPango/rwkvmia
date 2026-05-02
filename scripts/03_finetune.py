#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
03_finetune.py

功能：
1. 读取 configs/*.yaml 配置文件；
2. 从 data/processed/tofu/hfds/{train,validation} 加载 TOFU HF Dataset；
3. 加载 GPT-Neo 或 RWKV Causal LM；
4. 按 input_text + target_text 构造 causal LM 训练样本；
5. 使用 Hugging Face Trainer 进行轻量微调；
6. 保存 final checkpoint、tokenizer、训练日志和 config 副本。

运行示例：
python scripts/03_finetune.py --config configs/gptneo_tofu.yaml
python scripts/03_finetune.py --config configs/rwkv_tofu.yaml
"""

import argparse
import copy
import inspect
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config, e.g. configs/gptneo_tofu.yaml",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to resume training.",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(dtype_name: str):
    if dtype_name is None:
        return None
    name = str(dtype_name).lower()
    if name in ["float16", "fp16", "torch.float16"]:
        return torch.float16
    if name in ["bfloat16", "bf16", "torch.bfloat16"]:
        return torch.bfloat16
    if name in ["float32", "fp32", "torch.float32"]:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_name}")


def get_text_fields(example: Dict[str, Any]) -> Dict[str, str]:
    """
    兼容 02_prepare_tofu.py 的标准字段：
    - input_text
    - target_text
    同时兼容 question / answer 字段。
    """
    if "input_text" in example and "target_text" in example:
        input_text = example["input_text"]
        target_text = example["target_text"]
    elif "question" in example and "answer" in example:
        input_text = f"Q: {example['question']}\nA:"
        target_text = str(example["answer"])
    elif "full_text" in example:
        # 兜底：如果只有 full_text，就把它全部作为训练文本
        input_text = ""
        target_text = str(example["full_text"])
    else:
        raise KeyError(
            "Dataset example must contain either "
            "(input_text, target_text), (question, answer), or full_text."
        )

    input_text = "" if input_text is None else str(input_text)
    target_text = "" if target_text is None else str(target_text)
    return {"input_text": input_text, "target_text": target_text}


@dataclass
class CausalLMCollator:
    tokenizer: Any
    max_length: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        eos = self.tokenizer.eos_token or ""
        pad_id = self.tokenizer.pad_token_id

        for ex in features:
            fields = get_text_fields(ex)
            prompt = fields["input_text"]
            answer = fields["target_text"]

            # prompt 与 answer 分开 tokenize，这样可以把 prompt 部分 label 置为 -100
            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            answer_text = answer + eos
            answer_ids = self.tokenizer(
                answer_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            # 右截断。若样本过长，保留前 max_length 个 token。
            # TOFU QA 通常较短，max_length=256 一般足够。
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        max_len = max(len(x) for x in batch_input_ids)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for input_ids, labels, attention_mask in zip(
            batch_input_ids, batch_labels, batch_attention_mask
        ):
            pad_len = max_len - len(input_ids)
            padded_input_ids.append(input_ids + [pad_id] * pad_len)
            padded_labels.append(labels + [-100] * pad_len)
            padded_attention_mask.append(attention_mask + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }


def build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    """
    兼容 transformers 不同版本：
    有些版本使用 evaluation_strategy，有些新版本使用 eval_strategy。
    """
    tcfg = cfg["training"]

    base_kwargs = dict(
        output_dir=tcfg["output_dir"],
        logging_dir=tcfg.get("logging_dir", None),
        num_train_epochs=tcfg.get("num_train_epochs", 2),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
        learning_rate=tcfg.get("learning_rate", 2e-5),
        weight_decay=tcfg.get("weight_decay", 0.01),
        warmup_ratio=tcfg.get("warmup_ratio", 0.03),
        fp16=tcfg.get("fp16", False),
        bf16=tcfg.get("bf16", False),
        logging_steps=tcfg.get("logging_steps", 10),
        eval_steps=tcfg.get("eval_steps", 100),
        save_steps=tcfg.get("save_steps", 100),
        save_total_limit=tcfg.get("save_total_limit", 2),
        report_to=tcfg.get("report_to", "none"),
        remove_unused_columns=False,
        save_strategy="steps",
        logging_strategy="steps",
        do_train=True,
        do_eval=True,
    )

    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        base_kwargs["eval_strategy"] = "steps"
    else:
        base_kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**base_kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    os.makedirs(out_cfg["final_model_dir"], exist_ok=True)

    # 保存本次实际使用的 config，方便复现实验
    save_yaml(cfg, out_cfg["config_copy_path"])

    print("=" * 80)
    print("[INFO] Loading datasets...")
    print(f"[INFO] train: {data_cfg['hfds_train_dir']}")
    print(f"[INFO] validation: {data_cfg['hfds_validation_dir']}")

    train_dataset = load_from_disk(data_cfg["hfds_train_dir"])
    eval_dataset = load_from_disk(data_cfg["hfds_validation_dir"])

    if len(train_dataset) == 0:
        raise ValueError(f"Train dataset is empty: {data_cfg['hfds_train_dir']}")

    if len(eval_dataset) == 0:
        raise ValueError(f"Validation dataset is empty: {data_cfg['hfds_validation_dir']}")

    print(f"[INFO] train size: {len(train_dataset)}")
    print(f"[INFO] validation size: {len(eval_dataset)}")
    print(f"[INFO] train columns: {train_dataset.column_names}")

    print("=" * 80)
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name_or_path"],
        use_fast=model_cfg.get("use_fast_tokenizer", True),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    if tokenizer.pad_token is None:
        # GPT-Neo / RWKV 这类 causal LM 常常没有 pad token
        tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] tokenizer.pad_token was None. Set pad_token = eos_token.")

    print("=" * 80)
    print("[INFO] Loading model...")
    torch_dtype = resolve_torch_dtype(model_cfg.get("torch_dtype", "float16"))

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # 确保 pad_token_id 写入 config，避免 Trainer 中出现 warning 或 loss 异常
    model.config.pad_token_id = tokenizer.pad_token_id

    # 训练时不需要 cache，且 use_cache=True 可能和 gradient checkpointing 冲突
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if train_cfg.get("gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
            print("[INFO] gradient_checkpointing enabled.")
        except Exception as e:
            print(f"[WARN] gradient_checkpointing_enable failed: {e}")

    data_collator = CausalLMCollator(
        tokenizer=tokenizer,
        max_length=int(data_cfg.get("max_length", 256)),
    )

    training_args = build_training_args(cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("=" * 80)
    print("[INFO] Start training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("=" * 80)
    print("[INFO] Saving final model...")
    trainer.save_model(out_cfg["final_model_dir"])
    tokenizer.save_pretrained(out_cfg["final_model_dir"])

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics_path = os.path.join(train_cfg["output_dir"], "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[DONE] Fine-tuning finished.")
    print(f"[DONE] final model saved to: {out_cfg['final_model_dir']}")
    print(f"[DONE] metrics saved to: {metrics_path}")
    print(f"[DONE] config copied to: {out_cfg['config_copy_path']}")


if __name__ == "__main__":
    main()