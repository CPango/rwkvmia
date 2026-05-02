import os
import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets

# ------------------------------------------------------
# 请根据你的目录结构调整
RAW_DATA_DIR = Path("data/raw")
OUTPUT_DATA_DIR = Path("data/processed/tofu")
MODEL_NAME_OR_PATH = "EleutherAI/gpt-neo-1.3B"  # 同 smoke test
SPLIT_RATIOS = (0.7, 0.15, 0.15)                # train/val/test
# ------------------------------------------------------

def load_user_texts(raw_dir: Path):
    """
    遍历 raw data 目录，按 user_id 收集所有文本
    -- 假定结构是 raw_dir/<user_id>/*.txt
    """
    user_texts = defaultdict(list)
    for user_dir in raw_dir.iterdir():
        if user_dir.is_dir():
            user_id = user_dir.name
            for f in user_dir.glob("*.txt"):
                text = f.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    user_texts[user_id].append(text.strip())
    return user_texts


def split_users(user_ids, ratios):
    random.shuffle(user_ids)
    n = len(user_ids)
    train_end = int(ratios[0] * n)
    val_end = train_end + int(ratios[1] * n)
    return (
        user_ids[:train_end],
        user_ids[train_end:val_end],
        user_ids[val_end:],
    )


def prepare_tokenized_dataset(tokenizer, user_texts):
    """
    构造 tokenized 数据
    """
    data = []
    for user_id, text_list in tqdm(user_texts.items()):
        # 合并成一个长字符串
        big_text = "\n".join(text_list)
        enc = tokenizer(
            big_text,
            truncation=False,
            return_attention_mask=True,
        )
        # 通常 causal LM 训练 labels 与 inputs 相同
        enc["labels"] = enc["input_ids"].copy()
        enc["user_id"] = [user_id] * len(enc["input_ids"])
        data.append(enc)
    return data


def main():
    # 1. 加载用户文本
    user_texts = load_user_texts(RAW_DATA_DIR)
    print(f"user count: {len(user_texts)}")

    if len(user_texts) < 1:
        raise RuntimeError("没有检测到任何原始用户数据")

    # 2. 划分用户 split
    user_ids = list(user_texts.keys())
    train_ids, val_ids, test_ids = split_users(user_ids, SPLIT_RATIOS)

    print("splits:", len(train_ids), len(val_ids), len(test_ids))

    # 3. 按 split 构造 text_list
    split_mapping = {
        "train": {uid: user_texts[uid] for uid in train_ids},
        "validation": {uid: user_texts[uid] for uid in val_ids},
        "test": {uid: user_texts[uid] for uid in test_ids},
    }

    # 4. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    processed = {}
    for split, mapped in split_mapping.items():
        print(f"processing split: {split} ...")
        data = prepare_tokenized_dataset(tokenizer, mapped)

        # 5. 转成 HF Dataset
        ds = datasets.Dataset.from_list(data)
        processed[split] = ds

        # 6. 保存到磁盘
        outdir = OUTPUT_DATA_DIR / split
        outdir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(outdir))
        print(f"saved split {split} to {outdir}")

    print("TOFU dataset creation finished")


if __name__ == "__main__":
    main()