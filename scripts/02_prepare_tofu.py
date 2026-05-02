# -*- coding: utf-8 -*-
"""
02_prepare_tofu.py

功能：
1. 从 Hugging Face 加载 locuslab/TOFU full 数据集；
2. 标准化字段：sample_id / author_id / question / answer / input_text / target_text / full_text；
3. 由于 TOFU full 数据没有显式 author 字段，因此根据 TOFU 官方设定：
   200 个虚构作者，每个作者 20 组 QA，按每 20 条构造一个 author_id；
4. 按 author_id 做作者级 train / validation / test 划分；
5. 生成两种保存格式：
   A) JSONL:
      data/processed/tofu/jsonl/train.jsonl
      data/processed/tofu/jsonl/validation.jsonl
      data/processed/tofu/jsonl/test.jsonl

   B) Hugging Face Dataset saved_to_disk:
      data/processed/tofu/hfds/train/
      data/processed/tofu/hfds/validation/
      data/processed/tofu/hfds/test/

运行：
python scripts/02_prepare_tofu.py
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm
from datasets import load_dataset, Dataset


# ====================================================
# 配置区
# ====================================================

TOFU_DATASET = "locuslab/TOFU"
TOFU_CONFIG = "full"

OUTPUT_DIR = Path("data/processed/tofu")
JSONL_DIR = OUTPUT_DIR / "jsonl"
HFDS_DIR = OUTPUT_DIR / "hfds"

SEED = 42

# TOFU 官方设定：200 个虚构作者，每个作者 20 组问答
AUTHOR_BLOCK_SIZE = 20

# 作者级划分比例
TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

# 是否覆盖旧的 jsonl / hfds 处理结果
OVERWRITE = True


# ====================================================
# 工具函数
# ====================================================

def ensure_dirs() -> None:
    """
    确保输出目录存在。
    如果 OVERWRITE=True，则清理旧的 jsonl 和 hfds，避免旧 Arrow 文件和新数据混在一起。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OVERWRITE:
        if JSONL_DIR.exists():
            shutil.rmtree(JSONL_DIR)
        if HFDS_DIR.exists():
            shutil.rmtree(HFDS_DIR)

    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    HFDS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_record(
    item: Dict,
    source_split: str,
    sample_id: str,
    sample_idx: int
) -> Dict:
    """
    将 TOFU 原始记录标准化为统一格式。

    注意：
    - TOFU full 配置通常只有 question / answer 字段；
    - full 数据按作者连续组织，每个虚构作者对应 20 组 QA；
    - 因此，如果没有原始 author 字段，就按 sample_idx // 20 构造 author_id。
    """
    q = str(item.get("question", "")).strip()
    a = str(item.get("answer", "")).strip()

    raw_author = item.get("author", None)
    if raw_author is not None and str(raw_author).strip():
        author_id = str(raw_author).strip()
    else:
        author_id = f"author_{sample_idx // AUTHOR_BLOCK_SIZE:03d}"

    input_text = f"Q: {q}\nA:"
    target_text = a
    full_text = f"{input_text} {target_text}".strip()

    return {
        "sample_id": sample_id,
        "author_id": author_id,
        "question": q,
        "answer": a,
        "input_text": input_text,
        "target_text": target_text,
        "full_text": full_text,
        "source_split": source_split,
        "phase": None,
        "member_label": None,
    }


def save_jsonl(records: List[Dict], path: Path) -> None:
    """
    保存 JSONL 文件。
    """
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_authors(author_ids: List[str]) -> Tuple[set, set, set]:
    """
    作者级划分，避免同一作者的 QA 同时出现在 train / validation / test 中。
    """
    random.seed(SEED)
    author_ids = list(author_ids)
    random.shuffle(author_ids)

    n = len(author_ids)
    n_train = int(n * TRAIN_RATIO)
    n_validation = int(n * VALIDATION_RATIO)

    train_authors = set(author_ids[:n_train])
    validation_authors = set(author_ids[n_train:n_train + n_validation])
    test_authors = set(author_ids[n_train + n_validation:])

    return train_authors, validation_authors, test_authors


def check_author_overlap(split_records: Dict[str, List[Dict]]) -> None:
    """
    检查不同 phase 之间是否存在 author_id 重叠。
    """
    split_authors = {
        phase: set(rec["author_id"] for rec in recs)
        for phase, recs in split_records.items()
    }

    phases = list(split_authors.keys())
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            p1, p2 = phases[i], phases[j]
            overlap = split_authors[p1] & split_authors[p2]
            if overlap:
                raise ValueError(
                    f"Author overlap detected between {p1} and {p2}: "
                    f"{list(overlap)[:10]}"
                )


def print_split_summary(split_records: Dict[str, List[Dict]]) -> None:
    """
    打印每个 split 的记录数、作者数、标签分布。
    """
    print("\nSummary:")
    for phase, recs in split_records.items():
        authors = set(rec["author_id"] for rec in recs)
        label_count = {}
        for rec in recs:
            label = rec["member_label"]
            label_count[label] = label_count.get(label, 0) + 1

        print(
            f"  {phase}: "
            f"records={len(recs)}, "
            f"authors={len(authors)}, "
            f"label_count={label_count}"
        )


def sanity_check_raw_records(records: List[Dict]) -> None:
    """
    对 TOFU full 的基本结构做检查。
    """
    total_records = len(records)
    if total_records == 0:
        raise ValueError("No records loaded from TOFU dataset.")

    if total_records % AUTHOR_BLOCK_SIZE != 0:
        print(
            f"[WARN] total_records={total_records} is not divisible by "
            f"AUTHOR_BLOCK_SIZE={AUTHOR_BLOCK_SIZE}. "
            f"Please check author_id construction."
        )

    total_authors = len(set(rec["author_id"] for rec in records))
    print(f"[INFO] total normalized records: {total_records}")
    print(f"[INFO] total constructed authors: {total_authors}")

    # 对 TOFU full 来说，预期是 4000 条、200 作者
    if total_records == 4000 and total_authors != 200:
        raise ValueError(
            f"Expected 200 authors for 4000 TOFU records, got {total_authors}. "
            f"Please check author_id construction."
        )


# ====================================================
# 主流程
# ====================================================

def main() -> None:
    print("Ensuring output directories exist ...")
    ensure_dirs()

    print("Loading TOFU dataset ...")
    ds_dict = load_dataset(TOFU_DATASET, TOFU_CONFIG)

    all_raw_records: List[Dict] = []
    global_sample_idx = 0

    print("Normalizing records ...")
    for split_name, ds in ds_dict.items():
        print(f"  processing original split: {split_name}, len={len(ds)}")

        for item in tqdm(ds):
            sample_id = f"{split_name}_{global_sample_idx:06d}"
            rec = normalize_record(
                item=item,
                source_split=split_name,
                sample_id=sample_id,
                sample_idx=global_sample_idx,
            )
            all_raw_records.append(rec)
            global_sample_idx += 1

    sanity_check_raw_records(all_raw_records)

    # 按 author_id 聚合记录
    author_to_records = defaultdict(list)
    for rec in all_raw_records:
        author_to_records[rec["author_id"]].append(rec)

    all_authors = sorted(author_to_records.keys())
    print(f"[INFO] total authors for split: {len(all_authors)}")

    # 检查每个作者是否都是 20 条
    author_sizes = {a: len(rs) for a, rs in author_to_records.items()}
    bad_authors = {a: n for a, n in author_sizes.items() if n != AUTHOR_BLOCK_SIZE}
    if bad_authors:
        print("[WARN] Some authors do not have exactly 20 records:")
        print(dict(list(bad_authors.items())[:10]))
    else:
        print(f"[INFO] every author has {AUTHOR_BLOCK_SIZE} records.")

    train_authors, validation_authors, test_authors = split_authors(all_authors)

    print(f"[INFO] train authors: {len(train_authors)}")
    print(f"[INFO] validation authors: {len(validation_authors)}")
    print(f"[INFO] test authors: {len(test_authors)}")

    split_records = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for author_id, records in author_to_records.items():
        if author_id in train_authors:
            phase = "train"
            member_label = 1
        elif author_id in validation_authors:
            phase = "validation"
            member_label = 0
        elif author_id in test_authors:
            phase = "test"
            member_label = 0
        else:
            raise RuntimeError(f"Author not assigned to any split: {author_id}")

        for rec in records:
            new_rec = dict(rec)
            new_rec["phase"] = phase
            new_rec["member_label"] = member_label
            split_records[phase].append(new_rec)

    check_author_overlap(split_records)

    # 为了输出稳定，按 sample_id 排序
    for phase in split_records:
        split_records[phase] = sorted(split_records[phase], key=lambda x: x["sample_id"])

    print("Saving JSONL files ...")
    for phase, recs in split_records.items():
        out_file = JSONL_DIR / f"{phase}.jsonl"
        save_jsonl(recs, out_file)
        print(f"[OK] {phase}.jsonl -> {out_file} (count={len(recs)})")

    print("Saving HuggingFace Dataset format ...")
    for phase, recs in split_records.items():
        ds = Dataset.from_list(recs)
        hf_path = HFDS_DIR / phase
        ds.save_to_disk(str(hf_path))
        print(f"[OK] HFDS '{phase}' -> {hf_path} (count={len(ds)})")

    # 保存划分元信息，方便论文和后续复现实验说明
    split_meta = {
        "dataset": TOFU_DATASET,
        "config": TOFU_CONFIG,
        "seed": SEED,
        "author_block_size": AUTHOR_BLOCK_SIZE,
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "num_records": len(all_raw_records),
        "num_authors": len(all_authors),
        "num_train_authors": len(train_authors),
        "num_validation_authors": len(validation_authors),
        "num_test_authors": len(test_authors),
        "member_label_definition": {
            "1": "samples used for target model fine-tuning",
            "0": "samples not used for target model fine-tuning",
        },
    }

    meta_path = OUTPUT_DIR / "split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] split meta -> {meta_path}")

    print_split_summary(split_records)

    print("\nALL DONE.")


if __name__ == "__main__":
    main()