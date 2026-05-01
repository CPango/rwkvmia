"""Prepare a minimal author-level TOFU split for remote experiments.

This script downloads/loads the small TOFU dataset through Hugging Face
datasets on the remote server, normalizes examples, and writes JSONL files
under data/processed/tofu. It does not train models or create feature caches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tofu import load_tofu_dataset, normalize_tofu_records, split_by_author, write_jsonl
from src.utils.config import load_paths_config, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare minimal TOFU JSONL splits.")
    parser.add_argument("--dataset", default="locuslab/TOFU", help="Hugging Face dataset name.")
    parser.add_argument("--subset", default="full", help="TOFU subset/config name.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument("--output-dir", default=None, help="Override processed TOFU output directory.")
    parser.add_argument("--member-fraction", type=float, default=0.5)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument(
        "--records-per-author-fallback",
        type=int,
        default=20,
        help="Fallback for TOFU subsets without an explicit author field.",
    )
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def output_dir_from_config(override: str | None) -> Path:
    if override:
        return resolve_repo_path(override)
    paths = load_paths_config()
    configured = paths.get("data", {}).get("processed_tofu_dir", "data/processed/tofu")
    return resolve_repo_path(configured)


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    output_dir = output_dir_from_config(args.output_dir)

    dataset = load_tofu_dataset(args.dataset, subset=args.subset, split=args.split)
    records = normalize_tofu_records(
        dataset,
        records_per_author_fallback=args.records_per_author_fallback,
    )
    partitions = split_by_author(
        records,
        member_fraction=args.member_fraction,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )

    counts = {
        name: write_jsonl(output_dir / f"{name}.jsonl", partition)
        for name, partition in partitions.items()
    }
    author_counts = {
        name: len({record["author_id"] for record in partition})
        for name, partition in partitions.items()
    }
    metadata = {
        "dataset": args.dataset,
        "subset": args.subset,
        "source_split": args.split,
        "seed": args.seed,
        "member_fraction": args.member_fraction,
        "eval_fraction": args.eval_fraction,
        "records_per_author_fallback": args.records_per_author_fallback,
        "total_records": len(records),
        "total_authors": len({record["author_id"] for record in records}),
        "record_counts": counts,
        "author_counts": author_counts,
        "outputs": {name: str(output_dir / f"{name}.jsonl") for name in partitions},
    }
    write_metadata(output_dir / "metadata.json", metadata)

    print(f"Wrote TOFU processed files to: {output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
