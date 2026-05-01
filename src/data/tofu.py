"""Minimal TOFU preprocessing helpers for membership-inference setup."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


AUTHOR_FIELD_CANDIDATES = (
    "author_id",
    "author",
    "person_id",
    "name",
    "subject",
    "profile_id",
)


def load_tofu_dataset(
    dataset_name: str = "locuslab/TOFU",
    subset: str | None = "full",
    split: str = "train",
):
    """Load TOFU with datasets.load_dataset."""
    from datasets import load_dataset

    if subset:
        return load_dataset(dataset_name, subset, split=split)
    return load_dataset(dataset_name, split=split)


def infer_author_id(row: dict[str, Any]) -> str:
    """Infer a stable author identifier from known TOFU-like fields."""
    for field in AUTHOR_FIELD_CANDIDATES:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)

    biography = row.get("biography") or row.get("bio")
    if isinstance(biography, str) and biography.strip():
        digest = hashlib.sha1(biography.strip().encode("utf-8")).hexdigest()[:16]
        return f"bio_{digest}"

    raise ValueError(
        "Could not infer an author id from this TOFU row. "
        f"Available fields: {sorted(row.keys())}"
    )


def normalize_tofu_row(
    row: dict[str, Any],
    index: int,
    *,
    fallback_author_id: str | None = None,
) -> dict[str, Any]:
    """Convert a raw TOFU row into a compact JSONL record."""
    question = row.get("question") or row.get("prompt")
    answer = row.get("answer") or row.get("response") or row.get("completion")
    if not question or not answer:
        raise ValueError(
            "Expected TOFU row to contain question/prompt and answer/response fields. "
            f"Available fields: {sorted(row.keys())}"
        )

    try:
        author_id = infer_author_id(row)
    except ValueError:
        if fallback_author_id is None:
            raise
        author_id = fallback_author_id
    prompt = f"Question: {str(question).strip()}\nAnswer:"
    return {
        "example_id": str(row.get("id", index)),
        "author_id": author_id,
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "prompt": prompt,
        "text": f"{prompt} {str(answer).strip()}",
    }


def normalize_tofu_records(
    rows: Iterable[dict[str, Any]],
    *,
    records_per_author_fallback: int | None = 20,
) -> list[dict[str, Any]]:
    """Normalize all TOFU rows and attach stable per-row indexes."""
    records = []
    for index, row in enumerate(rows):
        fallback_author_id = None
        if records_per_author_fallback:
            fallback_author_id = f"ordered_author_{index // records_per_author_fallback:04d}"
        records.append(
            normalize_tofu_row(
                dict(row),
                index,
                fallback_author_id=fallback_author_id,
            )
        )
    return records


def split_by_author(
    records: Sequence[dict[str, Any]],
    *,
    member_fraction: float = 0.5,
    eval_fraction: float = 0.2,
    seed: int = 13,
) -> dict[str, list[dict[str, Any]]]:
    """Split records into member, nonmember, and eval partitions by author id."""
    if not 0.0 < member_fraction < 1.0:
        raise ValueError("member_fraction must be between 0 and 1.")
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("eval_fraction must be at least 0 and below 1.")

    authors = sorted({record["author_id"] for record in records})
    if len(authors) < 2:
        raise ValueError("Need at least two inferred authors for an author-level split.")

    rng = random.Random(seed)
    rng.shuffle(authors)

    eval_count = max(1, int(round(len(authors) * eval_fraction))) if eval_fraction > 0 else 0
    eval_authors = set(authors[:eval_count])
    remaining_authors = authors[eval_count:]
    if len(remaining_authors) < 2:
        raise ValueError("Not enough non-eval authors to create member and nonmember splits.")

    member_count = max(1, int(round(len(remaining_authors) * member_fraction)))
    member_count = min(member_count, len(remaining_authors) - 1)
    member_authors = set(remaining_authors[:member_count])
    nonmember_authors = set(remaining_authors[member_count:])

    return {
        "train_members": [record for record in records if record["author_id"] in member_authors],
        "train_nonmembers": [record for record in records if record["author_id"] in nonmember_authors],
        "eval": [record for record in records if record["author_id"] in eval_authors],
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL and return the number of rows written."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
