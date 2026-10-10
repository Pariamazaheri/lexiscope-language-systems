"""Hash-disjoint review splits and non-overlapping language-model blocks."""

import hashlib
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def clean_text(value):
    value = html.unescape(str(value))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_splits(path):
    frame = pd.read_csv(path)
    if set(["text", "label"]) - set(frame) or not frame.label.isin([0, 1]).all():
        raise ValueError("Expected text and binary labels")
    if frame.text.isna().any():
        raise ValueError("Null review text")
    frame["review_id"] = np.arange(len(frame))
    frame["text"] = frame.text.map(clean_text)
    frame["text_sha256"] = frame.text.map(digest)
    if frame.text.eq("").any():
        raise ValueError("Empty review")
    if frame.groupby("text_sha256").label.nunique().max() > 1:
        raise ValueError("Conflicting labels on duplicate reviews")
    count = len(frame)
    frame = frame.drop_duplicates("text_sha256")
    train, rest = train_test_split(
        frame, test_size=0.3, stratify=frame.label, random_state=42
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest.label, random_state=43
    )
    parts = {"train": train, "validation": val, "test": test}
    manifest = pd.concat(
        [
            v[["review_id", "text_sha256", "label"]].assign(split=k)
            for k, v in parts.items()
        ]
    )
    return (
        parts,
        manifest,
        {
            "source_rows": count,
            "duplicates_removed": count - len(frame),
            "split_rows": {k: len(v) for k, v in parts.items()},
        },
    )


def encode_reviews(frame, tokenizer, length=128):
    encoded = tokenizer(
        frame.text.tolist(),
        max_length=length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return TensorDataset(
        encoded.input_ids,
        encoded.attention_mask,
        torch.tensor(frame.label.to_numpy(), dtype=torch.long),
    )


def language_blocks(path, tokenizer, length=64, train_limit=32768):
    text = Path(path).read_text(encoding="utf-8-sig")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    a, b = int(len(tokens) * 0.7), int(len(tokens) * 0.85)
    ranges = {
        "train": (0, min(a, train_limit)),
        "validation": (a, min(b, a + 4096)),
        "test": (b, min(len(tokens), b + 4096)),
    }
    datasets, rows = {}, []
    for split, (start, end) in ranges.items():
        x, y = [], []
        # Step length+1 makes even the boundary targets disjoint across blocks.
        for offset in range(start, end - length, length + 1):
            block = tokens[offset : offset + length + 1]
            x.append(block[:-1])
            y.append(block[1:])
            rows.append(
                {
                    "split": split,
                    "start_token": offset,
                    "end_token_exclusive": offset + length + 1,
                    "block_sha256": digest(",".join(map(str, block))),
                }
            )
        if not x:
            raise ValueError("Corpus too short for three independent splits")
        datasets[split] = TensorDataset(torch.tensor(x), torch.tensor(y))
    return (
        datasets,
        pd.DataFrame(rows),
        {"total_source_tokens": len(tokens), "token_ranges": ranges},
    )
