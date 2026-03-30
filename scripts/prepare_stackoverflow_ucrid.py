#!/usr/bin/env python3
"""
Prepare the jacoxu/StackOverflow short-text dataset for the UCRID pipeline.

Design:
- Original dataset has 20 labels and 20,000 titles.
- We keep a configurable subset of labels as in-domain (ID).
- Remaining labels are treated as OOS and only appear in val/test.
- Output format matches the current UCRID JSON loader:
  {"text": ..., "intent": ..., "label": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict


LABEL_NAMES = {
    1: "wordpress",
    2: "oracle",
    3: "svn",
    4: "apache",
    5: "excel",
    6: "matlab",
    7: "visual-studio",
    8: "cocoa",
    9: "osx",
    10: "bash",
    11: "spring",
    12: "hibernate",
    13: "scala",
    14: "sharepoint",
    15: "ajax",
    16: "qt",
    17: "drupal",
    18: "linq",
    19: "haskell",
    20: "magento",
}


def read_raw_dataset(title_path: str, label_path: str):
    with open(title_path, "r", encoding="utf-8") as ft:
        titles = [line.strip() for line in ft if line.strip()]
    with open(label_path, "r", encoding="utf-8") as fl:
        labels = [int(line.strip()) for line in fl if line.strip()]

    if len(titles) != len(labels):
        raise ValueError(f"Mismatched sizes: {len(titles)} titles vs {len(labels)} labels")

    rows = []
    for text, raw_label in zip(titles, labels):
        rows.append(
            {
                "text": text,
                "raw_label": raw_label,
                "raw_intent": LABEL_NAMES[raw_label],
            }
        )
    return rows


def stratified_split(rows, train_ratio: float, val_ratio: float, seed: int):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["raw_label"]].append(row)

    train, val, test = [], [], []
    for label, items in grouped.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train >= n:
            n_train = n - 2
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def convert_rows(rows, id_labels, id_label_to_idx, oos_label):
    converted = []
    for row in rows:
        raw_label = row["raw_label"]
        if raw_label in id_labels:
            converted.append(
                {
                    "text": row["text"],
                    "intent": LABEL_NAMES[raw_label],
                    "label": id_label_to_idx[raw_label],
                }
            )
        else:
            converted.append(
                {
                    "text": row["text"],
                    "intent": "oos",
                    "label": oos_label,
                }
            )
    return converted


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/mnt/data3/wzc/llm_oos_detection/dataset/StackOverflow_raw/rawText",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/data3/wzc/llm_oos_detection/dataset/stackoverflow_ucrid",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_id_labels", type=int, default=15)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    rows = read_raw_dataset(
        os.path.join(args.input_dir, "title_StackOverflow.txt"),
        os.path.join(args.input_dir, "label_StackOverflow.txt"),
    )

    all_labels = sorted(LABEL_NAMES.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_labels)
    id_labels = sorted(all_labels[: args.num_id_labels])
    oos_labels = sorted(all_labels[args.num_id_labels :])
    oos_label = len(id_labels)
    id_label_to_idx = {raw_label: idx for idx, raw_label in enumerate(id_labels)}

    train_raw, val_raw, test_raw = stratified_split(
        rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train = [r for r in train_raw if r["raw_label"] in id_labels]
    val = convert_rows(val_raw, set(id_labels), id_label_to_idx, oos_label)
    test = convert_rows(test_raw, set(id_labels), id_label_to_idx, oos_label)

    save_json(os.path.join(args.output_dir, "train.json"), convert_rows(train, set(id_labels), id_label_to_idx, oos_label))
    save_json(os.path.join(args.output_dir, "val.json"), val)
    save_json(os.path.join(args.output_dir, "test.json"), test)
    save_json(
        os.path.join(args.output_dir, "metadata.json"),
        {
            "source": "jacoxu/StackOverflow",
            "seed": args.seed,
            "num_id_labels": len(id_labels),
            "num_oos_labels": len(oos_labels),
            "id_labels_raw": id_labels,
            "oos_labels_raw": oos_labels,
            "id_label_names": [LABEL_NAMES[x] for x in id_labels],
            "oos_label_names": [LABEL_NAMES[x] for x in oos_labels],
            "oos_label": oos_label,
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        },
    )

    print("Prepared StackOverflow for UCRID")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  id labels     : {id_labels} -> {[LABEL_NAMES[x] for x in id_labels]}")
    print(f"  oos labels    : {oos_labels} -> {[LABEL_NAMES[x] for x in oos_labels]}")
    print(f"  oos label idx : {oos_label}")
    print(f"  train / val / test = {len(train)} / {len(val)} / {len(test)}")


if __name__ == "__main__":
    main()
