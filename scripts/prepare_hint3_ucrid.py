#!/usr/bin/env python3
"""
Prepare HINT3 v2 subset data for the current UCRID pipeline.

The current project expects JSON files:
  train.json / val.json / test.json
with fields:
  {"text": ..., "intent": ..., "label": ...}

HINT3 provides only train/test CSV files, and OOS examples appear in test as
"NO_NODES_DETECTED". To support the current UCRID validation-time threshold
search, this script splits the original test set into val/test while
preserving the ID/OOS ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import OrderedDict


OOS_NAME = "NO_NODES_DETECTED"


def load_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "text": row["sentence"].strip(),
                    "raw_label": row["label"].strip(),
                }
            )
    return rows


def stratified_binary_split(rows, val_ratio: float, seed: int):
    rng = random.Random(seed)
    id_rows = [r for r in rows if r["raw_label"] != OOS_NAME]
    oos_rows = [r for r in rows if r["raw_label"] == OOS_NAME]
    rng.shuffle(id_rows)
    rng.shuffle(oos_rows)

    id_val_n = max(1, int(round(len(id_rows) * val_ratio))) if id_rows else 0
    oos_val_n = max(1, int(round(len(oos_rows) * val_ratio))) if oos_rows else 0

    val_rows = id_rows[:id_val_n] + oos_rows[:oos_val_n]
    test_rows = id_rows[id_val_n:] + oos_rows[oos_val_n:]
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return val_rows, test_rows


def build_label_map(train_rows):
    intent_names = sorted({r["raw_label"] for r in train_rows if r["raw_label"] != OOS_NAME})
    return OrderedDict((intent, idx) for idx, intent in enumerate(intent_names))


def convert_rows(rows, label_map, oos_label: int):
    converted = []
    for r in rows:
        if r["raw_label"] == OOS_NAME:
            converted.append(
                {
                    "text": r["text"],
                    "intent": "oos",
                    "label": oos_label,
                }
            )
        else:
            converted.append(
                {
                    "text": r["text"],
                    "intent": r["raw_label"],
                    "label": label_map[r["raw_label"]],
                }
            )
    return converted


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["curekart", "sofmattress", "powerplay11"])
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    root = f"/mnt/data3/wzc/llm_oos_detection/dataset/HINT3/dataset/{args.version}"
    train_csv = os.path.join(root, "train", f"{args.subset}_train.csv")
    test_csv = os.path.join(root, "test", f"{args.subset}_test.csv")

    train_rows = load_csv(train_csv)
    raw_test_rows = load_csv(test_csv)
    val_rows, test_rows = stratified_binary_split(raw_test_rows, args.val_ratio, args.seed)

    label_map = build_label_map(train_rows)
    oos_label = len(label_map)

    train_json = convert_rows(train_rows, label_map, oos_label)
    val_json = convert_rows(val_rows, label_map, oos_label)
    test_json = convert_rows(test_rows, label_map, oos_label)

    save_json(os.path.join(args.output_dir, "train.json"), train_json)
    save_json(os.path.join(args.output_dir, "val.json"), val_json)
    save_json(os.path.join(args.output_dir, "test.json"), test_json)
    save_json(
        os.path.join(args.output_dir, "metadata.json"),
        {
            "subset": args.subset,
            "version": args.version,
            "oos_name": OOS_NAME,
            "num_intents": len(label_map),
            "oos_label": oos_label,
            "label_map": label_map,
            "train_size": len(train_json),
            "val_size": len(val_json),
            "test_size": len(test_json),
            "note": "Validation set is split from the original test set to provide OOS-aware threshold search for the current UCRID pipeline.",
        },
    )

    print(f"Prepared HINT3 {args.version} / {args.subset}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  num_intents  : {len(label_map)}")
    print(f"  oos_label    : {oos_label}")
    print(f"  train / val / test = {len(train_json)} / {len(val_json)} / {len(test_json)}")


if __name__ == "__main__":
    main()
