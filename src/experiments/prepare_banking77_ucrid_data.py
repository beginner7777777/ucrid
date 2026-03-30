"""
Prepare Banking77 data for UCRID pipeline.

Outputs JSON files compatible with existing CLINC loader:
  - train.json (ID only)
  - val.json   (ID + CLINC OOS)
  - test.json  (ID + CLINC OOS)

Data source priority:
  1) Local CSVs (offline): --banking_train_csv / --banking_test_csv
  2) HuggingFace datasets (online fallback): PolyAI/banking77
"""

import argparse
import json
import os
import random
from typing import List, Tuple

import pandas as pd


DEFAULT_CLINC_DIR = "/mnt/data3/wzc/llm_oos_detection/dataset/clinc150/data"
DEFAULT_OUT_DIR = "/mnt/data3/wzc/llm_oos_detection/dataset/banking77_ucrid"


def load_label_names_from_dataset_infos(dataset_infos_path: str) -> List[str]:
    with open(dataset_infos_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    names = payload["default"]["features"]["label"]["names"]
    return list(names)


def load_banking_from_local_csv(
    train_csv: str,
    test_csv: str,
    label_names: List[str],
    val_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict], List[dict]]:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # PolyAI raw CSV columns are typically: text, category
    if "text" not in train_df.columns:
        raise ValueError(f"`text` column not found in {train_csv}")
    if "category" not in train_df.columns:
        raise ValueError(f"`category` column not found in {train_csv}")
    if "text" not in test_df.columns:
        raise ValueError(f"`text` column not found in {test_csv}")
    if "category" not in test_df.columns:
        raise ValueError(f"`category` column not found in {test_csv}")

    intent_to_id = {name: idx for idx, name in enumerate(label_names)}

    train_rows = []
    for _, row in train_df.iterrows():
        intent = str(row["category"])
        if intent not in intent_to_id:
            raise ValueError(f"Unknown intent `{intent}` in train CSV.")
        train_rows.append(
            {"text": str(row["text"]), "intent": intent, "label": intent_to_id[intent], "is_oos": False}
        )

    test_rows = []
    for _, row in test_df.iterrows():
        intent = str(row["category"])
        if intent not in intent_to_id:
            raise ValueError(f"Unknown intent `{intent}` in test CSV.")
        test_rows.append(
            {"text": str(row["text"]), "intent": intent, "label": intent_to_id[intent], "is_oos": False}
        )

    # Build validation split from train (stratified per label).
    rng = random.Random(seed)
    grouped = {}
    for sample in train_rows:
        grouped.setdefault(sample["label"], []).append(sample)

    new_train, val = [], []
    for _, samples in grouped.items():
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))
        val.extend(samples[:n_val])
        new_train.extend(samples[n_val:])

    rng.shuffle(new_train)
    rng.shuffle(val)
    rng.shuffle(test_rows)
    return new_train, val, test_rows


def load_banking_from_hf(seed: int) -> Tuple[List[dict], List[dict], List[dict], List[str]]:
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77")
    label_names = ds["train"].features["label"].names

    def convert(split_name: str) -> List[dict]:
        rows = []
        for ex in ds[split_name]:
            lbl = int(ex["label"])
            rows.append(
                {"text": ex["text"], "intent": label_names[lbl], "label": lbl, "is_oos": False}
            )
        return rows

    train = convert("train")
    val = convert("validation")
    test = convert("test")
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test, list(label_names)


def load_clinc_oos_texts(parquet_path: str, seed: int, limit: int) -> List[str]:
    df = pd.read_parquet(parquet_path)
    oos = df[df["label"].isna()]
    texts = oos["utterance"].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--clinc_dir", default=DEFAULT_CLINC_DIR)
    parser.add_argument("--banking_train_csv", default=None)
    parser.add_argument("--banking_test_csv", default=None)
    parser.add_argument(
        "--dataset_infos",
        default="/mnt/data3/wzc/llm_oos_detection/dataset/banking77/dataset_infos.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_oos_count", type=int, default=100)
    parser.add_argument("--test_oos_count", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    use_local_csv = args.banking_train_csv and args.banking_test_csv
    if use_local_csv:
        label_names = load_label_names_from_dataset_infos(args.dataset_infos)
        train_id, val_id, test_id = load_banking_from_local_csv(
            args.banking_train_csv,
            args.banking_test_csv,
            label_names,
            args.val_ratio,
            args.seed,
        )
    else:
        train_id, val_id, test_id, label_names = load_banking_from_hf(args.seed)

    oos_label = len(label_names)
    val_oos = load_clinc_oos_texts(
        os.path.join(args.clinc_dir, "validation-00000-of-00001.parquet"),
        args.seed,
        args.val_oos_count,
    )
    test_oos = load_clinc_oos_texts(
        os.path.join(args.clinc_dir, "test-00000-of-00001.parquet"),
        args.seed + 1,
        args.test_oos_count,
    )

    val = val_id + [{"text": t, "intent": "oos", "label": oos_label, "is_oos": True} for t in val_oos]
    test = test_id + [{"text": t, "intent": "oos", "label": oos_label, "is_oos": True} for t in test_oos]
    random.Random(args.seed).shuffle(val)
    random.Random(args.seed + 1).shuffle(test)

    splits = {"train": train_id, "val": val, "test": test}
    for split, data in splits.items():
        with open(os.path.join(args.out_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "intent_names.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"id_to_intent": {str(i): n for i, n in enumerate(label_names)}, "oos_label": oos_label},
            f,
            ensure_ascii=False,
            indent=2,
        )

    for split in ["train", "val", "test"]:
        oos_count = sum(1 for x in splits[split] if x["intent"] == "oos")
        print(f"{split}: total={len(splits[split])}, oos={oos_count}")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
