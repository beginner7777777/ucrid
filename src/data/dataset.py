"""
Data Loading Module for CLINC150 Dataset
Handles data loading, preprocessing, and batching
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


def _load_intent_map(data_dir: str) -> Dict[int, str]:
    """Load intent id->name mapping from intents parquet (if available)."""
    intents_dir = os.path.join(data_dir, "..", "intents")
    for fname in os.listdir(intents_dir) if os.path.isdir(intents_dir) else []:
        if fname.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(os.path.join(intents_dir, fname))
            return dict(zip(df["id"].astype(int), df["name"]))
    return {}


@dataclass
class IntentExample:
    """Single intent recognition example"""
    text: str
    label: int
    intent_name: str
    is_oos: bool


class CLINC150Dataset(Dataset):
    """CLINC150 Dataset for Intent Recognition and OOS Detection"""

    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 64,
        oos_label: int = 150
    ):
        """
        Initialize CLINC150 dataset

        Args:
            data_path: Path to JSON data file
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
            oos_label: Label ID for OOS class
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.oos_label = oos_label

        # Load data
        self.examples = self._load_data()
        self.intent_names = self._get_intent_names()

    def _load_data(self) -> List[IntentExample]:
        """Load data from JSON or parquet file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        if self.data_path.endswith(".parquet"):
            return self._load_parquet(self.data_path)

        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        examples = []
        for item in data:
            text = item['text']
            intent_name = item['intent']

            if intent_name == 'oos':
                label = self.oos_label
                is_oos = True
            else:
                label = item.get('label', 0)
                is_oos = False

            examples.append(IntentExample(
                text=text,
                label=label,
                intent_name=intent_name,
                is_oos=is_oos
            ))

        return examples

    def _load_parquet(self, path: str) -> List[IntentExample]:
        """Load data from HuggingFace-style parquet (columns: utterance, label)."""
        import pandas as pd

        df = pd.read_parquet(path)
        id_to_name = _load_intent_map(os.path.dirname(path))

        examples = []
        for _, row in df.iterrows():
            text = row["utterance"]
            if pd.isna(row["label"]):
                label = self.oos_label
                intent_name = "oos"
                is_oos = True
            else:
                label = int(row["label"])
                intent_name = id_to_name.get(label, str(label))
                is_oos = False
            examples.append(IntentExample(
                text=text,
                label=label,
                intent_name=intent_name,
                is_oos=is_oos
            ))
        return examples

    def _get_intent_names(self) -> List[str]:
        """Get unique intent names"""
        intent_names = sorted(set(ex.intent_name for ex in self.examples))
        return intent_names

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example"""
        example = self.examples[idx]

        # Tokenize text
        encoding = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(example.label, dtype=torch.long),
            'is_oos': torch.tensor(example.is_oos, dtype=torch.bool),
            'text': example.text,
            'intent_name': example.intent_name
        }

    def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of intents in dataset"""
        distribution = {}
        for example in self.examples:
            intent = example.intent_name
            distribution[intent] = distribution.get(intent, 0) + 1
        return distribution

    def get_oos_ratio(self) -> float:
        """Get ratio of OOS samples"""
        num_oos = sum(1 for ex in self.examples if ex.is_oos)
        return num_oos / len(self.examples)


def create_dataloader(
    dataset: CLINC150Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader for dataset

    Args:
        dataset: CLINC150Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def load_clinc150_data(
    data_dir: str,
    tokenizer: BertTokenizer,
    max_length: int = 64,
    batch_size: int = 32,
    num_workers: int = 4,
    oos_label: int = 150,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CLINC150 train, validation, and test datasets

    Args:
        data_dir: Directory containing data files
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load datasets — support both JSON and parquet layouts
    def _resolve_path(data_dir, stem, parquet_name):
        json_path = os.path.join(data_dir, f"{stem}.json")
        if os.path.exists(json_path):
            return json_path
        parquet_path = os.path.join(data_dir, "data", parquet_name)
        if os.path.exists(parquet_path):
            return parquet_path
        raise FileNotFoundError(
            f"No data file found for '{stem}' in {data_dir} "
            f"(tried {json_path} and {parquet_path})"
        )

    train_dataset = CLINC150Dataset(
        _resolve_path(data_dir, "train", "train-00000-of-00001.parquet"),
        tokenizer,
        max_length,
        oos_label,
    )
    val_dataset = CLINC150Dataset(
        _resolve_path(data_dir, "val", "validation-00000-of-00001.parquet"),
        tokenizer,
        max_length,
        oos_label,
    )
    test_dataset = CLINC150Dataset(
        _resolve_path(data_dir, "test", "test-00000-of-00001.parquet"),
        tokenizer,
        max_length,
        oos_label,
    )

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = create_dataloader(
        val_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = create_dataloader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    # Print statistics
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train OOS ratio: {train_dataset.get_oos_ratio():.2%}")
    print(f"Val OOS ratio: {val_dataset.get_oos_ratio():.2%}")
    print(f"Test OOS ratio: {test_dataset.get_oos_ratio():.2%}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Test single dataset
    dataset = CLINC150Dataset(
        'data/clinc150/train.json',
        tokenizer,
        max_length=64
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Intent distribution: {dataset.get_intent_distribution()}")

    # Test single example
    example = dataset[0]
    print(f"\nExample:")
    print(f"Text: {example['text']}")
    print(f"Label: {example['labels']}")
    print(f"Is OOS: {example['is_oos']}")
