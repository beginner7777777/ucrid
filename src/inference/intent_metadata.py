"""
Helpers for building intent metadata used by the UCRID cascade.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "for", "from", "get", "give", "have", "how", "i",
    "in", "is", "it", "me", "my", "of", "on", "please", "show", "tell", "the",
    "to", "what", "when", "where", "with", "you", "your",
}


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def normalize_intent_name(intent_name: str) -> str:
    """Convert dataset intent ids like `card_declined` into readable text."""
    return intent_name.replace("_", " ").replace("-", " ").strip()


def build_intent_definition(intent_name: str, examples: Iterable[str], max_keywords: int = 4) -> str:
    """
    Build a lightweight natural-language description from examples.
    This is a deterministic fallback when curated definitions are unavailable.
    """
    examples = list(examples)
    keywords = Counter()

    for example in examples[:20]:
        for token in _tokenize(example):
            if token in _STOPWORDS:
                continue
            keywords[token] += 1

    top_keywords = [token for token, _ in keywords.most_common(max_keywords)]
    readable_name = normalize_intent_name(intent_name)

    if top_keywords:
        return (
            f"Handles requests about {readable_name}. "
            f"Typical keywords: {', '.join(top_keywords)}."
        )

    if examples:
        return f"Handles requests about {readable_name}. Example: \"{examples[0]}\"."

    return f"Handles requests about {readable_name}."


def build_intent_metadata(
    examples: List,
    num_intents: int,
    oos_label: int,
) -> Tuple[List[str], Dict[str, str], Dict[str, List[str]], List[str], Dict[str, int]]:
    """
    Build intent names, heuristic descriptions, training pools, and label maps.
    """
    id_to_name: Dict[int, str] = {}
    train_examples_by_name: Dict[str, List[str]] = defaultdict(list)
    oos_pool: List[str] = []

    for example in examples:
        if getattr(example, "is_oos", False) or getattr(example, "label", None) == oos_label:
            oos_pool.append(example.text)
            continue

        label = int(example.label)
        intent_name = example.intent_name
        id_to_name[label] = intent_name
        train_examples_by_name[intent_name].append(example.text)

    intent_names = [id_to_name.get(label, f"intent_{label}") for label in range(num_intents)]
    intent_defs = {
        name: build_intent_definition(name, train_examples_by_name.get(name, []))
        for name in intent_names
    }
    intent_name_to_id = {name: idx for idx, name in enumerate(intent_names)}

    return intent_names, intent_defs, dict(train_examples_by_name), oos_pool, intent_name_to_id
