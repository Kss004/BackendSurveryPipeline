"""
Tiny, pure helpers that do not depend on pandas / FastAPI.
Keep them here so they can be unit-tested in isolation.
"""
from __future__ import annotations

import re
from typing import List


def round_to_nearest_10(n: int) -> int:
    """8 → 10, 12 → 10, 23 → 20 …"""
    return int(10 * round(n / 10))


def normalise_quote(text: str) -> str:
    """
    Lower-case, strip punctuation and extra spaces so we can
    de-duplicate semantically identical quotes.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text


def dedup_quotes(quotes: List[str], top_k: int = 3) -> List[str]:
    """
    Guarantees we never show the same sentence twice.
    Keeps original order and original casing.
    """
    seen: set[str] = set()
    out: List[str] = []
    for q in quotes:
        key = normalise_quote(q)
        if key not in seen:
            seen.add(key)
            out.append(q)
            if len(out) == top_k:
                break
    return out
