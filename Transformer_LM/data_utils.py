"""Utilities for loading TinyStories data and converting it into LM batches.

This module intentionally leaves the core model files untouched.  It provides
three building blocks that the training script can reuse:
1) ``BytePairTokenizer`` for loading the vocab/merges that were trained via
   ``my_bpe``.
2) ``TinyStoriesDataset`` that chunks long token streams into (input, target)
   pairs.
3) ``create_dataloader`` convenience helper to build ``DataLoader`` objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Dataset, DataLoader


def _load_vocab(path: Path) -> dict[int, bytes]:
    """Load the ``token_id -> bytes`` mapping that ``my_bpe`` exported."""

    with path.open("r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    return {int(idx): value.encode("latin-1") for idx, value in vocab_json.items()}


def _load_merges(path: Path) -> list[tuple[bytes, bytes]]:
    """Load merge rules; each line stores ``left<TAB>right``."""

    merges: list[tuple[bytes, bytes]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                left, right = line.split("\t", 1)
            else:
                left, right = line.split(" ", 1)
            merges.append((left.encode("latin-1"), right.encode("latin-1")))
    return merges


@dataclass
class TokenizerConfig:
    vocab_path: Path
    merges_path: Path
    eot_token: str = "<|endoftext|>"


class BytePairTokenizer:
    """Minimal byte-pair tokenizer compatible with ``my_bpe`` artifacts."""

    def __init__(self, config: TokenizerConfig):
        self.vocab_path = config.vocab_path
        self.merges_path = config.merges_path
        self.id_to_bytes = _load_vocab(self.vocab_path)
        self.bytes_to_id = {value: idx for idx, value in self.id_to_bytes.items()}
        self.merges = _load_merges(self.merges_path)
        self.eot_token = config.eot_token
        self.eot_id = self.token_to_id(self.eot_token) if self.eot_token else None

    def token_to_id(self, token: str) -> int:
        token_bytes = token.encode("utf-8")
        if token_bytes not in self.bytes_to_id:
            raise ValueError(f"Token {token!r} is missing from vocab {self.vocab_path}")
        return self.bytes_to_id[token_bytes]

    def encode(self, text: str, add_eot: bool = True) -> list[int]:
        """Encode raw text into token ids using BPE merges."""

        tokens = [bytes([b]) for b in text.encode("utf-8")]
        for left, right in self.merges:
            merged: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == left and tokens[i + 1] == right:
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        ids = [self.bytes_to_id[token] for token in tokens]
        if add_eot and self.eot_id is not None:
            ids.append(self.eot_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        byte_stream = b"".join(self.id_to_bytes[int(idx)] for idx in ids)
        return byte_stream.decode("utf-8", errors="replace")


class TinyStoriesDataset(Dataset):
    """Dataset that yields (input, target) blocks from TinyStories text."""

    def __init__(self, token_buffer: torch.Tensor, seq_len: int):
        if token_buffer.numel() <= seq_len:
            raise ValueError("Dataset needs to contain more tokens than seq_len")
        self.seq_len = seq_len
        self.tokens = token_buffer

    @classmethod
    def from_files(cls, tokenizer: BytePairTokenizer, files: Sequence[Path], seq_len: int):
        texts = [path.read_text(encoding="utf-8") for path in files]
        return cls(cls._encode_collection(tokenizer, texts), seq_len)

    @classmethod
    def from_texts(cls, tokenizer: BytePairTokenizer, texts: Sequence[str], seq_len: int):
        return cls(cls._encode_collection(tokenizer, texts), seq_len)

    @staticmethod
    def _encode_collection(tokenizer: BytePairTokenizer, texts: Sequence[str]) -> torch.Tensor:
        token_buffer: List[int] = []
        for text in texts:
            token_buffer.extend(tokenizer.encode(text, add_eot=True))
        return torch.tensor(token_buffer, dtype=torch.long)

    def __len__(self) -> int:
        return self.tokens.numel() - self.seq_len - 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]
        return x, y


def create_dataloader(dataset: TinyStoriesDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
    )
