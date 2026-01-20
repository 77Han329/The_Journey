from typing import Iterable, Iterator
from train_utils import load_vocab, load_merges


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab                      # dict[int, bytes]
        self.merges = merges                    # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens or []

        # decode 用的反向表（非常有用）
        self.id_to_bytes = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        vocab = load_vocab(vocab_path)
        merges = load_merges(merges_path)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("encode() not implemented yet")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")