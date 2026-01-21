from datasets import load_dataset
import regex as re  # 注意：用 regex 库，支持 Unicode 类别 \p{L} \p{N}

# GPT-2 风格的预分词正则表达式
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    re.IGNORECASE
)


def pretokenize(text: str) -> list[list[int]]:
    """
    将文本预分词成 chunks，每个 chunk 转成字节列表。
    BPE 只在 chunk 内部合并，不会跨 chunk 边界。
    """
    chunks = GPT2_PAT.findall(text)
    return [list(chunk.encode("utf-8")) for chunk in chunks]


def preprocessing(input_path: str) -> list[list[int]]:
    
    if "TinyStories" in input_path:
        dataset = load_dataset("roneneldan/TinyStories")
    else: 
        raise ValueError("input_path need to be tiny stories!")
    
    texts = dataset["train"]["text"][:500]

    print(f"You are training BPE on {len(texts)} texts on Tiny Stories dataset")
    print("="*80)

    full_text = "".join(texts)
    print(f"Your current training data contains {len(full_text)} chars")

    # 预分词：将文本分成 chunks，每个 chunk 是独立的字节列表
    chunks = pretokenize(full_text)
    total_bytes = sum(len(chunk) for chunk in chunks)
    
    print(f"Your training data contains {total_bytes} bytes in {len(chunks)} chunks")
    
    return chunks  # 返回 list[list[int]]


def get_stats_on_inputs_ids(chunks: list[list[int]]) -> dict[tuple[int,int], int]:
    """统计所有 chunks 中的字节对频率，但不跨 chunk 边界"""
    stats = {}
    
    for chunk in chunks:
        # 只在每个 chunk 内部统计相邻对
        for pair in zip(chunk, chunk[1:]):
            stats[pair] = stats.get(pair, 0) + 1 
    
    return stats

def find_pair(stats:dict[tuple[int,int],int])->tuple[int,int]:
    
    pair = max(stats,key=stats.get)
    
    return pair

def merge_chunk(chunk: list[int], merge_pair: tuple[int,int], idx: int) -> list[int]:
    """对单个 chunk 应用 merge"""
    new_ids = []
    i = 0
    while i < len(chunk):
        if i < len(chunk) - 1 and chunk[i] == merge_pair[0] and chunk[i+1] == merge_pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(chunk[i])
            i += 1
    return new_ids


def merge(chunks: list[list[int]], merge_pair: tuple[int,int], idx: int) -> list[list[int]]:
    """对所有 chunks 应用 merge，每个 chunk 独立处理"""
    return [merge_chunk(chunk, merge_pair, idx) for chunk in chunks] 

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    ## 1. preprocessing: 返回 list[list[int]]，每个 chunk 是独立的
    chunks = preprocessing(input_path=input_path)

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    num_merge = vocab_size - 256

    for i in range(num_merge):
        # 统计所有 chunks 中的字节对频率（不跨 chunk 边界）
        current_stats = get_stats_on_inputs_ids(chunks)
        
        if not current_stats:
            print(f"No more pairs to merge at iteration {i}")
            break
            
        merge_pair = find_pair(current_stats)
        idx = i + 256
        
        # 对所有 chunks 应用这个 merge
        chunks = merge(chunks, merge_pair, idx)
        
        p0, p1 = vocab[merge_pair[0]], vocab[merge_pair[1]]
        merges.append((p0, p1))
        vocab[idx] = p0 + p1
        
    # 添加特殊 token
    idx = 256 + len(merges)
    for tok in special_tokens:
        vocab[idx] = tok.encode("utf-8")
        idx += 1
        
    return vocab, merges


import json
from typing import Dict, List, Tuple

# =========================================================
# vocab: token_id (int) <-> bytes
# =========================================================

def save_vocab(vocab: Dict[int, bytes], path: str) -> None:
    """
    Save vocab as JSON:
    { "token_id": "latin-1-decoded-bytes" }
    """
    vocab_json = {
        str(token_id): token_bytes.decode("latin-1")
        for token_id, token_bytes in vocab.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> Dict[int, bytes]:
    """
    Load vocab from JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    return {
        int(token_id): token_str.encode("latin-1")
        for token_id, token_str in vocab_json.items()
    }


def save_vocab_byte_idx(vocab: Dict[int, bytes], path: str) -> None:
    """
    (Optional helper)
    Save reverse vocab:
    { "latin-1-decoded-bytes": token_id }
    """
    vocab_json = {
        token_bytes.decode("latin-1"): token_id
        for token_id, token_bytes in vocab.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)


# =========================================================
# merges: ordered list of (bytes, bytes)
# =========================================================

def save_merges(merges: List[Tuple[bytes, bytes]], path: str) -> None:
    """
    Save merges as a line-based rule file.
    Each line: <left>\t<right>
    (TAB-separated, NOT space-separated)
    """
    with open(path, "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(
                f"{left.decode('latin-1')}\t{right.decode('latin-1')}\n"
            )


def load_merges(path):
    merges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            if "\t" in line:
                left_str, right_str = line.split("\t", 1)
            else:
                # 兼容旧格式（空格）
                left_str, right_str = line.split(" ", 1)

            merges.append(
                (left_str.encode("latin-1"), right_str.encode("latin-1"))
            )
    return merges