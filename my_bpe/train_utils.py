from datasets import load_dataset






def preprocessing(input_path:str) -> list[int]:
    
    if "TinyStories" in input_path:
        dataset = load_dataset("roneneldan/TinyStories")
    else: 
        raise ValueError("input_path need to be tiny stories!")
    
    input_ids = dataset["train"]["text"][:500]

    print(f"You are training BPE on {len(input_ids)} texts on Tiny Stroies dataset")
    print("="*80)

    input_ids = "".join(text for text in input_ids)

    print(f"Your current training data contains {len(input_ids)} chars")

    input_ids = list(input_ids.encode("utf-8"))

    len(input_ids)
    print(f"Your training data contains {len(input_ids)} bytes")
    
    return input_ids


def get_stats_on_inputs_ids(input_ids:list[int]) -> dict[tuple[int,int],int]:
    stats = {}
    
    for pair in zip(input_ids,input_ids[1:]):
        stats[pair] = stats.get(pair,0) + 1 
    
    return stats

def find_pair(stats:dict[tuple[int,int],int])->tuple[int,int]:
    
    pair = max(stats,key=stats.get)
    
    return pair

def merge(input_ids:list[int], merge_pair:tuple[int,int], idx:int):
    
    new_ids = []
    i = 0
    while i < len(input_ids):
        if i < len(input_ids) -1 and input_ids[i] == merge_pair[0] and input_ids[i+1] == merge_pair[1]:
            new_ids.append(idx)
            i+=2
        else:
            new_ids.append(input_ids[i])
            i+=1
    return new_ids 

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    ## 1. preprocessing input text
    input_ids = preprocessing(input_path=input_path)

    vocab = {i:bytes([i]) for i in range(256)}
    
    merges = []
     
    num_merge = vocab_size - 256
   
    
    
    
    for i in range(num_merge):
        current_stats = get_stats_on_inputs_ids(input_ids=input_ids)
        merge_pair = find_pair(current_stats)
        
        idx = i + 256
        
        print(f"merging {merge_pair} into {idx}")
        
        input_ids = merge(input_ids=input_ids,
                        merge_pair=merge_pair,
                        idx=i+256)
        
        p0, p1 = vocab[merge_pair[0]], vocab[merge_pair[1]]
        
        merges.append((p0,p1))
        vocab[idx] = vocab[merge_pair[0]] + vocab[merge_pair[1]]
        
    idx = 256 + num_merge
    for tok in special_tokens:
        vocab[idx] = tok.encode("utf-8")
        idx+=1
        
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


def load_merges(path: str) -> List[Tuple[bytes, bytes]]:
    """
    Load merges from file.
    """
    merges: List[Tuple[bytes, bytes]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            left_str, right_str = line.split("\t", 1)
            merges.append(
                (left_str.encode("latin-1"), right_str.encode("latin-1"))
            )
    return merges