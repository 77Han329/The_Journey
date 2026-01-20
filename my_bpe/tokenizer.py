from typing import Iterable, Iterator
from train_utils import load_vocab, load_merges,get_stats_on_inputs_ids


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab                      # dict[int, bytes]
        self.merges = merges                    # list[tuple[bytes, bytes]]
        self.special_tokens = special_tokens or []

        # decode 用的反向表（非常有用）
        self.vocab_byte_idx = {v: k for k, v in vocab.items()} # dict[bytes,int]

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        vocab = load_vocab(vocab_path)
        merges = load_merges(merges_path)
        return cls(vocab, merges, special_tokens)

    # def encode(self, text: str) -> list[int]:
    #     input_ids = [bytes([b]) for b in text.encode("utf-8")]
        
    #     for (p0,p1) in self.merges:
            
    #         i = 0
            
    #         new_input_ids = []
            
    #         while i < len(input_ids):
    #             if i < len(input_ids)-1 and input_ids[i] == p0 and input_ids[i+1] == p1:
    #                 new_input_ids.append(p0+p1)
    #                 i+=2
    #             else:
    #                 new_input_ids.append(input_ids[i])
    #                 i+=1
                    
    #         input_ids = new_input_ids
        
        
    #         input_ids = [self.vocab_byte_idx[bytes] for bytes in input_ids]
            
    #         return input_ids
            
            
    def encode(self, text: str) -> list[int]:
    # --------------------------------------------------
    # 1. text -> UTF-8 bytes -> list[int] (0–255)
    # --------------------------------------------------
        ids = list(text.encode("utf-8"))

        # --------------------------------------------------
        # 2. 按 merges 顺序逐条应用
        #    第 i 条 merge -> 新 token id = 256 + i
        # --------------------------------------------------
        for merge_idx, (p0, p1) in enumerate(self.merges):
            new_token_id = 256 + merge_idx

            i = 0
            new_ids = []

            while i < len(ids):
                # 判断：相邻两个 token 的 bytes 是否匹配这条 merge
                if (
                    i + 1 < len(ids)
                    and self.vocab[ids[i]] == p0
                    and self.vocab[ids[i + 1]] == p1
                ):
                    # 匹配成功：用新的 token id 替换
                    new_ids.append(new_token_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            ids = new_ids

        return ids

 
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        
        ## 思路
        ## 1. 我们收到了一个 list[int]
        ## 2. 第一步要把list 里面的int 变成byte的样子，然后合并到一个byte string里面去
        ##      - list 里面的int 要变成byte 需要借助训练得到的vocab， vocab 里面存储了所有int -> byte 的映射
        ##      - 这里的关键是要想清楚，如果是0-255 那没问题，如果是别的比如说300，当时存在vocab 里面的300 也是由前256byte 拼接组成
        ids = b"".join(self.vocab[idx] for idx in ids)
        
        output_text = ids.decode("utf-8",errors="replace")
        
        return output_text
            