
from train_utils import train_bpe , save_vocab, save_merges,save_vocab_byte_idx,load_vocab,load_merges

# def main():
#     vocab, merges = train_bpe(
#         input_path="roneneldan/TinyStories",
#         vocab_size=400,
#         special_tokens=["<|endoftext|>"]
#     )

#     save_vocab(vocab, "vocab.json")
#     save_vocab_byte_idx(vocab, "vocab_byte_idx")
#     save_merges(merges, "merges.txt")


# if __name__ == "__main__":
#     main()

vocab, merges = train_bpe(
    "roneneldan/TinyStories",
    vocab_size=400,
    special_tokens=[]
)

save_vocab(vocab, "vocab.json")
save_merges(merges, "merges.txt")

vocab2 = load_vocab("vocab.json")
merges2 = load_merges("merges.txt")

assert vocab == vocab2
assert merges == merges2