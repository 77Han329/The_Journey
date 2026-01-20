
from train_utils import train_bpe , save_vocab, save_merges,save_vocab_byte_idx,load_vocab,load_merges

def main():
    print("Training BPE...")
    vocab, merges = train_bpe(
        input_path="roneneldan/TinyStories",
        vocab_size=400,
        special_tokens=["<|endoftext|>"]
    )

    print("Saving vocab...")
    save_vocab(vocab, "vocab.json")
    # save_vocab_byte_idx(vocab, "vocab_byte_idx")
    save_merges(merges, "merges.txt")


if __name__ == "__main__":
    main()

