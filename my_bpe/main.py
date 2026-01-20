
from train_utils import train_bpe , save_vocab, save_merges,save_vocab_byte_idx,load_vocab,load_merges
import argparse
from tokenizer import Tokenizer


def test_tokenizer():
    
    tokenizer = Tokenizer.from_files(vocab_path="vocab.json",merges_path="merges.txt",special_tokens=["<|endoftext|>"])
    
    
    example_text = "I'm a helpfull assistant, I am happy to see you and provide you some helps."
    
    print(tokenizer.encode(example_text))
    
    if tokenizer.decode(tokenizer.encode(example_text)):
        print("Yes!!")
    else:
        print("NO!!")




def main(data_path, vocab_size):
    # print("Start Training BPE")
    # vocab, merges = train_bpe(input_path=data_path, vocab_size=vocab_size,special_tokens=["<|endoftext|>"])
    # print("End of Training")
    # save_vocab(vocab=vocab,path="vocab.json")
    
    # save_merges(merges=merges,path="merges.txt")

    test_tokenizer()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="roneneldan/TinyStories",
        help="dataset which we want to perform bpe training"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1000,
        help="vocab size for BPE"
    )

    args = parser.parse_args()

    data_path = args.data_path
    vocab_size = args.vocab_size

    main(data_path, vocab_size)