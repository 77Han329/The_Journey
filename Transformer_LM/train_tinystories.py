"""
End-to-end training script for the TransformerLM model on TinyStories.

Highlights:
    * Creates TinyStories dataloaders (train + optional validation)
    * Uses the custom AdanW optimizer + gradient clipping
    * Supports checkpoint save / load and simple text decoding
    * Leaves the core model code untouched as requested
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Make local modules importable when running `python Transformer_LM/train_tinystories.py`
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_utils import (  # noqa: E402
    BytePairTokenizer,
    TokenizerConfig,
    TinyStoriesDataset,
    create_dataloader,
)
from function import cross_entropy  # noqa: E402
from nn import TransformerLM  # noqa: E402
from optimizer import AdanW  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_tokenizer_dir = SCRIPT_DIR.parent / "my_bpe"
    parser = argparse.ArgumentParser(description="Train TransformerLM on TinyStories.")

    # Model architecture knobs
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer depth.")
    parser.add_argument("--d_model", type=int, default=512, help="Model width.")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feedforward hidden size.")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads.")
    parser.add_argument("--seq_len", type=int, default=256, help="Context window.")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE base.")
    parser.add_argument("--norm_eps", type=float, default=1e-5, help="RMSNorm epsilon.")
    parser.add_argument(
        "--use_rope", action="store_true", help="Enable rotary embeddings in the MHA layers."
    )

    # Optimizer + training setup
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--betas", nargs=3, type=float, default=(0.98, 0.92, 0.99))
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None, help="torch device string.")

    # Tokenizer paths
    parser.add_argument(
        "--vocab_path",
        type=Path,
        default=default_tokenizer_dir / "vocab.json",
        help="Path to vocab.json exported by my_bpe.",
    )
    parser.add_argument(
        "--merges_path",
        type=Path,
        default=default_tokenizer_dir / "merges.txt",
        help="Path to merges.txt exported by my_bpe.",
    )
    parser.add_argument("--eot_token", type=str, default="<|endoftext|>")

    # Data options
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="roneneldan/TinyStories",
        help="HF dataset identifier (used when train/valid files are not given).",
    )
    parser.add_argument("--train_file", type=Path, default=None, help="Optional local train .txt.")
    parser.add_argument("--valid_file", type=Path, default=None, help="Optional local val .txt.")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--valid_split", type=str, default="validation")
    parser.add_argument("--max_train_samples", type=int, default=2000, help="truncate dataset.")
    parser.add_argument("--max_valid_samples", type=int, default=512)
    parser.add_argument("--skip_validation", action="store_true", help="Disable validation eval.")

    # Checkpointing
    parser.add_argument("--save_dir", type=Path, default=SCRIPT_DIR / "checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=500, help="Steps between checkpoints.")
    parser.add_argument("--resume_from", type=Path, default=None, help="Existing checkpoint path.")

    # Decoding preview
    parser.add_argument("--preview_prompt", type=str, default="Once upon a time")
    parser.add_argument("--preview_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)

    return parser.parse_args()


def load_texts(source: Optional[Path], dataset_name: str, split: str, max_samples: Optional[int]) -> list[str]:
    """
    Return a list of texts from either a local file/directory or a HF dataset split.
    """

    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        if path.is_file():
            return [path.read_text(encoding="utf-8")]

        texts: list[str] = []
        for idx, file_path in enumerate(sorted(path.glob("*.txt"))):
            texts.append(file_path.read_text(encoding="utf-8"))
            if max_samples is not None and idx + 1 >= max_samples:
                break
        return texts

    # Fallback: grab split from datasets
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "datasets library is required to download TinyStories. Install via `pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
    return dataset["text"]


def prepare_dataloader(
    tokenizer: BytePairTokenizer,
    args: argparse.Namespace,
    is_train: bool,
) -> torch.utils.data.DataLoader:
    source = args.train_file if is_train else args.valid_file
    split = args.train_split if is_train else args.valid_split
    max_samples = args.max_train_samples if is_train else args.max_valid_samples

    texts = load_texts(source, args.dataset_name, split, max_samples)
    dataset = TinyStoriesDataset.from_texts(tokenizer, texts, args.seq_len)
    return create_dataloader(dataset, args.batch_size, shuffle=is_train)


def build_tokenizer(args: argparse.Namespace) -> BytePairTokenizer:
    cfg = TokenizerConfig(
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        eot_token=args.eot_token,
    )
    return BytePairTokenizer(cfg)


def instantiate_model(tokenizer: BytePairTokenizer, args: argparse.Namespace) -> TransformerLM:
    d_k = args.d_model // args.num_heads
    model = TransformerLM(
        vocab_size=len(tokenizer.id_to_bytes),
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_k=d_k,
        d_ff=args.d_ff,
        num_head=args.num_heads,
        theta=args.rope_theta,
        eps=args.norm_eps,
        use_rope=args.use_rope,
        norm="prenorm",
    )
    return model


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizer: AdanW,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": vars(args),
    }
    torch.save(payload, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(path: Path, model: TransformerLM, optimizer: Optional[AdanW] = None, device: torch.device = torch.device("cpu")):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    print(f"[checkpoint] Restored from {path}")
    return start_epoch, global_step


def evaluate(model: TransformerLM, loader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            batch_tokens = inputs.size(0) * inputs.size(1)
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: BytePairTokenizer,
    prompt: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    """
    Simple autoregressive decoding used after training to sanity-check the model.
    """

    device = next(model.parameters()).device
    token_ids = tokenizer.encode(prompt, add_eot=False)
    context = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        context_window = context[:, -seq_len:]
        logits = model(context_window)
        next_token_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None:
            values, _ = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
            min_values = values[:, -1].unsqueeze(-1)
            next_token_logits = next_token_logits.masked_fill(next_token_logits < min_values, -float("inf"))

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)

    return tokenizer.decode(context[0].tolist())


def main():
    args = parse_args()
    tokenizer = build_tokenizer(args)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[launch] Using device: {device}")
    train_loader = prepare_dataloader(tokenizer, args, is_train=True)
    valid_loader = None if args.skip_validation else prepare_dataloader(tokenizer, args, is_train=False)

    model = instantiate_model(tokenizer, args).to(device)
    optimizer = AdanW(
        model.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if args.resume_from is not None:
        start_epoch, global_step = load_checkpoint(args.resume_from, model, optimizer, device)

    best_val_loss = float("inf")
    grad_accum = 1

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = cross_entropy(logits, targets) / grad_accum
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if (batch_idx + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} loss={loss.item():.4f} "
                    f"ppl={math.exp(loss.item()):.2f}"
                )

            if global_step % args.ckpt_interval == 0:
                ckpt_path = args.save_dir / f"step_{global_step}.pt"
                save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, args)

        elapsed = time.time() - epoch_start
        print(f"[epoch] {epoch} finished in {elapsed:.1f}s")

        if valid_loader is not None:
            val_loss, val_ppl = evaluate(model, valid_loader, device)
            print(f"[valid] loss={val_loss:.4f} ppl={val_ppl:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, global_step, args)

    final_ckpt = args.save_dir / "last.pt"
    save_checkpoint(final_ckpt, model, optimizer, args.epochs - 1, global_step, args)

    preview = generate_text(
        model,
        tokenizer,
        prompt=args.preview_prompt,
        seq_len=args.seq_len,
        max_new_tokens=args.preview_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("=" * 80)
    print("Sampled text:")
    print(preview)
    print("=" * 80)


if __name__ == "__main__":
    main()
